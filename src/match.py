"""
match.py
--------
Scores and filters scanned jobs against the target role config before any
LLM tokens get spent tailoring them. Deliberately simple (keyword
overlap, not embeddings) so it's transparent and free to run on every job —
tune the keyword lists in config.yaml rather than the code.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Iterable, Optional

from src.ats_score import requirements, score_resume
from src.scan_jobs import Job

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{1,}", text.lower()))


def score_job(
    job: Job,
    must_have: Iterable[str],
    nice_to_have: Iterable[str],
    exclude: Iterable[str],
    exclude_description: Iterable[str] = (),
) -> Optional[float]:
    """
    Returns None if the job should be dropped outright, else a 0-1 relevance score.

    IMPORTANT: `exclude` is matched against the TITLE only. Seniority/role words
    ("senior", "staff", "manager", "solutions", "sales", ...) appear constantly
    inside otherwise-fine descriptions ("report to a senior engineer", "our
    solutions", "sales team"), so matching them against the description drops
    almost every real posting. You exclude a "Senior Engineer" by its title, not
    because its description says "senior".

    `exclude_description` is the separate list for things that are genuinely
    description-level requirements, e.g. "5+ years", "security clearance".
    """
    title_lc = job.title.lower()
    # Word-boundary match (not substring): so "sr" drops "Sr. Engineer" without
    # also killing "SRE", and "lead" doesn't nuke "Leadership"/"misleading".
    for term in exclude:
        if re.search(rf"\b{re.escape(term.lower())}\b", title_lc):
            return None

    if exclude_description:
        desc_lc = (job.description or "").lower()
        for term in exclude_description:
            if term.lower() in desc_lc:
                return None

    haystack = _tokenize(f"{job.title} {job.description}")

    must_have = list(must_have)
    if must_have:
        hits = sum(1 for term in must_have if term.lower() in haystack)
        if hits == 0:
            return None

    nice_to_have = list(nice_to_have)
    if not nice_to_have:
        return 1.0
    hits = sum(1 for term in nice_to_have if term.lower() in haystack)
    return round(hits / len(nice_to_have), 3)


def is_phd_only_title(title: str) -> bool:
    """True when a title targets PhDs with no MS/BS alternative (e.g. '2027
    Summer Intern, PhD, Software Engineer'). 'MS/PhD', 'BS/MS/PhD' and titles
    with no PhD mention are NOT phd-only -- an MS candidate is eligible for those."""
    t = (title or "").lower()
    if not re.search(r"\bph\.?\s?d\b", t):
        return False
    # A co-listed lower degree means the role is open to non-PhDs too.
    if re.search(r"\b(m\.?s|ms|master|masters|b\.?s|bs|bachelor|bachelors|undergrad)\b", t):
        return False
    return True


def is_undergrad_only_title(title: str) -> bool:
    """True when a title targets undergraduates with no graduate alternative
    (e.g. 'Campus Undergraduate Full-Time Engineer - 2027'). Titles that also
    name a master's/graduate/PhD track ('BS/MS', 'Undergraduate or Graduate')
    are NOT undergrad-only -- an MS candidate is eligible for those."""
    t = (title or "").lower()
    if not re.search(r"\b(undergrad|undergraduate|undergraduates|bachelor|bachelor'?s|b\.s\.?|bs)\b", t):
        return False
    if re.search(r"\b(m\.?s|ms|master|masters|master'?s|graduate|graduates|grad|ph\.?\s?d|mba)\b", t):
        return False
    return True


# Phrases that state a GRADUATION-DATE requirement. Deliberately excludes bare
# "graduate"/"new grad" -- those name the role ("2027 New Grad Program"), and the
# year near them is usually a START date, which a Dec-2026 grad is eligible for.
_GRAD_PHRASE = re.compile(
    r"graduating|graduation|class of|expected to graduate|"
    r"graduate (?:in|between|by|before|after|no later than|from)",
    re.I,
)
_YEAR = re.compile(r"\b(20\d{2})\b")
_LOWER_BOUND = re.compile(r"or later|or after|and later|and after|onward|or beyond|or more recently", re.I)
_UPPER_BOUND = re.compile(r"\b(?:by|before|prior to|no later than)\b", re.I)


def graduation_window(description: str) -> tuple[int, int] | None:
    """The (earliest, latest) graduation year a posting accepts, or None if it
    states no graduation-date requirement. Reads the clause after each
    graduation phrase (to the end of the sentence):
      "graduating fall 2027 or spring 2028"   -> (2027, 2028)
      "graduation between Dec 2026 and 2027"  -> (2026, 2027)
      "graduating in 2027 or later"           -> (2027, 9999)
      "must graduate by June 2027"            -> (0, 2027)
    Heuristic by nature -- postings phrase this many ways -- so it only fires on
    explicit graduation-date language and returns None when unsure."""
    text = description or ""
    lo, hi, found = 9999, 0, False
    for m in _GRAD_PHRASE.finditer(text):
        clause = re.split(r"[.\n;]", text[m.start():m.start() + 160], maxsplit=1)[0]
        years = [int(y) for y in _YEAR.findall(clause)]
        if not years:
            continue
        found = True
        if _LOWER_BOUND.search(clause):
            lo, hi = min(lo, min(years)), 9999
        elif _UPPER_BOUND.search(clause):
            lo, hi = 0, max(hi, max(years))
        else:
            lo, hi = min(lo, min(years)), max(hi, max(years))
    return (lo, hi) if found else None


def graduation_mismatch(description: str, candidate_year: int | None) -> tuple[int, int] | None:
    """The posting's (lo, hi) graduation window if the candidate falls OUTSIDE
    it; None if eligible, no requirement is stated, or the year is unknown."""
    if not candidate_year:
        return None
    window = graduation_window(description)
    if window and not (window[0] <= candidate_year <= window[1]):
        return window
    return None


_INTERN_TITLE = re.compile(r"\b(?:intern|internship|co-?op|apprentice(?:ship)?)\b", re.I)
# "you will return to school after the internship" -- open only to students who
# are still enrolled once it ends.
_STILL_ENROLLED = re.compile(
    r"return(?:ing)? to (?:school|campus|(?:your |an? )?(?:academic|degree|university) program)"
    r"|(?:still|remain|continue to be) enrolled (?:after|following|upon)",
    re.I,
)


# A term / enrollment reference with a year: "Summer 2027", "2027 summer",
# "enrolled ... through Spring 2027", "graduating in May 2027".
_TERM_YEAR = re.compile(
    r"\b(?:spring|summer|fall|autumn|winter)\s+(?:of\s+|term\s+)?(20\d{2})\b"
    r"|\b(20\d{2})\s+(?:spring|summer|fall|autumn|winter)\b",
    re.I,
)
# Enrollment / graduation-date wording. Not bare "new grad(uate)" -- that names
# a start date, which a December grad is eligible for.
_ENROLL_CLAUSE = re.compile(r"enroll|pursuing|\bstudents?\b|" + _GRAD_PHRASE.pattern, re.I)


def internship_after_graduation(title: str, description: str, candidate_year: int | None) -> str | None:
    """Why an internship is out of reach for someone graduating in December of
    candidate_year, or None. Internships are for current students, so drop one
    that needs you enrolled AFTER December candidate_year:
      - its term year in the title is later ("2027 Summer Intern")
      - the description puts the term later ("runs Summer 2027")
      - an enrollment/graduation clause names a later year ("enrolled through
        Spring 2027", "graduating in May 2027")
      - it requires returning to school afterwards
    Years with no term or enrollment context ("founded in 2027") are ignored.
    Non-intern titles always return None."""
    if not candidate_year or not _INTERN_TITLE.search(title or ""):
        return None
    years = [int(y) for y in _YEAR.findall(title or "")]
    if years and min(years) > candidate_year:
        return f"internship term {min(years)} is after you graduate"
    text = description or ""
    terms = [int(a or b) for a, b in _TERM_YEAR.findall(text)]
    if terms and min(terms) > candidate_year:
        return f"internship term {min(terms)} is after you graduate"
    for clause in re.split(r"[.\n;]", text):
        if _ENROLL_CLAUSE.search(clause):
            ys = [int(y) for y in _YEAR.findall(clause)]
            later = [y for y in ys if y > candidate_year]
            # "between December 2026 and 2027" includes you -- only a clause
            # whose every year is past your graduation rules you out.
            if later and len(later) == len(ys):
                return f"requires being enrolled in {min(later)}"
    if _STILL_ENROLLED.search(text):
        return "requires returning to school after the internship"
    return None


# ---- experience requirement -----------------------------------------------------
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15}
_N = r"\b(\d{1,2}|" + "|".join(_NUM_WORDS) + r")\b"
# "3+ years", "three (3) years", "3-5 years", "at least 3 years", "3 or more years".
# Excludes adjectival "4-year degree" (the hyphen form never matches) and "years old/ago".
_YEARS_RE = re.compile(
    r"(?:at least|minimum(?: of)?|min\.?|over|more than)?\s*" + _N +
    r"\s*(?:\(\s*\d{1,2}\s*\))?\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*" + _N + r"\s*)?\+?\s*"
    r"(?:or more\s+)?(?:years?|yrs?)\b(?!\s*(?:old|ago|degree|program))",
    re.I,
)
_SOFT = re.compile(r"prefer|nice to have|nice-to-have|\bbonus\b|a plus|\bideally\b", re.I)
# A bare "2 years" is a requirement only when it reads like one: "2 years of
# hands-on work", "2 years in software development", "2 years working as ...".
# "+", "or more", "at least", "minimum" already mark one ("2+ years building
# production systems"). Never counted: time spans ("in the last 2 years",
# "within 2 years", "a 2-year program").
_EXP_AFTER = re.compile(
    r"^\W*(?:of\b|in\b|as\b|with\b|working|work\b|building|developing|designing|"
    r"programming|coding|hands-on|professional|industry|relevant|post-?grad|full-?time)", re.I)
_SPAN_BEFORE = re.compile(r"(?:within|next|past|last|first|every|per|after|for the)\s*$", re.I)
_STRONG = re.compile(r"\+|or more|at least|minimum|\bmin\b|over|more than", re.I)


def _num(tok: str) -> int:
    return int(tok) if tok.isdigit() else _NUM_WORDS[tok.lower()]


_PHD = re.compile(r"\b(?:ph\.?\s?d|doctorate|doctoral degree)\b", re.I)
_PHD_MUST = re.compile(r"\b(?:required|requires?|must (?:have|hold|be)|minimum qualifications?)\b", re.I)
# A lower degree or experience offered as an alternative keeps an MS candidate eligible.
_PHD_ALT = re.compile(r"\b(?:master'?s?|m\.?s\.?|bachelor'?s?|b\.?s\.?|or equivalent|equivalent (?:practical |industry |research )?experience|"
                      r"pursuing|candidate|enrolled)\b", re.I)


def phd_required(description: str) -> bool:
    """True when the posting states a PhD as a hard requirement with no lower
    degree or experience alternative in the same sentence: "PhD in Computer
    Science required". Skips "PhD preferred", "MS or PhD", "PhD or equivalent
    experience" and PhD-student internships. Errs toward keeping a posting --
    a bare "PhD in ML" bullet with no requirement word is not counted."""
    text = re.sub(r"<[^>]+>", " ", html.unescape(html.unescape(description or "")))
    for sentence in re.split(r"(?<=[.;!?])\s+|\n+|•", text):
        if _PHD.search(sentence) and _PHD_MUST.search(sentence) \
                and not _PHD_ALT.search(sentence) and not _SOFT.search(sentence):
            return True
    return False


def required_experience_years(description: str) -> int | None:
    """The smallest years-of-experience the posting REQUIRES, or None if it
    states none. Counts a mention that reads as a requirement ("2+ years of
    relevant industry experience", "minimum 2 years in software development",
    "2+ yrs building ..."), skips time spans ("in the last 2 years") and
    sentences marked preferred / a plus /
    bonus. Taking the minimum over all mentions keeps alternatives fair:
    "3+ years, or 1+ year with a Master's" -> 1."""
    text = description or ""
    found: list[int] = []
    for sentence in re.split(r"(?<=[.;!?])\s+|\n+|•", text):
        if _SOFT.search(sentence):
            continue
        for m in _YEARS_RE.finditer(sentence):
            after = sentence[m.end():m.end() + 80].lower()
            before = sentence[max(0, m.start() - 40):m.start()].lower()
            if _SPAN_BEFORE.search(before):
                continue
            if ("experience" in after or "experience" in before or "industry" in after
                    or _EXP_AFTER.search(after)
                    or (_STRONG.search(m.group(0)) and re.match(r"^\W*[a-z]", after))):
                found.append(_num(m.group(1)))
    return min(found) if found else None


def filter_and_rank(jobs: list[Job], matching_cfg: dict) -> list[tuple[Job, float]]:
    must_have = matching_cfg.get("must_have_keywords", [])
    nice_to_have = matching_cfg.get("nice_to_have_keywords", [])
    exclude = matching_cfg.get("exclude_keywords", [])
    exclude_description = matching_cfg.get("exclude_description_keywords", [])
    min_score = matching_cfg.get("min_score", 0.0)
    drop_phd_only = matching_cfg.get("exclude_phd_only_titles", False)
    drop_phd_required = matching_cfg.get("exclude_phd_required", True)
    drop_undergrad_only = matching_cfg.get("exclude_undergrad_only_titles", True)
    # A high keyword score doesn't make you ELIGIBLE: an internship for students
    # graduating 2027-2028 is out of reach for a Dec-2026 grad no matter how well
    # the skills match. Screen on the stated graduation window before scoring.
    grad_year = matching_cfg.get("graduation_year")
    drop_grad = matching_cfg.get("exclude_graduation_mismatch", True)
    # Same idea for experience: "three (3) years" is a hard gap for a candidate
    # whose professional experience is one short internship.
    max_req_years = matching_cfg.get("max_required_experience_years")
    drop_late_intern = matching_cfg.get("exclude_post_graduation_internships", True)

    scored: list[tuple[Job, float]] = []
    for job in jobs:
        if drop_phd_only and is_phd_only_title(job.title):
            continue
        if drop_phd_required and phd_required(job.description):
            logger.info("Degree screen: dropped %s @ %s (description requires a PhD)", job.title, job.company)
            continue
        if drop_undergrad_only and is_undergrad_only_title(job.title):
            logger.info("Degree screen: dropped %s @ %s (undergraduate-only title)", job.title, job.company)
            continue
        if drop_grad and grad_year:
            window = graduation_mismatch(job.description, grad_year)
            if window:
                lo, hi = window
                span = (f"{lo}+" if hi == 9999 else f"by {hi}" if lo == 0
                        else f"{lo}" if lo == hi else f"{lo}-{hi}")
                logger.info("Graduation screen: dropped %s @ %s (requires graduating %s; you: %s)",
                            job.title, job.company, span, grad_year)
                continue
        if drop_late_intern and grad_year:
            why = internship_after_graduation(job.title, job.description, grad_year)
            if why:
                logger.info("Internship screen: dropped %s @ %s (%s; you graduate %s)",
                            job.title, job.company, why, grad_year)
                continue
        if max_req_years is not None:
            req = required_experience_years(job.description)
            if req is not None and req > max_req_years:
                logger.info("Experience screen: dropped %s @ %s (requires %d+ years; limit %d)",
                            job.title, job.company, req, max_req_years)
                continue
        score = score_job(job, must_have, nice_to_have, exclude, exclude_description)
        if score is None or score < min_score:
            continue
        scored.append((job, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


# Title signals for the role tier. Core engineering/ML/data IC roles rank first;
# business / management / customer-facing roles (which pure ATS keyword-matching
# surfaces because they name skills) get demoted or dropped.
_CORE_ROLE = (
    # software
    "software engineer", "backend", "frontend", "full stack", "full-stack", "developer",
    "sde", "swe", "programmer", "android", "ios", "mobile", "embedded", "firmware",
    "web engineer", "api engineer",
    # data / ML
    "data engineer", "data scientist", "data science", "data analyst", "analytics engineer",
    "machine learning", "deep learning", "ml engineer", "mlops", "ai engineer",
    "research engineer", "research scientist", "applied scientist",
    # AI/ML research (labs title these many ways); UX/market research is demoted below
    "researcher", "research associate", "research assistant", "research fellow", "research resident",
    "research intern", "ai research", "ml research", "member of technical staff", "computer vision",
    "nlp", "natural language", "reinforcement learning", "robotics", "computational biolog",
    "bioinformatic", "medical imaging", "ai scientist", "ml scientist",
    # devops / platform / infra / systems / reliability
    "devops", "sre", "site reliability", "reliability engineer", "cloud engineer",
    "infrastructure engineer", "platform engineer", "systems engineer", "systems software",
    "distributed systems", "network engineer", "performance engineer",
    # security / quality
    "security engineer", "qa engineer", "test engineer", "quality engineer",
    "software development engineer in test", "sdet",
)
_DEMOTE_ROLE = (
    "consultant", "strategy", "strategist", "specialist", "planner", "program manager",
    "product manager", "business analyst", "operations", "recruiter", "account",
    "sales", "field engineer", "solutions engineer", "solutions architect",
    "forward deployed", "designer", "evangelist", "advocate",
    "developer relations", "devrel", "ux research", "user research", "market research", "design research", "clinical research",
)


# IT support / help desk / sysadmin: a real CS-adjacent track, ranked below core
# SWE/ML/data. Checked BEFORE the demote list so "IT Support Specialist" isn't
# dropped as a business "specialist" role.
_IT_ROLE = (
    "it support", "it technician", "it specialist", "it analyst", "it assistant", "it associate",
    "it engineer", "it intern", "it administrator", "it operations", "help desk", "helpdesk",
    "service desk", "desktop support", "technical support", "end user support", "support technician",
    "computer support", "user support", "deskside",
)


def is_it_support_role(title: str) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", (title or "").lower()) for k in _IT_ROLE)


def role_tier(title: str) -> int:
    """2 = core SWE/ML/data IC, 1 = IT support or generic engineer/dev/scientist, 0 = neutral,
    -1 = business/management/customer-facing (checked first so 'Solutions
    Engineer' demotes despite the word 'engineer')."""
    t = (title or "").lower()
    if is_it_support_role(t):
        return 1
    if any(k in t for k in _DEMOTE_ROLE):
        return -1
    if any(k in t for k in _CORE_ROLE):
        return 2
    if any(g in t for g in ("engineer", "developer", "scientist", "programmer")):
        return 1
    return 0


def rerank_by_ats(
    ranked: list[tuple[Job, float]],
    master_resume: dict,
    min_ats: int = 0,
    min_jd_skills: int = 0,
    prefer_engineering: bool = False,
    min_ats_it: int | None = None,
) -> list[tuple[Job, float]]:
    """
    Re-order role-filtered jobs by how well the master resume matches each
    posting -- scoring the master directly is a close, slightly conservative
    proxy for the tailored resume's ATS (the tailored one carries the same skill
    set, often surfacing a bit more). So the jobs generated first are the ones
    you'll score highest on.

    min_jd_skills drops postings that name too few skills to be a substantive
    technical role (those trivially hit 100% and are usually non-tech).
    min_ats drops postings you match below that %. Both default to 0 (keep all,
    just re-sort). Ties break on the original keyword-relevance score.

    prefer_engineering: drop business/customer-facing roles and rank genuine
    SWE/ML/data engineering roles first (by title), then by ATS within each tier.

    min_ats_it: a separate, lower cutoff for IT support / help-desk titles. Their
    postings list admin tools (Okta, Jamf, Google Workspace) that are learned on
    the job, so an entry-level fit scores lower than on an engineering posting.
    None = use min_ats.
    """
    out: list[tuple[int, float, float, Job]] = []
    for job, score in ranked:
        tier = role_tier(job.title) if prefer_engineering else 0
        if prefer_engineering and tier < 0:
            continue  # business / management / customer-facing -> out
        r = score_resume(master_resume, job.description)
        pscore = r["score"] or 0
        n_jd = len(requirements(job.description))   # requirement units ("a or b" counts once)
        floor = min_ats_it if (min_ats_it is not None and is_it_support_role(job.title)) else min_ats
        if n_jd >= min_jd_skills and pscore >= floor:
            out.append((tier, float(pscore), score, job))
    out.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [(job, score) for _tier, _p, score, job in out]
