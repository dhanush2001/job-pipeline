"""
match.py
--------
Scores and filters scanned jobs against the target role config before any
LLM tokens get spent tailoring them. Deliberately simple (keyword
overlap, not embeddings) so it's transparent and free to run on every job —
tune the keyword lists in config.yaml rather than the code.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from src.ats_score import score_resume
from src.scan_jobs import Job


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


def filter_and_rank(jobs: list[Job], matching_cfg: dict) -> list[tuple[Job, float]]:
    must_have = matching_cfg.get("must_have_keywords", [])
    nice_to_have = matching_cfg.get("nice_to_have_keywords", [])
    exclude = matching_cfg.get("exclude_keywords", [])
    exclude_description = matching_cfg.get("exclude_description_keywords", [])
    min_score = matching_cfg.get("min_score", 0.0)
    drop_phd_only = matching_cfg.get("exclude_phd_only_titles", False)

    scored: list[tuple[Job, float]] = []
    for job in jobs:
        if drop_phd_only and is_phd_only_title(job.title):
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
)


def role_tier(title: str) -> int:
    """2 = core SWE/ML/data IC, 1 = generic engineer/dev/scientist, 0 = neutral,
    -1 = business/management/customer-facing (checked first so 'Solutions
    Engineer' demotes despite the word 'engineer')."""
    t = (title or "").lower()
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
    """
    out: list[tuple[int, float, float, Job]] = []
    for job, score in ranked:
        tier = role_tier(job.title) if prefer_engineering else 0
        if prefer_engineering and tier < 0:
            continue  # business / management / customer-facing -> out
        r = score_resume(master_resume, job.description)
        pscore = r["score"] or 0
        n_jd = len(r["matched"]) + len(r["missing"])
        if n_jd >= min_jd_skills and pscore >= min_ats:
            out.append((tier, float(pscore), score, job))
    out.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [(job, score) for _tier, _p, score, job in out]
