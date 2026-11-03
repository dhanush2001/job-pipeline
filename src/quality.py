"""
quality.py
-----------
Checks generated resume content against its sources for fabrication.

Why this exists: the failure mode of LLM resume tailoring isn't clumsy
phrasing, it's invented facts. "Reduced p99 latency by 40%" reads beautifully
and is a serious problem if your master resume never mentioned 40%. At 100
resumes/day nobody reads every line, so the check has to be mechanical.

The prompt already forbids invention. This is the belt to that suspenders:
prompts are a request, this is a verification. It flags rather than blocks,
because legitimate rephrasing does introduce new words -- the point is to put
a short, specific list in front of you instead of a 100-PDF pile to eyeball.

Highest-signal check is numbers. A metric in the output that appears nowhere
in the source is the single most common and most damaging fabrication.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Numbers that are almost always safe: years, small counts, versions.
_BENIGN_NUMBERS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0"}

# Words that look like proper nouns at sentence start but aren't claims.
_SENTENCE_STARTERS = {
    "Built", "Designed", "Developed", "Implemented", "Led", "Created", "Wrote",
    "Managed", "Improved", "Reduced", "Increased", "Delivered", "Shipped",
    "Collaborated", "Architected", "Maintained", "Automated", "Optimized",
    "Migrated", "Deployed", "Tested", "Refactored", "Analyzed", "Researched",
    "The", "This", "My", "A", "An", "In", "As", "At", "For", "With", "During",
    "I", "We", "Their", "Our", "It", "These", "Those",
    # Cover-letter salutations / closings / sign-offs (prose, not fabrications):
    "Dear", "Hello", "Hi", "Sincerely", "Regards", "Best", "Warm", "Warmly",
    "Thank", "Thanks", "Thanking", "Cheers", "Respectfully", "Hiring", "Team",
    # Common prose/sentence openers that get capitalized mid-letter:
    "Would", "Could", "Should", "Will", "Can", "May", "Totally", "Either",
    "Neither", "Also", "More", "Most", "However", "Additionally", "Furthermore",
    "Moreover", "Having", "Being", "Experienced", "Passionate", "Excited",
    "Currently", "Recently", "After", "Before", "When", "While", "Through",
    "Throughout", "By", "From", "Since", "Although", "Though", "Because", "If",
    "Whether", "Given", "Using", "Working", "Please", "Beyond", "Both", "Here",
    # Seasons / time words that appear in intern-posting language:
    "Summer", "Winter", "Fall", "Spring", "Autumn",
}


@dataclasses.dataclass
class GroundingReport:
    job_id: str
    unsupported_numbers: list[str]
    unsupported_entities: list[str]
    # Number + unit exist in your sources, but the words around them don't match
    # the claim -- e.g. "cut latency by 20%" when the source's 20% is about cost.
    # Not provably wrong, but a human must check the claim means the same thing.
    mismatched_context: list[str] = dataclasses.field(default_factory=list)
    # Scope/impact words ("self-serve", "led", "reduced") the rewrite added to an
    # entry whose own sources never say them -- see verify_claims().
    unsupported_claims: list[str] = dataclasses.field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.unsupported_numbers or self.unsupported_entities or self.mismatched_context
                    or self.unsupported_claims)

    @property
    def severity(self) -> str:
        if self.unsupported_numbers:
            return "high"      # invented metrics are the dangerous case
        if self.mismatched_context or self.unsupported_claims:
            return "medium"    # real number on the wrong claim, or a claim the entry never made
        if self.unsupported_entities:
            return "low"
        return "clean"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def build_source_corpus(
    master_resume: dict[str, Any],
    github_repos: list[dict[str, Any]] | None = None,
    job_description: str = "",
    bullet_bank: str = "",
) -> str:
    """
    Everything the model was legitimately allowed to draw on. The job
    description is included because echoing a technology named in the posting
    ("experience with Kubernetes") is fair tailoring, not fabrication --
    excluding it would bury real findings under false positives. The bullet
    bank is included for the same reason: its numbers are the candidate's real
    work, so a metric copied from it must not be flagged.
    """
    parts: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
        elif isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, (int, float)):
            parts.append(str(obj))

    walk(master_resume)
    if github_repos:
        walk(github_repos)
    if job_description:
        parts.append(job_description)
    if bullet_bank:
        parts.append(bullet_bank)

    return _normalize(" ".join(parts))


def _is_year(token: str) -> bool:
    """A bare 4-digit calendar year (1900-2099). These come from the posting
    ('Summer 2027 intern') or dates, never from a fabricated metric, so they're
    pure noise in the grounding report -- don't flag them. A year carrying a
    %/x/comma suffix is NOT caught here and is still checked."""
    return bool(re.fullmatch(r"(?:19|20)\d{2}", token))


def _extract_numbers(text: str) -> list[str]:
    """Percentages, multipliers, and quantities -- the high-risk claims."""
    found = re.findall(r"\d+(?:\.\d+)?%|\d+(?:,\d{3})+|\d+(?:\.\d+)?x|\b\d+(?:\.\d+)?\b", text)
    return [n for n in found if n not in _BENIGN_NUMBERS and not _is_year(n)]


_ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9+#.\-]{2,}")
_PLAIN_WORD = re.compile(r"[A-Z][a-z]+")


def _extract_entities(text: str) -> list[str]:
    """Capitalized tokens that look like product, company, or tech names,
    checked one word at a time ("Python REST" -> Python, REST).

    A plain Capitalized word (no inner capitals, digits, or symbols) that opens
    a sentence or bullet is skipped: there it is almost always ordinary English
    ("Deploying", "Earlier", "Happy to..."), and flagging those buried the real
    signal. Names with shape -- PyTorch, FastAPI, PWA, Node.js, GPT-4 -- are
    always checked, wherever they appear."""
    out: list[str] = []
    for m in _ENTITY.finditer(text):
        tok = m.group(0).rstrip(".-")
        if len(tok) < 3 or tok in _SENTENCE_STARTERS:
            continue
        before = text[:m.start()].rstrip()
        starts_sentence = not before or before[-1] in ".!?:;\n\u2022-\u2013\u2014(\"'\u201c"
        if starts_sentence and _PLAIN_WORD.fullmatch(tok):
            continue
        out.append(tok)
    return out


def _number_grounded(bare: str, corpus: str) -> bool:
    """True only if `bare` (a number with %/x/commas already stripped) appears in
    the corpus as a standalone quantity -- NOT as a digit-substring of a longer
    number. So "20" is grounded by "earned 20%" but NOT by the year "2024" or by
    "120". Plain substring matching (the old behaviour) grounded both, letting an
    invented metric pass whenever its digits happened to appear inside an
    unrelated number.

    Thousands separators are first stripped FROM THE CORPUS (only between digits,
    so "2024, and" is untouched) so that a claim of "15,000" -- which arrives
    here as bare "15000" -- is still grounded by "15,000" in the source."""
    degrouped = re.sub(r"(?<=\d),(?=\d)", "", corpus)
    return re.search(rf"(?<![\d.]){re.escape(bare)}(?![\d.])", degrouped) is not None


# ---- what a number MEANS: its unit and the words around it ----------------------
# A bare digit match isn't evidence: "improved latency by 20%" is not supported
# by "served 20 users". So a claimed number must appear in the sources with a
# compatible UNIT ("%" with "%", "x" with "x", "users" with "users"), and should
# share at least one content word nearby -- otherwise it's flagged for review.
_STOP = {
    "the", "and", "for", "with", "from", "into", "over", "under", "than", "that", "this", "these",
    "those", "using", "used", "use", "via", "per", "across", "while", "within", "by", "of", "to",
    "in", "on", "at", "an", "a", "as", "is", "was", "were", "are", "be", "been", "its", "their",
    "our", "my", "your", "more", "most", "less", "up", "down", "out", "about", "all", "each",
    "every", "both", "also", "which", "who", "when", "after", "before", "during", "through",
    "time", "times", "total", "approximately", "around", "nearly", "over", "than",
}
_DEGROUP = re.compile(r"(?<=\d),(?=\d)")


def _stem(word: str) -> str:
    w = word.lower().strip(".-+#")
    if w.endswith("s") and len(w) > 3:
        w = w[:-1]
    return w[:5]


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9+#-]*", text.lower()) if len(w) >= 3 and w not in _STOP]


def _unit(text: str, token: str, end: int) -> str | None:
    """'%', 'x', a noun stem ('user'), or None when nothing meaningful follows."""
    if token.endswith("%"):
        return "%"
    if token.lower().endswith("x"):
        return "x"
    rest = text[end:end + 40].lower()
    if re.match(r"\s*(?:%|percent\b)", rest):
        return "%"
    if re.match(r"\s*(?:x\b|×|-fold\b)", rest):
        return "x"
    words = _content_words(rest)[:1]
    return _stem(words[0]) if words else None


def _context(text: str, start: int, end: int) -> set[str]:
    return {_stem(w) for w in _content_words(text[max(0, start - 70):start] + " " + text[end:end + 70])}


def _occurrences(text: str, bare: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in
            re.finditer(rf"(?<![\d.]){re.escape(bare)}(?![\d.])(?:%|x\b)?", text)]


def _check_number(token: str, claim: str, start: int, end: int, source: str) -> str:
    """'ok', 'unsupported', or 'context' (needs a human look).

    unsupported -- the number never appears in the sources with the same KIND of
                   quantity: a percentage backed only by a count ("20%" vs "20
                   users"), or a multiplier backed by a plain number.
    context     -- it appears with the same kind, but neither its noun nor any
                   nearby word matches the claim, so it may describe something else.
    """
    bare = token.rstrip("%xX").replace(",", "")
    unit = _unit(claim, token, end)
    ctx = _context(claim, start, end) - {unit}
    best = "unsupported"
    for s, e in _occurrences(source, bare):
        src_unit = _unit(source, source[s:e], e)
        noun_match = False
        if unit in ("%", "x") or src_unit in ("%", "x"):
            if unit != src_unit:
                continue                      # different kind of quantity entirely
        elif unit:
            # Noun units: allow a few words between ("20 internal users").
            noun_match = unit in {_stem(w) for w in _content_words(source[e:e + 60])[:5]}
        if noun_match or not ctx or ctx & _context(source, s, e):
            return "ok"
        best = "context"
    return best


def verify_grounding(
    generated_texts: list[str],
    corpus: str,
    job_id: str = "",
    number_corpus: str | None = None,
) -> GroundingReport:
    """
    generated_texts: every free-text field the model produced -- summary,
    resume bullets, project bullets, the cover letter, and the LinkedIn
    drafts. See ApplicationPackage.groundable_texts(), which the caller uses
    to assemble this list.

    number_corpus: the sources allowed to vouch for NUMERIC claims; defaults to
    `corpus`. Callers should pass a corpus WITHOUT the job description here,
    because a metric printed in a posting ("cut p99 latency by 40%") is the
    employer's target, not proof the candidate achieved it -- only the
    candidate's own sources (resume, repos, bullet bank) can ground a number.
    Entity/name checks still use the full `corpus`, since echoing a technology
    named in the posting is fair tailoring, not fabrication.
    """
    num_corpus = corpus if number_corpus is None else number_corpus
    blob = " ".join(t for t in generated_texts if t)
    source = _DEGROUP.sub("", num_corpus)

    # Each claimed number is checked where it occurs: present in the candidate's
    # sources, with a compatible unit, and ideally near a shared content word.
    unsupported_numbers: list[str] = []
    mismatched_context: list[str] = []
    for m in re.finditer(r"\d+(?:\.\d+)?%|\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?x\b|\b\d+(?:\.\d+)?\b", blob):
        num = m.group(0)
        if num in _BENIGN_NUMBERS or _is_year(num):
            continue
        if len(num) == 2 and m.start() > 0 and blob[m.start() - 1] in "'\u2019":
            continue  # abbreviated year: "Dec '26", "Fall \u201927"
        verdict = _check_number(num, blob, m.start(), m.end(), source)
        if verdict == "unsupported":
            if num not in unsupported_numbers:
                unsupported_numbers.append(num)
        elif verdict == "context":
            snippet = blob[max(0, m.start() - 40):m.end() + 30].strip()
            note = f"{num} (in: \"...{snippet}...\")"
            if not any(x.startswith(num + " ") for x in mismatched_context):
                mismatched_context.append(note)
    # A number unsupported in one place but fine in another is still unsupported.
    mismatched_context = [x for x in mismatched_context if x.split(" ", 1)[0] not in unsupported_numbers]

    unsupported_entities: list[str] = []
    # Per text, so each bullet's first word counts as a sentence start.
    for ent in (e for t in generated_texts if t for e in _extract_entities(t)):
        if _normalize(ent) in corpus:
            continue
        if ent not in unsupported_entities:
            unsupported_entities.append(ent)

    return GroundingReport(
        job_id=job_id,
        unsupported_numbers=unsupported_numbers,
        unsupported_entities=unsupported_entities,
        mismatched_context=mismatched_context,
    )


# ---- what a bullet CLAIMS: scope and impact words --------------------------------
# Numbers and names are checked above, but a rewrite can change what you did
# without either: "a dashboard staff used daily" becomes "a self-serve interface",
# "worked with" becomes "led". These words carry scope, ownership or impact, so
# each one in a bullet must also appear in THAT entry's own sources -- an
# "Improved" in your Matrix project doesn't vouch for an "improved" at Acme.
_CLAIMS: list[tuple[str, str]] = [
    ("self-serve", r"self[- ]?serv\w*"),
    ("led", r"led|leads?|leading|leadership"),
    ("spearheaded", r"spearhead\w*"),
    ("owned", r"own(?:ed|s|ing|ership)|owner"),
    ("architected", r"architect(?:ed|ing)?"),
    ("mentored", r"mentor\w*"),
    ("managed", r"manag(?:ed|es|ing)"),
    ("reduced", r"reduc\w*"),
    ("cut", r"cut(?:s|ting)?"),
    ("increased", r"increas\w*"),
    ("improved", r"improv\w*"),
    ("boosted", r"boost\w*"),
    ("accelerated", r"accelerat\w*"),
    ("saved", r"sav(?:ed|es|ing|ings)"),
    ("scaled", r"scal(?:ed|es|ing)|scalab\w*"),
    ("launched", r"launch\w*"),
    ("eliminated", r"eliminat\w*"),
    # "Investigated issues" -> "Resolved issues" claims an outcome nobody recorded.
    ("resolved", r"resolv(?:ed|es|ing)|fix(?:ed|es|ing)"),
    ("consolidated", r"consolidat\w*|centraliz\w*|unif(?:ied|ies|ying)"),
    ("dispatch", r"dispatch\w*"),
    ("doubled", r"doubl\w*|tripl\w*"),
    ("end-to-end", r"end[- ]to[- ]end"),
    ("independently", r"independent\w*|single[- ]handed\w*|solely|sole"),
    ("customers", r"customers?|clients?|revenue|enterprise\w*"),
    ("cross-functional", r"cross[- ]functional|stakeholders?|company[- ]wide|organization[- ]wide"),
    ("production-grade", r"production[- ]grade|high[- ]traffic|mission[- ]critical|large[- ]scale"),
    ("millions", r"millions?|thousands?|billions?"),
]
_CLAIM_RE = [(name, re.compile(rf"(?<![\w-])(?:{pat})(?![\w-])", re.I)) for name, pat in _CLAIMS]


def _plain_text(obj: Any) -> str:
    if isinstance(obj, dict):
        return " ".join(_plain_text(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(_plain_text(v) for v in obj)
    return str(obj) if isinstance(obj, (str, int, float)) else ""


def _key(name: str) -> str:
    """'tiny-serve: LLM Inference Server' -> 'tiny-serve'; matches bank headings and repos."""
    words = re.findall(r"[a-z0-9][a-z0-9+#.-]*", name.split(":")[0].lower())
    return words[0] if words else ""


def _bank_sections(bullet_bank: str) -> list[tuple[str, str]]:
    """(heading, body) for each '## ' section of the bullet bank."""
    parts = re.split(r"(?m)^## ", bullet_bank or "")
    return [(p.split("\n", 1)[0].lower(), p) for p in parts[1:]]


def verify_claims(package: Any, master_resume: dict[str, Any], bullet_bank: str = "",
                  github_repos: list[dict[str, Any]] | None = None) -> list[str]:
    """Scope/impact words each generated bullet adds that its own entry's sources
    never use. Sources per entry: the master entry, its bullet-bank section, and
    its GitHub repo. Summary and letters are checked against all of them, since
    they may draw on any entry. Flags for review; it can't know a new "improved"
    is false, only that nothing you wrote says it."""
    sections = _bank_sections(bullet_bank)
    repos = github_repos or []
    entries = {e.get("id"): e for e in master_resume.get("experience") or []}
    projects = master_resume.get("projects") or []

    def related(key: str) -> str:
        if not key:
            return ""
        text = [body for head, body in sections if key in head]
        text += [_plain_text(r) for r in repos if key in str(r.get("name", "")).lower()]
        return " ".join(text)

    checks: list[tuple[str, str, str]] = []      # (label, text, source)
    for eid, bullets in (getattr(package, "experience_bullets", {}) or {}).items():
        entry = entries.get(eid)
        if entry is None:
            continue
        key = _key(str(entry.get("company", "")))
        source = _plain_text(entry) + " " + related(key)
        checks += [(str(entry.get("company") or eid), b, source) for b in bullets]
    for proj in getattr(package, "projects", []) or []:
        name = str(proj.get("name", ""))
        key = _key(name)
        master = [p for p in projects if key and _key(str(p.get("name", ""))) == key]
        source = _plain_text(master) + " " + related(key)
        checks += [(name.split(":")[0], b, source) for b in proj.get("bullets", []) or []]
    everything = " ".join([_plain_text(master_resume), bullet_bank or "", _plain_text(repos)])
    if getattr(package, "summary", ""):
        checks.append(("summary", package.summary, everything))
    # A letter also describes the company ("the stack your team owns"); only its
    # sentences about you make claims.
    for sentence in re.split(r"(?<=[.!?])\s+", getattr(package, "cover_letter", "") or ""):
        if re.search(r"\b(?:I|my|me)\b", sentence):
            checks.append(("cover letter", sentence, everything))

    found: list[str] = []
    for label, text, source in checks:
        for name, rx in _CLAIM_RE:
            m = rx.search(text)
            if m and not rx.search(source):
                snippet = text[max(0, m.start() - 40):m.end() + 40].strip()
                note = f"{label}: \"{m.group(0)}\" (in: \"...{snippet}...\")"
                if note not in found:
                    found.append(note)
    return found


def write_grounding_report(reports: list[GroundingReport], path: str) -> int:
    """Write only the flagged ones. Returns the count needing review."""
    from pathlib import Path

    flagged = [r for r in reports if r.has_findings]
    if not flagged:
        return 0

    lines = [
        "# Grounding review",
        "",
        "Claims in generated resumes that could not be traced to your master",
        "resume, your GitHub repos, or the job posting. Rephrasing causes some",
        "false positives -- but check every number, since an invented metric is",
        "the fabrication that actually costs you an offer.",
        "",
    ]
    for r in sorted(flagged, key=lambda x: (x.severity != "high", x.job_id)):
        lines.append(f"## {r.job_id}  [{r.severity}]")
        if r.unsupported_numbers:
            lines.append(f"- **Unverified numbers:** {', '.join(r.unsupported_numbers)}")
        if r.mismatched_context:
            lines.append("- **Check the claim, not just the number** (it exists in your sources, "
                         "but next to different words):")
            lines += [f"  - {x}" for x in r.mismatched_context[:6]]
        if r.unsupported_claims:
            lines.append("- **New scope or impact claims** (the entry's own sources never say this):")
            lines += [f"  - {x}" for x in r.unsupported_claims[:8]]
        if r.unsupported_entities:
            lines.append(f"- Unverified names: {', '.join(r.unsupported_entities[:12])}")
        lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return len(flagged)
