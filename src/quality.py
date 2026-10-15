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

    @property
    def has_findings(self) -> bool:
        return bool(self.unsupported_numbers or self.unsupported_entities)

    @property
    def severity(self) -> str:
        if self.unsupported_numbers:
            return "high"      # invented metrics are the dangerous case
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


def _extract_entities(text: str) -> list[str]:
    """Capitalized tokens that look like product, company, or tech names."""
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9+#.]{2,}(?:\s[A-Z][a-zA-Z0-9+#.]{2,})?\b", text)
    return [c for c in candidates if c.split()[0] not in _SENTENCE_STARTERS]


def verify_grounding(
    generated_texts: list[str],
    corpus: str,
    job_id: str = "",
) -> GroundingReport:
    """
    generated_texts: every free-text field the model produced -- summary,
    resume bullets, project bullets, the cover letter, and the LinkedIn
    drafts. See ApplicationPackage.groundable_texts(), which the caller uses
    to assemble this list.
    """
    blob = " ".join(t for t in generated_texts if t)

    unsupported_numbers: list[str] = []
    for num in _extract_numbers(blob):
        bare = num.rstrip("%x").replace(",", "")
        if num.lower() in corpus or bare in corpus:
            continue
        if num not in unsupported_numbers:
            unsupported_numbers.append(num)

    unsupported_entities: list[str] = []
    for ent in _extract_entities(blob):
        if _normalize(ent) in corpus:
            continue
        if ent not in unsupported_entities:
            unsupported_entities.append(ent)

    return GroundingReport(
        job_id=job_id,
        unsupported_numbers=unsupported_numbers,
        unsupported_entities=unsupported_entities,
    )


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
        if r.unsupported_entities:
            lines.append(f"- Unverified names: {', '.join(r.unsupported_entities[:12])}")
        lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return len(flagged)
