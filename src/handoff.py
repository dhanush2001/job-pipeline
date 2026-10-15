"""
handoff.py -- deliver finished PDFs to convenient folders (Downloads/Desktop).

Shared by main.py (bulk run), tailor_one.py (single paste-in), and
tailor_worklist.py (fill-in file). Everything lands in a `job-applications/`
subfolder so it doesn't scatter loose files across Downloads/Desktop.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

HANDOFF_SUBDIR = "job-applications"

# The blank a user fills in on the needs_description worklist. Shared so the
# writer (main.py) and reader (tailor_worklist.py) agree on the exact string.
NEEDS_DESC_PLACEHOLDER = "<paste the job description here>"

# Paste-proof worklist format: the description is free text between two marker
# lines, so pasting a raw job posting (colons, bullets, blank lines, quotes)
# can't break parsing. render_worklist / parse_worklist live together here so
# the writer and reader stay in sync.
WORKLIST_PASTE_MARKER = "--- PASTE DESCRIPTION BELOW ---"
WORKLIST_END_MARKER = "--- END ---"


def render_worklist(jobs: list[dict]) -> str:
    """jobs: [{company, title, url}]. Returns the fill-in worklist text."""
    lines = [
        "# NEEDS-DESCRIPTION WORKLIST",
        "# Paste each job's description between the two marker lines, then run:",
        "#   python -m src.tailor_worklist <this file>",
        "# Leave a paste area unchanged to skip that job. Lines starting with # are ignored.",
        "",
    ]
    for i, j in enumerate(jobs, 1):
        lines += [
            f"### JOB {i}",
            f"Company: {j.get('company', '')}",
            f"Title: {j.get('title', '')}",
            f"URL: {j.get('url', '')}",
            WORKLIST_PASTE_MARKER,
            NEEDS_DESC_PLACEHOLDER,
            WORKLIST_END_MARKER,
            "",
        ]
    return "\n".join(lines) + "\n"


def parse_worklist(text: str) -> list[dict]:
    """
    Parse a filled-in worklist into [{company, title, url, description}].
    A `### JOB` line starts a block; Company/Title/URL are read from the header
    region, and everything between the PASTE and END markers is the (free-text)
    description -- no escaping or indentation required.
    """
    jobs: list[dict] = []
    cur: dict | None = None
    mode: str | None = None
    desc: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("### JOB"):
            if cur is not None:
                cur["description"] = "\n".join(desc).strip()
                jobs.append(cur)
            cur = {"company": "", "title": "", "url": "", "description": ""}
            desc, mode = [], "header"
            continue
        if cur is None:
            continue  # preamble / comments before the first job
        if s == WORKLIST_PASTE_MARKER:
            mode = "desc"
            continue
        if s == WORKLIST_END_MARKER:
            mode = None
            continue
        if mode == "header":
            if not s or s.startswith("#") or ":" not in s:
                continue
            key, _, val = s.partition(":")
            key = key.strip().lower()
            if key in ("company", "title", "url"):
                cur[key] = val.strip()
        elif mode == "desc":
            desc.append(raw)
    if cur is not None:
        cur["description"] = "\n".join(desc).strip()
        jobs.append(cur)
    return jobs


# Simple paste-in format: paste whole job postings into a plain file, one after
# another, separated by a line of 3+ equals signs. No rigid template, no need to
# type company/title -- those are pulled from the posting text later. An optional
# `Company:` / `Title:` / `URL:` line at the top of a block is honored if present.
PASTE_SEPARATOR = "==="
_LABEL_KEYS = {"company": "company", "title": "title", "role": "title", "url": "url"}


def _split_pasted_blocks(text: str) -> list[str]:
    blocks, cur = [], []
    for raw in text.splitlines():
        if re.fullmatch(r"\s*={3,}\s*", raw) or re.fullmatch(r"\s*-{3,}\s*", raw):
            blocks.append("\n".join(cur))
            cur = []
        else:
            cur.append(raw)
    blocks.append("\n".join(cur))
    return [b for b in (b.strip() for b in blocks) if b]


def parse_pasted_jobs(text: str) -> list[dict]:
    """
    Parse a plain file of pasted postings (separated by `===` lines) into
    [{company, title, url, description}]. company/title are filled from an
    optional labeled header at the top of each block; when absent they stay ""
    and the caller extracts them from the description. The whole block (minus
    any peeled labels) is the description, so pasting a raw posting just works.
    """
    jobs: list[dict] = []
    for block in _split_pasted_blocks(text):
        if block.lstrip().startswith("#"):  # allow a leading comment/instruction line
            lines = [ln for ln in block.splitlines() if not ln.lstrip().startswith("#")]
            block = "\n".join(lines).strip()
            if not block:
                continue
        job = {"company": "", "title": "", "url": "", "description": ""}
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            key, _, val = lines[i].partition(":")
            k = key.strip().lower()
            if _ and k in _LABEL_KEYS and val.strip():
                job[_LABEL_KEYS[k]] = val.strip()
                i += 1
            elif not lines[i].strip():  # skip blank lines between labels
                i += 1
            else:
                break
        job["description"] = "\n".join(lines[i:]).strip() or block
        jobs.append(job)
    return jobs


def looks_like_worklist(text: str) -> bool:
    """True if the file uses the structured `### JOB` template (vs. plain paste)."""
    return any(ln.strip().startswith("### JOB") for ln in text.splitlines())


def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("_")
    return s or "job"


def resolve_dirs(config: dict, override: str | None = None) -> list[Path]:
    """
    Existing target dirs, each with the job-applications/ subfolder appended.
    override is a comma-separated string; else config.output.handoff_dirs; else
    ~/Downloads + ~/Desktop. Non-existent parents are skipped silently.
    """
    if override:
        raw = [d.strip() for d in override.split(",") if d.strip()]
    else:
        # Missing key -> default; explicit empty list -> disabled (return none).
        configured = config.get("output", {}).get("handoff_dirs", None)
        raw = ["~/Downloads", "~/Desktop"] if configured is None else configured
    dirs: list[Path] = []
    for d in raw:
        parent = Path(d).expanduser()
        if parent.is_dir():
            dirs.append(parent / HANDOFF_SUBDIR)
    return dirs


def deliver(folder_name: str, resume_pdf, letter_pdf, dirs: list[Path],
            extra_texts: dict[str, str] | None = None) -> list[Path]:
    """
    Give each job its OWN folder: <dir>/<job-applications>/<folder_name>/ holding
    resume.pdf, cover_letter.pdf, cover_letter.txt, plus any extra_texts
    ({filename: text}, e.g. missing_skills.md). Returns the resume paths written.
    """
    resume_pdf = Path(resume_pdf)
    letter_pdf = Path(letter_pdf) if letter_pdf else None
    letter_txt = letter_pdf.with_suffix(".txt") if letter_pdf else None
    resumes: list[Path] = []
    for d in dirs:
        job_dir = d / folder_name
        job_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(resume_pdf, job_dir / "resume.pdf")
        resumes.append(job_dir / "resume.pdf")
        if letter_pdf and letter_pdf.exists():
            shutil.copy(letter_pdf, job_dir / "cover_letter.pdf")
        if letter_txt and letter_txt.exists():
            shutil.copy(letter_txt, job_dir / "cover_letter.txt")
        for fname, text in (extra_texts or {}).items():
            (job_dir / fname).write_text(text, encoding="utf-8")
    return resumes
