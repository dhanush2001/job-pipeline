"""
tailor_worklist.py -- tailor a resume for EVERY job in one file.

The simplest workflow: paste whole job postings into a plain text file, one
after another, separated by a line of `===`. You don't have to type the company
or title -- they're pulled from each posting automatically. Then run this and it
produces one tailored resume (+ cover letter + missing_skills.md) per job,
delivered to Downloads/Desktop.

    python -m src.tailor_worklist --new 5          # write a file with 5 paste slots
    # open it, paste a full posting into each slot (separate them with ===)
    python -m src.tailor_worklist jobs.txt         # tailor them all

It also reads the structured `### JOB` worklist that a bulk run writes to
output/needs_description.txt (company/title already known, you add descriptions)
-- the format is auto-detected, so the same command handles both.

    python -m src.tailor_worklist <file> --dry-run   # stub content, no API key
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import yaml

from src import handoff, usage
from src.ats_score import missing_skills_md
from src.tailor_one import _PROFILES, make_one, prepare
from src.tailor_resume import extract_company_title

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tailor_worklist")

PLACEHOLDER = handoff.NEEDS_DESC_PLACEHOLDER
_MIN_DESC = 40  # shorter than this = still a stub / not really filled in


def _needs_filling(description: str) -> bool:
    d = (description or "").strip()
    return (not d) or (PLACEHOLDER in d) or (len(d) < _MIN_DESC)


def _new_template(n: int) -> str:
    """A plain file with N paste slots for the simple 'paste whole postings' flow."""
    head = [
        "# JOBS TO TAILOR",
        "# Paste one full job posting per slot below. Separate postings with a line of ===.",
        "# Company and title are detected from each posting automatically -- you don't type them.",
        '# (Optional: add a "Company:", "Title:", or "URL:" line at the top of a slot to override.)',
        "# Add or remove slots freely; empty slots are skipped. Then run:",
        "#   python -m src.tailor_worklist <this file>",
        "",
    ]
    slots = [f"<paste job posting {i} here>" for i in range(1, n + 1)]
    return "\n".join(head) + ("\n\n" + handoff.PASTE_SEPARATOR + "\n\n").join(slots) + "\n"


def write_new_template(n: int, out_path: Path) -> int:
    if n < 1:
        logger.error("--new needs a positive count (got %d).", n)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        logger.error("%s already exists -- delete it or pass --out for a different path.", out_path)
        return 1
    out_path.write_text(_new_template(n), encoding="utf-8")
    print(f"Wrote a paste file with {n} slot(s) to {out_path}")
    print("Paste a full job posting into each slot (separate them with a === line), then run:")
    print(f"    python -m src.tailor_worklist {out_path}")
    return 0


def _heuristic_title(desc: str) -> str:
    for ln in desc.splitlines():
        if ln.strip():
            return ln.strip()[:80]
    return "Role"


def _heuristic_company(desc: str) -> str:
    m = re.search(r"\b(?:at|@|join)\s+([A-Z][A-Za-z0-9&.\- ]{2,40})", desc)
    return m.group(1).strip().rstrip(".") if m else "Company"


def _resolve_company_title(job: dict, config: dict, dry_run: bool) -> tuple[str, str]:
    """Use labeled company/title if present; otherwise pull them from the posting
    text (one cheap API call), falling back to a heuristic when that's not
    available (dry-run, no key, or the call fails)."""
    company = str(job.get("company", "")).strip()
    title = str(job.get("title", "")).strip()
    desc = job.get("description", "")
    if not (company and title) and not dry_run:
        ds = config.get("deepseek", {})
        prof = _PROFILES.get(config.get("quality", {}).get("profile", "balanced"), _PROFILES["balanced"])
        got = extract_company_title(desc, os.environ.get("DEEPSEEK_API_KEY", ""), model=ds.get("model") or prof[0])
        if got:
            company = company or got[0]
            title = title or got[1]
    return (company or _heuristic_company(desc), title or _heuristic_title(desc))


def run(args) -> int:
    if args.new is not None:
        return write_new_template(args.new, Path(args.out or "output/needs_description.txt"))
    if not args.worklist:
        logger.error("Give a file of jobs to tailor, or --new N to create a blank one.")
        return 2

    config = yaml.safe_load(Path(args.config).read_text())
    usage.reset()
    text = Path(args.worklist).read_text()
    if handoff.looks_like_worklist(text):
        jobs = handoff.parse_worklist(text)          # structured ### JOB template
    else:
        jobs = handoff.parse_pasted_jobs(text)       # plain 'paste postings, split by ===' file
    if not jobs:
        logger.error("No jobs found in %s.", args.worklist)
        return 2

    pending = [j for j in jobs if not _needs_filling(j.get("description", ""))]
    skipped = len(jobs) - len(pending)
    if skipped:
        logger.info("Skipping %d empty/unfilled slot(s).", skipped)
    if not pending:
        logger.error("Nothing to do -- paste job postings into %s first.", args.worklist)
        return 1

    ctx = prepare(config, args.dry_run)
    dirs = handoff.resolve_dirs(config, args.handoff)
    done = 0
    for j in pending:
        company, title = _resolve_company_title(j, config, args.dry_run)
        result = make_one(config, ctx, company, title, j["description"], args.dry_run)
        if result is None:
            continue
        resume_pdf, letter_pdf, unsupported, ats_result = result
        folder = f"{handoff.slug(company)}__{handoff.slug(title)}"
        missing_md = missing_skills_md(title, company, j.get("url", ""), ats_result, ctx["evidence_skills"])
        delivered = handoff.deliver(folder, resume_pdf, letter_pdf, dirs,
                                    extra_texts={"missing_skills.md": missing_md})
        done += 1
        ats = f"ATS {ats_result['score']}%  " if ats_result.get("score") is not None else ""
        flag = f"  !! unverified numbers: {', '.join(unsupported)}" if unsupported else ""
        loc = str(delivered[0]) if delivered else resume_pdf
        print(f"[{done}] {ats}{title} @ {company} -> {loc}{flag}")

    print(f"\nTailored {done}/{len(pending)} job(s).")
    if not args.dry_run:
        usage.finalize(config, logger, extra={"flow": "tailor_worklist", "jobs_tailored": done})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tailor a resume for every job in one file.")
    parser.add_argument("worklist", nargs="?",
                        help="File of pasted postings (or a filled ### JOB worklist); omit with --new")
    parser.add_argument("--new", type=int, metavar="N",
                        help="Write a file with N paste slots and exit")
    parser.add_argument("--out", help="Where --new writes the file (default output/needs_description.txt)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--handoff", help="Comma-separated dirs to copy PDFs into")
    parser.add_argument("--dry-run", action="store_true", help="Stub content, no API key needed")
    sys.exit(run(parser.parse_args()))
