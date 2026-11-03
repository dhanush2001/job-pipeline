"""
evaluate.py
-----------
Compare each tailored resume against your ORIGINAL master resume, so you can
judge the AI's edits with evidence instead of trusting the prompt.

Every run saves a record next to each PDF (output/resumes/<company>/<job>.json):
the posting, the content that actually rendered, its ATS result, the grounding
report, and the judge's notes. This script reads those records and writes
output/evaluation.md with, per job:

  * eligibility re-check   -- graduation window and required years vs. your profile
  * ATS: master vs tailored -- did tailoring actually close keyword gaps?
  * claims                  -- unverified numbers and "right number, wrong claim" cases
  * what changed            -- each bullet labelled kept / rewritten / new, with the
                               original beside the rewrite so you can read both
  * style checks            -- over-long bullets, bullets with no measurable result,
                               bold overuse, summary length, cover-letter basics

What it can't do: decide whether a rewrite is GOOD. The side-by-side is there so
a human can make that call in a few minutes per job. Run:

    python -m src.evaluate                 # every saved record
    python -m src.evaluate --limit 5       # the 5 most recent
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from src.ats_score import score_resume, skills_in
from src.guards import entry_skills, overclaimed_in_progress
from src.match import graduation_mismatch, required_experience_years

logger = logging.getLogger(__name__)

RECORD_VERSION = 1
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_HAS_NUMBER = re.compile(r"\d")
_GENERIC_OPENERS = ("i am writing to express", "i am excited to apply", "to whom it may concern",
                    "i would like to apply")


# ---- saving (called by main.py / tailor_one.py after each render) ------------------
def save_record(pdf_path: str, job: dict[str, Any], tailored: dict[str, Any] | None, ats: dict,
                grounding: Any, package: Any = None, review: dict | None = None,
                guard_notes: list[str] | None = None) -> str:
    """Write the evaluation record beside the PDF. Never raises -- an evaluation
    sidecar must not break a run."""
    path = str(Path(pdf_path).with_suffix(".json"))
    try:
        record = {
            "version": RECORD_VERSION,
            "job": job,
            "tailored": tailored or {},
            "ats": ats,
            "grounding": {
                "severity": getattr(grounding, "severity", ""),
                "unsupported_numbers": getattr(grounding, "unsupported_numbers", []),
                "mismatched_context": getattr(grounding, "mismatched_context", []),
                "unsupported_entities": getattr(grounding, "unsupported_entities", []),
                "unsupported_claims": getattr(grounding, "unsupported_claims", []),
            },
            "cover_letter": getattr(package, "cover_letter", "") if package else "",
            "referral_note": getattr(package, "referral_note", "") if package else "",
            "referral_dm": getattr(package, "referral_dm", "") if package else "",
            "judge": review or {},
            "guard": list(guard_notes or []),
            "resume_pdf": pdf_path,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save evaluation record %s: %s", path, exc)
    return path


# ---- comparison ----------------------------------------------------------------
def _plain(text: str) -> str:
    return _BOLD.sub(r"\1", text or "").strip()


def _label(bullet: str, originals: list[str]) -> tuple[str, str, float]:
    """kept / rewritten / new, plus the closest original and similarity."""
    if not originals:
        return "new", "", 0.0
    # Compare WORD sequences: character-level similarity rates any two English
    # sentences ~0.4, which would label an invented bullet "rewritten".
    words = re.findall(r"[a-z0-9+#.%]+", _plain(bullet).lower())
    best, best_ratio = "", 0.0
    for o in originals:
        r = difflib.SequenceMatcher(None, words, re.findall(r"[a-z0-9+#.%]+", _plain(o).lower())).ratio()
        if r > best_ratio:
            best, best_ratio = o, r
    kind = "kept" if best_ratio >= 0.9 else "rewritten" if best_ratio >= 0.4 else "new"
    return kind, best, best_ratio


def _sections(resume: dict) -> list[tuple[str, str, list[str]]]:
    out = []
    for e in resume.get("experience", []) or []:
        out.append(("experience", e.get("id") or e.get("company", "?"), list(e.get("bullets") or [])))
    for p in resume.get("projects", []) or []:
        out.append(("project", p.get("name", "?"), list(p.get("bullets") or [])))
    return out


def evaluate_record(record: dict, master: dict, matching_cfg: dict) -> dict:
    job, tailored = record.get("job", {}), record.get("tailored", {})
    desc = job.get("description", "")
    flags: list[str] = []

    # Eligibility -- re-checked so a screen that was off (or a pasted job) still shows up.
    grad_year = matching_cfg.get("graduation_year")
    window = graduation_mismatch(desc, grad_year)
    if window:
        flags.append(f"INELIGIBLE? posting's graduation window {window} excludes {grad_year}")
    req = required_experience_years(desc)
    limit = matching_cfg.get("max_required_experience_years")
    if req is not None and limit is not None and req > limit:
        flags.append(f"INELIGIBLE? requires {req}+ years of experience (your limit {limit})")

    # ATS: did tailoring help?
    before = score_resume(master, desc)
    after = record.get("ats") or score_resume(tailored, desc)
    closed = sorted(set(before.get("missing", [])) - set(after.get("missing", [])))

    # Claims.
    g = record.get("grounding", {})
    if g.get("unsupported_numbers"):
        flags.append(f"UNVERIFIED NUMBERS: {', '.join(g['unsupported_numbers'])}")
    if g.get("mismatched_context"):
        flags.append(f"{len(g['mismatched_context'])} number(s) may describe a different claim")
    if g.get("unsupported_claims"):
        flags.append(f"NEW CLAIMS: {len(g['unsupported_claims'])} scope/impact word(s) the entry's sources "
                     f"never use -- " + "; ".join(g["unsupported_claims"][:3]))

    # Skills placed under a job that doesn't support them (should be 0 now that
    # the guard runs; this re-check is what proves it).
    entries = {e.get("id"): e for e in master.get("experience", []) or []}
    misplaced = []
    for e in tailored.get("experience", []) or []:
        src = entries.get(e.get("id"))
        if src is None:
            continue
        allowed = entry_skills(src)
        for b in e.get("bullets") or []:
            extra = sorted(skills_in(_plain(b)) - allowed)
            if extra:
                misplaced.append(f"{e.get('id')}: {', '.join(extra)} in \"{_plain(b)[:80]}\"")
    if misplaced:
        flags.append(f"{len(misplaced)} bullet(s) credit a job with skills it doesn't support")
    overclaims = overclaimed_in_progress(tailored.get("summary", ""), master)
    if overclaims:
        flags.append("in-progress project described as finished")

    # What changed, bullet by bullet.
    master_by_key = {(k, n): b for k, n, b in _sections(master)}
    changes = []
    counts = {"kept": 0, "rewritten": 0, "new": 0}
    style: list[str] = []
    for kind, name, bullets in _sections(tailored):
        originals = master_by_key.get((kind, name), [])
        if kind == "project" and not originals:
            # Projects can come from GitHub repos -- not a fabrication by itself.
            changes.append({"section": f"{kind}: {name}", "note": "not in master (from GitHub or renamed)"})
        for b in bullets:
            label, orig, ratio = _label(b, originals)
            counts[label] += 1
            changes.append({"section": f"{kind}: {name}", "label": label, "tailored": b,
                            "original": orig, "similarity": round(ratio, 2)})
            plain = _plain(b)
            if len(plain) > 230:
                style.append(f"long bullet ({len(plain)} chars): {plain[:70]}...")
            if not _HAS_NUMBER.search(plain):
                style.append(f"no measurable result: {plain[:80]}")
            if len(_BOLD.findall(b)) > 4:
                style.append(f"{len(_BOLD.findall(b))} bold spans (max 4): {plain[:60]}...")
    dropped = sorted(n for (k, n), _ in master_by_key.items()
                     if k == "project" and n not in {nm for kk, nm, _ in _sections(tailored) if kk == "project"})

    summary = _plain(tailored.get("summary", ""))
    if summary and len(re.findall(r"[.!?](?:\s|$)", summary)) > 2:
        style.append("summary longer than 2 sentences")
    letter = (record.get("cover_letter") or "").strip()
    if letter:
        low = letter.lower()
        if any(low.startswith(g_) or g_ in low[:120] for g_ in _GENERIC_OPENERS):
            style.append("cover letter opens with a generic line")
        company = (job.get("company") or "").lower()
        if company and company not in low:
            style.append("cover letter never names the company")
        if len(letter.split()) > 400:
            style.append(f"cover letter is {len(letter.split())} words (aim for under ~350)")

    verdict = "REVIEW" if flags else ("CHECK STYLE" if style else "OK")
    return {"job": job, "verdict": verdict, "flags": flags, "style": style,
            "ats_before": before.get("score"), "ats_after": after.get("score"),
            "closed": closed, "still_missing": after.get("missing", []),
            "counts": counts, "changes": changes, "dropped_projects": dropped,
            "judge": record.get("judge") or {}, "pdf": record.get("resume_pdf", ""),
            "misplaced": misplaced, "overclaims": overclaims, "guard": record.get("guard") or []}


# ---- report --------------------------------------------------------------------------
def render_report(results: list[dict]) -> str:
    lines = ["# Tailored resume evaluation", "",
             "Objective checks of each tailored resume against your master resume. These "
             "catch ineligible postings, unverifiable or mismatched numbers, and edits that "
             "didn't help -- they do NOT judge writing quality. Read each side-by-side "
             "before you send.", "",
             "| verdict | company | role | ATS before -> after | kept / rewritten / new | flags |",
             "|---|---|---|---|---|---|"]
    for r in results:
        c = r["counts"]
        lines.append(f"| {r['verdict']} | {r['job'].get('company', '')} | {r['job'].get('title', '')} | "
                     f"{r['ats_before']}% -> {r['ats_after']}% | {c['kept']} / {c['rewritten']} / {c['new']} | "
                     f"{len(r['flags'])} |")
    lines.append("")
    for r in results:
        j = r["job"]
        lines += [f"## {j.get('company', '')} -- {j.get('title', '')}  [{r['verdict']}]", ""]
        if j.get("url"):
            lines.append(f"Posting: {j['url']}  ")
        lines.append(f"PDF: {r['pdf']}")
        lines.append("")
        for f in r["flags"]:
            lines.append(f"- **{f}**")
        lines.append(f"- ATS {r['ats_before']}% (master) -> {r['ats_after']}% (tailored)"
                     + (f"; closed: {', '.join(r['closed'])}" if r["closed"] else ""))
        if r["still_missing"]:
            lines.append(f"- Still missing: {', '.join(r['still_missing'])}")
        if r["dropped_projects"]:
            lines.append(f"- Projects left out for this role: {', '.join(r['dropped_projects'])}")
        if r["judge"].get("red_flags"):
            lines.append(f"- Judge flagged: {'; '.join(r['judge']['red_flags'][:4])}")
        for m in r.get("misplaced", []):
            lines.append(f"- misplaced skill: {m}")
        for o in r.get("overclaims", []):
            lines.append(f"- overclaim: {o}")
        for gn in r.get("guard", []):
            lines.append(f"- guard fixed: {gn}")
        for s in r["style"][:8]:
            lines.append(f"- style: {s}")
        lines += ["", "### What changed", ""]
        for ch in r["changes"]:
            if "label" not in ch:
                lines.append(f"- _{ch['section']}: {ch['note']}_")
                continue
            if ch["label"] == "kept":
                continue
            lines.append(f"- **{ch['label']}** ({ch['section']}, similarity {ch['similarity']})")
            if ch["original"]:
                lines.append(f"  - before: {_plain(ch['original'])}")
            lines.append(f"  - after:  {_plain(ch['tailored'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def load_records(directory: str, limit: int | None = None) -> list[dict]:
    files = sorted(Path(directory).rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(rec, dict) and rec.get("version") == RECORD_VERSION:
            out.append(rec)
        if limit and len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare tailored resumes against your master resume.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dir", help="where the .json records are (default: resume.output_dir)")
    ap.add_argument("--out", default="output/evaluation.md")
    ap.add_argument("--limit", type=int, help="only the N most recent records")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = yaml.safe_load(Path(args.config).read_text())
    master = yaml.safe_load(Path(config["resume"]["master_data"]).read_text())
    master.pop("private", None)  # personal notes: never sent to a model or printed
    matching = dict(config.get("matching", {}))
    if not matching.get("graduation_year"):
        from src.main import _expected_grad_year
        matching["graduation_year"] = _expected_grad_year(master)

    directory = args.dir or config["resume"].get("output_dir", "output/resumes")
    records = load_records(directory, args.limit)
    if not records:
        print(f"No evaluation records in {directory}. Run the pipeline first -- each tailored "
              "resume now saves a .json beside its PDF.", file=sys.stderr)
        return 1
    results = [evaluate_record(r, master, matching) for r in records]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_report(results), encoding="utf-8")
    review = sum(r["verdict"] == "REVIEW" for r in results)
    print(f"Evaluated {len(results)} resume(s): {review} need review -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
