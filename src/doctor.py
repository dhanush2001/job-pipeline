"""
doctor.py -- preflight check. Run this FIRST, before anything else.

    python -m src.doctor

Reports what is configured, what is missing, and what the next run would
actually do. Makes no network calls, spends no API credits, sends nothing.

    python -m src.doctor --render   # also compile a test PDF from your data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, BAD = "  OK  ", " WARN ", " MISS "
_results: list[tuple[str, str, str]] = []


def check(status: str, label: str, detail: str = "") -> None:
    _results.append((status, label, detail))


def _placeholder(text: str) -> bool:
    markers = ("Jane", "Doe", "jane.doe@example.com", "555-555-5555",
               "examplecorp", "anotherexample", "Acme Corp", "janedoe")
    return any(m in text for m in markers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--render", action="store_true", help="also compile a test PDF")
    args = parser.parse_args()

    print("=" * 68)
    print("  JOB PIPELINE -- PREFLIGHT CHECK")
    print("=" * 68)

    # --- system tools ---
    for tool, why in [("latexmk", "compiles resumes"), ("pdflatex", "LaTeX engine")]:
        if shutil.which(tool):
            check(OK, f"{tool} installed", why)
        else:
            check(BAD, f"{tool} NOT installed",
                  "sudo apt install texlive-latex-base texlive-latex-extra latexmk")

    # --- python deps ---
    for mod, pkg in [("requests", "requests"), ("yaml", "PyYAML"), ("jinja2", "Jinja2")]:
        try:
            __import__(mod)
            check(OK, f"{pkg} installed")
        except ImportError:
            check(BAD, f"{pkg} MISSING", "pip install -r requirements.txt")

    # --- secrets ---
    if os.environ.get("DEEPSEEK_API_KEY"):
        check(OK, "DEEPSEEK_API_KEY set", "(not validated -- no call made)")
    else:
        check(BAD, "DEEPSEEK_API_KEY not set", "export DEEPSEEK_API_KEY=...  <- REQUIRED")
    for var, why in [("GITHUB_TOKEN", "raises GitHub API limit 60->5000/hr"),
                     ("ADZUNA_APP_ID", "general job boards"),
                     ("PIPELINE_CONTACT", "your email in the User-Agent")]:
        check(OK if os.environ.get(var) else WARN, f"{var} {'set' if os.environ.get(var) else 'not set'}", why)

    # --- config ---
    cfg_path = Path(args.config)
    config = None
    if not cfg_path.exists():
        check(BAD, f"{args.config} MISSING", "cp config.example.yaml config.yaml")
    else:
        import yaml
        config = yaml.safe_load(cfg_path.read_text())
        check(OK, f"{args.config} found")

        cand = config.get("candidate", {})
        if _placeholder(json.dumps(cand)):
            check(BAD, "candidate details are PLACEHOLDERS", "still says Jane Doe -- edit config.yaml")
        else:
            check(OK, "candidate details", f"{cand.get('first_name')} {cand.get('last_name')}")

        gh = config.get("github", {}).get("username", "")
        check(OK if gh else WARN,
              f"github.username = {gh or '(blank)'}",
              "GitHub project tailoring ON" if gh else "GitHub project tailoring is OFF while blank")

        srcs = config.get("sources", {})
        ats_platforms = ("greenhouse", "lever", "ashby", "smartrecruiters", "workable")
        per_ats = {name: (srcs.get(name) or []) for name in ats_platforms}
        slugs = [s for lst in per_ats.values() for s in lst]
        enabled = [f"{name} ({len(lst)})" for name, lst in per_ats.items() if lst]
        if not slugs:
            check(WARN, "no ATS companies listed")
        elif _placeholder(json.dumps(slugs)):
            check(BAD, "company slugs are PLACEHOLDERS", f"{slugs} do not exist -- use real ones")
        else:
            check(OK, f"{len(slugs)} company slugs across {len(enabled)} ATS", ", ".join(enabled))

        prof = config.get("quality", {}).get("profile", "balanced")
        gen = config.get("limits", {}).get("max_jobs_per_run", "?")
        check(OK, f"quality profile: {prof}")
        cache_on = config.get("deepseek", {}).get("cache", {}).get("enabled", False)
        check(OK, f"local response cache: {'ON' if cache_on else 'off'}",
              "re-runs reuse saved responses (no API cost)" if cache_on
              else "each run calls the API fresh")
        judge_on = config.get("judge", {}).get("enabled", False)
        check(OK, f"judge (recruiter/XYZ rewrite): {'ON' if judge_on else 'off'}")
        check(OK, f"generate up to {gen}/run")

    # --- resume data + template ---
    tpl = (config or {}).get("resume", {}).get("template", "")
    if tpl and Path(tpl).exists():
        check(OK, "resume template found", tpl)
    elif tpl:
        check(BAD, "resume template MISSING", tpl)

    bank = (config or {}).get("resume", {}).get("bullet_bank", "")
    if not bank:
        check(WARN, "bullet_bank not set", "optional -- DeepSeek tailors from master resume + repos only")
    elif Path(bank).exists():
        n = len(Path(bank).read_text().splitlines())
        check(OK, "bullet_bank found", f"{bank} ({n} lines) -- fed to tailoring + grounding")
    else:
        check(BAD, "bullet_bank MISSING", f"{bank} configured but not found")

    md = (config or {}).get("resume", {}).get("master_data", "data/master_resume.yaml")
    if Path(md).exists():
        import yaml
        data = yaml.safe_load(Path(md).read_text())
        if _placeholder(json.dumps(data)):
            check(BAD, "master resume is the EXAMPLE", "still Jane Doe -- replace with your content")
        else:
            n_exp = len(data.get("experience", []))
            n_proj = len(data.get("projects", []))
            check(OK, "master resume", f"{data.get('name')} -- {n_exp} roles, {n_proj} projects")
            missing_ids = [e for e in data.get("experience", []) if not e.get("id")]
            if missing_ids:
                check(BAD, "experience entries missing 'id'",
                      "tailored bullets cannot be matched back without stable ids")
            else:
                check(OK, "all experience entries have stable ids")
    else:
        check(BAD, f"{md} MISSING", "cp data/master_resume.example.yaml data/master_resume.yaml")

    # --- pipeline state ---
    for path, label in [("data/seen_jobs.json", "jobs already scanned"),
                        ("data/apply_queue.json", "tailored packages from last run")]:
        if Path(path).exists():
            blob = json.loads(Path(path).read_text() or "{}")
            n = len(blob.get("seen_ids") or blob.get("queue") or [])
            check(OK, f"{label}: {n}")
        else:
            check(WARN, f"{path} not created yet", "normal before the first run")

    # --- print ---
    print()
    for status, label, detail in _results:
        print(f"[{status}] {label}" + (f"\n         -> {detail}" if detail else ""))

    blockers = [r for r in _results if r[0] == BAD]
    print("\n" + "=" * 68)
    if blockers:
        print(f"  {len(blockers)} BLOCKER(S) -- fix these before running:")
        for _, label, detail in blockers:
            print(f"    - {label}" + (f"  ({detail})" if detail else ""))
    else:
        print("  No blockers. Next:")
        print("    python -m src.main --config config.yaml --limit 10   # generate tailored applications")
        print("    python -m src.tailor_one   # or paste a single job description")
    print("=" * 68)

    if args.render and config:
        print("\nRendering a test PDF from your master resume...")
        sys.path.insert(0, ".")
        import yaml

        from src.render_resume import render_resume
        data = yaml.safe_load(Path(md).read_text())
        out = "output/resumes/_doctor_test.pdf"
        if render_resume(data, tpl, out):
            print(f"  OK -> {out}   (open it and check it looks like your resume)")
        else:
            print("  FAILED -- see the LaTeX error above")

    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
