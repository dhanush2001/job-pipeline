"""
tailor_one.py -- generate ONE tailored resume from a pasted job description.

For roles you find yourself (LinkedIn, Workday, a referral -- anywhere the
scanners don't reach): paste the description, and this produces a tailored
resume + cover letter PDF and copies them to your Downloads/Desktop
(job-applications/ subfolder) with clear names, ready to upload by hand.

Usage:
    python -m src.tailor_one                                  # interactive
    python -m src.tailor_one --company "Acme" --title "New Grad SWE" --desc-file jd.txt
    python -m src.tailor_one --company "Acme" --title "New Grad SWE" --dry-run   # no key

The reusable pieces (prepare / make_one) are also imported by tailor_worklist.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from src import handoff, usage
from src.ats_score import missing_skills_md, score_resume, skills_in
from src.github_projects import fetch_repos
from src.judge import judge_and_rewrite
from src.quality import build_source_corpus, verify_grounding
from src.render_resume import merge_tailored_content, render_cover_letter, render_resume
from src.tailor_resume import generate_application_package, stub_application_package

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tailor_one")

_PROFILES = {
    "fast": ("deepseek-v4-flash", "disabled", None),
    "balanced": ("deepseek-v4-flash", "low", 8000),
    "high": ("deepseek-v4-pro", "high", 16000),
}


def prepare(config: dict, dry_run: bool) -> dict:
    """Load everything shared across jobs: master resume, candidate, bullet
    bank, and (unless dry-run) GitHub repos. Done once, reused per job."""
    master_resume = yaml.safe_load(Path(config["resume"]["master_data"]).read_text())
    bank_path = config["resume"].get("bullet_bank", "")
    bullet_bank = ""
    if bank_path and Path(bank_path).exists():
        bullet_bank = Path(bank_path).read_text(encoding="utf-8")

    github_repos: list[dict] = []
    gh_cfg = config.get("github", {})
    if not dry_run and gh_cfg.get("username"):
        repos = fetch_repos(
            username=gh_cfg["username"],
            token=os.environ.get("GITHUB_TOKEN"),
            max_repos=gh_cfg.get("max_repos", 12),
            include_forks=gh_cfg.get("include_forks", False),
            include_readmes=gh_cfg.get("include_readmes", True),
        )
        github_repos = [r.to_prompt_dict() for r in repos]

    return {
        "master_resume": master_resume,
        "candidate": config["candidate"],
        "bullet_bank": bullet_bank,
        "github_repos": github_repos,
        "evidence_skills": skills_in(json.dumps(master_resume), bullet_bank, json.dumps(github_repos)),
    }


def make_one(config, ctx, company, title, description, dry_run):
    """
    Generate + render one job to the repo output/ dir. Returns
    (resume_pdf, letter_pdf, unsupported_numbers, ats_score) or None on failure.
    Does NOT copy to handoff dirs -- the caller does that so it can batch.
    """
    master_resume = ctx["master_resume"]
    if dry_run:
        pkg = stub_application_package(master_resume, title, company)
    else:
        ds = config.get("deepseek", {})
        prof = _PROFILES.get(config.get("quality", {}).get("profile", "balanced"), _PROFILES["balanced"])
        cache = ds.get("cache", {})
        pkg = generate_application_package(
            master_resume=master_resume, job_title=title, company=company,
            job_description=description, api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            model=ds.get("model") or prof[0], github_repos=ctx["github_repos"],
            bullet_bank=ctx["bullet_bank"], reasoning=ds.get("reasoning") or prof[1],
            temperature=ds.get("temperature", 0.4), max_tokens=ds.get("max_tokens") or prof[2],
            cache_enabled=cache.get("enabled", False), cache_dir=cache.get("dir", ".cache/deepseek"),
        )
    if pkg is None:
        logger.error("Generation failed for %s @ %s (is DEEPSEEK_API_KEY set? try --dry-run).", title, company)
        return None

    # Judge pass -- the same recruiter/hiring-manager rewrite the scan flow runs:
    # rewrite experience bullets + summary in XYZ form, weaving in ONLY the
    # evidence-backed missing keywords. Runs before grounding so the rewrites are
    # re-verified. Skipped on dry-run (stub content) and when judge is disabled.
    if not dry_run and config.get("judge", {}).get("enabled", False) and pkg.experience_bullets:
        ds = config.get("deepseek", {})
        prof = _PROFILES.get(config.get("quality", {}).get("profile", "balanced"), _PROFILES["balanced"])
        merged_pre = merge_tailored_content(
            master_resume, pkg, ctx["github_repos"],
            job_text=f"{title} {description}",
        )
        missing = [k for k in score_resume(merged_pre, description)["missing"]
                   if k in ctx["evidence_skills"]]
        review = judge_and_rewrite(
            master_resume=master_resume, github_repos=ctx["github_repos"],
            bullet_bank=ctx["bullet_bank"], job_title=title, company=company,
            job_description=description, experience_bullets=pkg.experience_bullets,
            missing_keywords=missing, api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            model=ds.get("model") or prof[0], current_summary=pkg.summary,
            reasoning=ds.get("reasoning") or prof[1],
            temperature=ds.get("temperature", 0.4), max_tokens=ds.get("max_tokens") or prof[2],
        )
        if review and review.get("experience"):
            pkg.experience_bullets.update(review["experience"])
        if review and review.get("summary"):
            pkg.summary = review["summary"]

    corpus = build_source_corpus(master_resume, ctx["github_repos"], description, ctx["bullet_bank"])
    report = verify_grounding(pkg.groundable_texts(), corpus, job_id=f"{company} — {title}")

    base = f"{handoff.slug(company)}/{handoff.slug(title)}"
    resume_pdf = Path(config["resume"]["output_dir"]) / f"{base}.pdf"
    include_summary = config.get("resume", {}).get("include_summary", False)
    merged = merge_tailored_content(
        master_resume, pkg, ctx["github_repos"], include_summary,
        job_text=f"{title} {description}",
    )
    force_one_page = config.get("resume", {}).get("force_one_page", True)
    if not render_resume(merged, config["resume"]["template"], str(resume_pdf), fit_one_page=force_one_page):
        logger.error("Resume render failed for %s @ %s (is latexmk installed?).", title, company)
        return None
    ats = score_resume(merged, description)

    letter_pdf = None
    cl_cfg = config.get("cover_letter", {})
    if cl_cfg.get("enabled", True) and pkg.cover_letter:
        candidate_letter = Path(cl_cfg.get("output_dir", "output/cover_letters")) / f"{base}.pdf"
        if render_cover_letter(
            pkg, ctx["candidate"], company,
            cl_cfg.get("template", "templates/cover_letter.tex.j2"), str(candidate_letter),
        ):
            letter_pdf = candidate_letter

    return str(resume_pdf), (str(letter_pdf) if letter_pdf else None), report.unsupported_numbers, ats


def _read_description(args) -> str:
    if args.desc:
        return args.desc
    if args.desc_file:
        return Path(args.desc_file).read_text(encoding="utf-8")
    print("Paste the job description, then press Ctrl-D (Ctrl-Z + Enter on Windows):\n", file=sys.stderr)
    return sys.stdin.read()


def run(args) -> int:
    config = yaml.safe_load(Path(args.config).read_text())
    usage.reset()
    ctx = prepare(config, args.dry_run)

    company = args.company or input("Company: ").strip()
    title = args.title or input("Role / title: ").strip()
    if not company or not title:
        logger.error("Company and title are both required.")
        return 2
    description = _read_description(args).strip()
    if not description and not args.dry_run:
        logger.error("No job description provided. Paste one or use --desc-file.")
        return 2

    if args.dry_run:
        logger.info("DRY RUN: stub content, no API call, no key needed.")
    result = make_one(config, ctx, company, title, description, args.dry_run)
    if result is None:
        return 1
    resume_pdf, letter_pdf, unsupported, ats = result
    ats_score = ats.get("score")

    dirs = handoff.resolve_dirs(config, args.handoff)
    folder = f"{handoff.slug(company)}__{handoff.slug(title)}"
    missing_md = missing_skills_md(title, company, args.url or "", ats, ctx["evidence_skills"])
    delivered = handoff.deliver(folder, resume_pdf, letter_pdf, dirs, extra_texts={"missing_skills.md": missing_md})

    print("\n" + "=" * 68)
    print(f"  {title} @ {company}")
    if args.url:
        print(f"  Apply here:  {args.url}")
    print(f"  Resume:      {resume_pdf}")
    if ats_score is not None:
        print(f"  ATS match:   {ats_score}%")
    if letter_pdf:
        print(f"  Cover (txt): {Path(letter_pdf).with_suffix('.txt')}  <- paste into the form")
    if delivered:
        print("  Copied to:")
        for p in delivered:
            print(f"    {p}")
    else:
        print("  (No Downloads/Desktop found -- files are in output/ above.)")
    if unsupported:
        print(f"  !! CHECK unverified numbers before sending: {', '.join(unsupported)}")
    print("=" * 68)
    if not args.dry_run:
        usage.finalize(config, logger, extra={"flow": "tailor_one", "jobs_tailored": 1})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one tailored resume from a pasted job description.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--company")
    parser.add_argument("--title")
    parser.add_argument("--desc", help="Job description inline")
    parser.add_argument("--desc-file", help="Path to a file containing the job description")
    parser.add_argument("--url", help="Apply URL (printed for reference)")
    parser.add_argument("--handoff", help="Comma-separated dirs to copy PDFs into (default: ~/Downloads,~/Desktop)")
    parser.add_argument("--dry-run", action="store_true", help="Stub content, no API key needed")
    sys.exit(run(parser.parse_args()))
