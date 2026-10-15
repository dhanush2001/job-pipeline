"""
main.py  --  scan, match, generate, render
-------------------------------------------
Only touches public JSON APIs (Greenhouse, Lever, Ashby, SmartRecruiters,
Workable, Adzuna, GitHub) and the DeepSeek API, then compiles PDFs. No browser,
no ATS form -- everything it produces is for you to review and upload by hand.

At 100 jobs/run the DeepSeek calls dominate wall-clock time, so they run
through a thread pool. LaTeX compiles stay on the main thread: latexmk is
already CPU-bound and parallel temp-dir compiles gain little.

Outputs per job: tailored resume PDF, cover letter PDF + .txt, and LinkedIn
referral drafts written per company into output/referrals/ for manual sending.

Usage:
    python -m src.main --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from src import handoff
from src.ats_score import missing_skills_md, score_resume, skills_in
from src.github_projects import fetch_repos
from src import usage
from src.judge import judge_and_rewrite
from src.locations import country_of, is_targeted
from src.match import filter_and_rank, rerank_by_ats
from src.render_resume import merge_tailored_content, render_cover_letter, render_resume
from src.quality import build_source_corpus, verify_grounding, write_grounding_report
from src.scan_jobs import scan_all
from src.tailor_resume import generate_application_package, stub_application_package

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

# Below this many chars, a job description is too thin to tailor against, so the
# job is routed to the fill-in worklist instead of a (poor, ungrounded) tailoring.
MIN_DESC_CHARS = 120


def load_seen(path: str) -> set[str]:
    if not Path(path).exists():
        return set()
    # Tolerate an empty or half-written file (e.g. an interrupted save): treat it
    # as "nothing seen yet" rather than crashing the whole run. The run then
    # rebuilds the log from this run's jobs.
    text = Path(path).read_text().strip()
    if not text:
        return set()
    try:
        return set(json.loads(text).get("seen_ids", []))
    except json.JSONDecodeError:
        logger.warning("Seen-log %s is corrupt -- ignoring it this run.", path)
        return set()


def save_seen(path: str, seen_ids: set[str]) -> None:
    # Atomic write: a full temp file then rename, so an interrupt mid-save can't
    # leave an empty/half-written seen-log (which used to crash the next run).
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"seen_ids": sorted(seen_ids)}, indent=2))
    tmp.replace(p)


def _slug(text: str) -> str:
    """Filesystem-safe token: keep alnum . _ -, collapse everything else to '_'."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("_")
    return slug or "unknown"


def run(config_path: str, dry_run: bool = False, limit: int | None = None) -> None:
    config = yaml.safe_load(Path(config_path).read_text())

    if dry_run:
        logger.info(
            "DRY RUN: no DeepSeek calls, no API key needed. Using stub content "
            "from your master resume to exercise scan -> render -> outputs."
        )

    usage.reset()  # fresh token accounting for this run

    seen_log_path = config.get("seen_log_path", "data/seen_jobs.json")
    already_seen = load_seen(seen_log_path)

    master_resume = yaml.safe_load(Path(config["resume"]["master_data"]).read_text())

    # Optional bullet bank: a curated menu of pre-written project bullets the
    # model may draw from and rephrase. Loaded once (same for every job) and
    # fed both to the tailoring prompt and the grounding corpus. It supplements
    # -- does not replace -- the live GitHub fetch below, which still grounds
    # claims and surfaces new repos.
    bank_path = config["resume"].get("bullet_bank", "")
    bullet_bank = ""
    if bank_path and Path(bank_path).exists():
        bullet_bank = Path(bank_path).read_text(encoding="utf-8")
        logger.info("Loaded bullet bank (%d chars) from %s", len(bullet_bank), bank_path)
    elif bank_path:
        logger.warning("bullet_bank %s configured but not found -- skipping", bank_path)

    candidate = config["candidate"]
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    workers = config.get("deepseek", {}).get("concurrency", 6)
    ds_cfg = config.get("deepseek", {})

    # Quality profiles bundle the settings that move together. Explicit
    # model/reasoning keys in config still win over the profile.
    PROFILES = {
        "fast":     {"model": "deepseek-v4-flash", "reasoning": "disabled", "max_tokens": None},
        "balanced": {"model": "deepseek-v4-flash", "reasoning": "low",      "max_tokens": 8000},
        "high":     {"model": "deepseek-v4-pro",   "reasoning": "high",     "max_tokens": 16000},
    }
    profile_name = config.get("quality", {}).get("profile", "balanced")
    profile = PROFILES.get(profile_name, PROFILES["balanced"])
    if profile_name not in PROFILES:
        logger.warning("Unknown quality profile %r -- using balanced", profile_name)

    model = ds_cfg.get("model") or profile["model"]
    reasoning = ds_cfg.get("reasoning") or profile["reasoning"]
    max_tokens = ds_cfg.get("max_tokens") or profile["max_tokens"]
    logger.info("Quality profile '%s': model=%s reasoning=%s", profile_name, model, reasoning)
    temperature = ds_cfg.get("temperature", 0.4)
    cache_cfg = ds_cfg.get("cache", {})
    cache_enabled = cache_cfg.get("enabled", False)
    cache_dir = cache_cfg.get("dir", ".cache/deepseek")
    if cache_enabled:
        logger.info("Local response cache ON -> %s (re-runs reuse saved responses, no API cost)", cache_dir)
    if reasoning not in ("disabled", "off", "none"):
        logger.info(
            "Thinking mode '%s' is ON -- reasoning tokens bill at the output rate "
            "and temperature is ignored by the API.", reasoning,
        )

    # Fetch GitHub repos ONCE per run -- same for every posting, and each
    # fetch spends GitHub API rate-limit budget.
    gh_cfg = config.get("github", {})
    github_repos = []
    if gh_cfg.get("username"):
        repos = fetch_repos(
            username=gh_cfg["username"],
            token=os.environ.get("GITHUB_TOKEN"),
            max_repos=gh_cfg.get("max_repos", 12),
            include_forks=gh_cfg.get("include_forks", False),
            include_readmes=gh_cfg.get("include_readmes", True),
        )
        github_repos = [r.to_prompt_dict() for r in repos]

    jobs = scan_all(config)

    # Resolve each job's country (Adzuna jobs already carry it; company-board
    # jobs get it inferred from their location text), then keep only the
    # countries we're targeting. This drops foreign roles BEFORE any LLM call.
    matching_cfg = config.get("matching", {})
    targets = matching_cfg.get("target_countries", [])
    for j in jobs:
        if not j.country:
            j.country = country_of(j.location)
    if targets:
        before = len(jobs)
        jobs = [j for j in jobs if is_targeted(j.country, targets)]
        logger.info("Country filter: kept %d of %d jobs in %s (+ Remote)",
                    len(jobs), before, ", ".join(targets))

    new_jobs = [j for j in jobs if j.id not in already_seen]
    ranked = filter_and_rank(new_jobs, matching_cfg)

    # Re-rank by how well the master resume matches each posting, so the jobs
    # generated first are the ones you'll score highest on. min_jd_skills /
    # min_ats optionally drop weak matches.
    ranked = rerank_by_ats(
        ranked, master_resume,
        min_ats=matching_cfg.get("min_ats", 0),
        min_jd_skills=matching_cfg.get("min_jd_skills", 0),
        prefer_engineering=matching_cfg.get("prefer_engineering_roles", True),
    )

    # Country fill order: take the best US roles first, then the next priority
    # countries, then the rest. A STABLE sort by priority index keeps each
    # country's match-quality order intact, so the cap below fills with the US
    # first and only spills into Ireland/Singapore/Canada/etc. to make up a
    # shortfall. Jobs in countries off the list share the lowest priority.
    country_priority = matching_cfg.get("country_priority", [])
    if country_priority:
        pidx = {c: i for i, c in enumerate(country_priority)}
        ranked.sort(key=lambda pair: pidx.get(pair[0].country, len(country_priority)))

    # Per-company cap: keep at most N roles from any one company so a company
    # that posts dozens of near-identical reqs can't eat the whole run. Applied
    # after the priority/quality sort, so it keeps each company's BEST N.
    max_per_company = matching_cfg.get("max_per_company", 0)
    if max_per_company and max_per_company > 0:
        from collections import Counter
        seen_co: Counter = Counter()
        kept: list = []
        for pair in ranked:
            co = pair[0].company or "?"
            if seen_co[co] >= max_per_company:
                continue
            seen_co[co] += 1
            kept.append(pair)
        dropped = len(ranked) - len(kept)
        if dropped:
            logger.info("Per-company cap (%d/company): dropped %d extra role(s)", max_per_company, dropped)
        ranked = kept

    max_per_run = limit if limit is not None else config.get("limits", {}).get("max_jobs_per_run", 100)
    capped = ranked[:max_per_run]
    if country_priority and capped:
        from collections import Counter
        dist = Counter(j.country for j, _ in capped)
        logger.info("Country fill (cap %d): %s", max_per_run,
                    ", ".join(f"{c}={n}" for c, n in sorted(dist.items(), key=lambda x: -x[1])))

    # Split by whether we actually have a description to tailor against. Jobs
    # with none (e.g. a source that omits it, or a failed detail fetch) can't be
    # tailored well, so they go to a fill-in worklist instead -- you paste the
    # description and run `python -m src.tailor_worklist`.
    selected = [(j, s) for (j, s) in capped if len((j.description or "").strip()) >= MIN_DESC_CHARS]
    needs_desc = [(j, s) for (j, s) in capped if len((j.description or "").strip()) < MIN_DESC_CHARS]
    logger.info(
        "%d new jobs matched out of %d scanned; tailoring %d, %d need a pasted description",
        len(ranked), len(jobs), len(selected), len(needs_desc),
    )

    # --- Generation: parallel, since this is network-bound ---
    packages: dict[str, tuple] = {}
    judge_reviews: dict[str, dict] = {}
    judge_enabled = config.get("judge", {}).get("enabled", False) and not dry_run
    # Skills the candidate can HONESTLY claim (master resume + bank + repos). The
    # judge is only allowed to weave in missing keywords from THIS set, so it
    # can't add a JD keyword the candidate lacks (which grounding would miss,
    # since the keyword is in the posting).
    evidence_skills = skills_in(json.dumps(master_resume), bullet_bank, json.dumps(github_repos))

    def _generate(pair):
        job, score = pair
        if dry_run:
            pkg = stub_application_package(master_resume, job.title, job.company)
            return job, score, pkg, None
        pkg = generate_application_package(
            master_resume=master_resume,
            job_title=job.title,
            company=job.company,
            job_description=job.description,
            api_key=deepseek_key,
            model=model,
            github_repos=github_repos,
            bullet_bank=bullet_bank,
            reasoning=reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )
        # Second pass: senior-recruiter/hiring-manager judge rewrites the
        # experience bullets in XYZ form, weaving in the ATS-missing keywords.
        review = None
        if pkg is not None and judge_enabled:
            merged_pre = merge_tailored_content(
                master_resume, pkg, github_repos,
                job_text=f"{job.title} {job.description}",
            )
            # Only keywords the posting wants AND the candidate actually has,
            # but that aren't on the resume yet -- honest gaps to close.
            missing = [k for k in score_resume(merged_pre, job.description)["missing"]
                       if k in evidence_skills]
            review = judge_and_rewrite(
                master_resume=master_resume, github_repos=github_repos,
                bullet_bank=bullet_bank, job_title=job.title, company=job.company,
                job_description=job.description, experience_bullets=pkg.experience_bullets,
                missing_keywords=missing, api_key=deepseek_key, model=model,
                current_summary=pkg.summary,
                reasoning=reasoning, temperature=temperature, max_tokens=max_tokens,
            )
            if review and review["experience"]:
                pkg.experience_bullets.update(review["experience"])  # swap in rewritten bullets
            if review and review.get("summary"):
                pkg.summary = review["summary"]  # stop-the-scroll summary rewrite
        return job, score, pkg, review

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_generate, pair) for pair in selected]
        for done, future in enumerate(as_completed(futures), 1):
            job, score, pkg, review = future.result()
            if pkg is None:
                logger.warning("Generation failed for %s -- skipping", job.id)
                continue
            packages[job.id] = (job, score, pkg)
            if review:
                judge_reviews[job.id] = review
            if done % 10 == 0:
                logger.info("Generated %d/%d packages", done, len(futures))

    # --- Rendering: sequential, CPU-bound ---
    queue: list[dict] = []
    referral_entries: list[dict] = []
    resume_dir = config["resume"]["output_dir"]
    letter_dir = config.get("cover_letter", {}).get("output_dir", "output/cover_letters")
    want_letters = config.get("cover_letter", {}).get("enabled", True)
    include_summary = config.get("resume", {}).get("include_summary", False)
    force_one_page = config.get("resume", {}).get("force_one_page", True)

    # Copy finished PDFs into Downloads/Desktop (job-applications/) for manual
    # upload. Empty when those folders don't exist (e.g. CI) or handoff_dirs: [].
    handoff_dirs = handoff.resolve_dirs(config)
    handoff_count = 0

    grounding_reports = []

    for job, score, pkg in packages.values():
        # One subfolder per company; filename is the job id within it.
        # e.g. output/resumes/stripe/4512345.pdf
        company_dir = _slug(job.company)
        job_file = _slug(job.id.split(":")[-1])
        # Deliver nested by country: <Country>/<company>__<role>__<jobid>. The
        # job id keeps each per-job folder unique so two roles with the same
        # title at one company can't overwrite each other.
        country_dir = _slug(job.country or "Remote")
        handoff_folder = f"{country_dir}/{company_dir}__{handoff.slug(job.title)}__{job_file}"

        # Verify before rendering -- a fabricated metric shouldn't reach a PDF
        # unflagged just because the LaTeX compiled fine. Covers the cover
        # letter and LinkedIn drafts too, not just the resume body.
        corpus = build_source_corpus(master_resume, github_repos, job.description, bullet_bank)
        report = verify_grounding(pkg.groundable_texts(), corpus, job_id=f"{job.company} — {job.title}")
        grounding_reports.append(report)
        if report.severity == "high":
            logger.warning(
                "UNVERIFIED NUMBERS in %s @ %s: %s -- check before sending",
                job.title, job.company, ", ".join(report.unsupported_numbers),
            )
        if pkg.job_analysis.get("gaps"):
            logger.info("Gaps for %s: %s", job.company, "; ".join(pkg.job_analysis["gaps"][:3]))
        merged = merge_tailored_content(
            master_resume, pkg, github_repos, include_summary,
            job_text=f"{job.title} {job.description}",
        )
        pdf_path = f"{resume_dir}/{company_dir}/{job_file}.pdf"
        if not render_resume(merged, config["resume"]["template"], pdf_path, fit_one_page=force_one_page):
            logger.warning("Skipping %s -- resume render failed", job.id)
            continue

        ats = score_resume(merged, job.description)

        letter_pdf = None
        if want_letters and pkg.cover_letter:
            letter_pdf = f"{letter_dir}/{company_dir}/{job_file}.pdf"
            if not render_cover_letter(
                pkg, candidate, job.company,
                config.get("cover_letter", {}).get("template", "templates/cover_letter.tex.j2"),
                letter_pdf,
            ):
                logger.warning("Cover letter render failed for %s", job.id)
                letter_pdf = None

        queue.append({
            "job_id": job.id, "title": job.title, "company": job.company,
            "location": job.location, "country": job.country or "Remote",
            "score": score, "apply_url": job.apply_url,
            "source": job.source,
            "resume_pdf": pdf_path,
            "cover_letter_pdf": letter_pdf,
            "cover_letter_txt": str(Path(letter_pdf).with_suffix(".txt")) if letter_pdf else None,
            "ats_score": ats["score"], "ats_matched": ats["matched"], "ats_missing": ats["missing"],
            "handoff_folder": handoff_folder,
        })
        referral_entries.append({
            "company": job.company, "title": job.title, "url": job.url,
            "note": pkg.referral_note, "dm": pkg.referral_dm,
            "note_len": len(pkg.referral_note),
        })
        if handoff_dirs:
            missing_md = missing_skills_md(job.title, job.company, job.apply_url, ats,
                                           evidence_skills, judge_reviews.get(job.id))
            handoff.deliver(handoff_folder, pdf_path, letter_pdf, handoff_dirs,
                            extra_texts={"missing_skills.md": missing_md})
            handoff_count += 1
        already_seen.add(job.id)

    queue_path = config.get("apply_queue_path", "data/apply_queue.json")
    Path(queue_path).parent.mkdir(parents=True, exist_ok=True)
    Path(queue_path).write_text(json.dumps({"queue": queue}, indent=2))

    _write_apply_index(queue, config.get("output", {}).get("apply_index", "output/apply_index.md"),
                       handoff_dirs, country_priority=matching_cfg.get("country_priority", []))
    # Aggregate ATS/judge overviews stay in the repo output/ -- the per-job
    # missing_skills.md in each Desktop folder carries this info per job.
    _write_ats_report(queue, config.get("output", {}).get("ats_report", "output/ats_report.md"), None)
    _write_judge_report(judge_reviews, packages,
                        config.get("output", {}).get("judge_review", "output/judge_review.md"), None)

    worklist_path = config.get("output", {}).get("needs_description", "output/needs_description.yaml")
    if _write_needs_description(needs_desc, worklist_path, handoff_dirs):
        logger.info(
            "%d job(s) had no fetchable description -> %s. Paste each description "
            "in, then run `python -m src.tailor_worklist %s`.",
            len(needs_desc), worklist_path, worklist_path,
        )

    flagged = write_grounding_report(
        grounding_reports, config.get("quality", {}).get("grounding_report", "output/grounding_review.md")
    )
    if flagged:
        logger.warning(
            "%d/%d packages have claims that could not be traced to your sources "
            "-- see output/grounding_review.md",
            flagged, len(grounding_reports),
        )

    referrals_dir = config.get("referrals", {}).get("output_dir", "output/referrals")
    _write_referrals(referral_entries, referrals_dir)
    # A dry run is a preview -- don't burn the seen-log, or the real run would
    # skip everything the preview touched.
    if not dry_run:
        save_seen(seen_log_path, already_seen)

    if handoff_count and handoff_dirs:
        logger.info("Copied %d resume set(s) to: %s", handoff_count,
                    ", ".join(str(d) for d in handoff_dirs))

    # --- Token usage + estimated cost for this session (and running total). ---
    # Skip on dry runs -- no API calls were made, so there's nothing to log.
    if not dry_run:
        usage.finalize(config, logger, extra={
            "flow": "scan", "jobs_tailored": len(queue),
            "profile": config.get("quality", {}).get("profile", "")})

    logger.info(
        "Done: %d tailored applications. apply_index.txt = copy-paste links (one "
        "per line); apply_index.md = full table of links + which PDF to upload; "
        "referral drafts in %s/",
        len(queue), referrals_dir,
    )


def _write_needs_description(needs_jobs: list[tuple], path: str, handoff_dirs: list) -> bool:
    """
    Write a paste-proof fill-in worklist for jobs whose description couldn't be
    fetched: company/title/url plus a marked area to paste the description into.
    Dropped in output/ and each handoff dir. Returns True if anything written.
    """
    if not needs_jobs:
        return False
    text = handoff.render_worklist([
        {"company": job.company, "title": job.title, "url": job.apply_url}
        for job, _score in needs_jobs
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    fname = Path(path).name
    for d in handoff_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(text, encoding="utf-8")
    return True


def _write_judge_report(judge_reviews: dict, packages: dict, path: str, handoff_dirs: list | None = None) -> None:
    """
    What the judge agent found and fixed per resume: the red flags a hiring
    manager would catch, and the keywords it wove into the rewritten (XYZ)
    experience bullets and summary. Written to output/ and each handoff dir.
    """
    if not judge_reviews:
        return
    handoff_dirs = handoff_dirs or []
    lines = [
        "# Recruiter / hiring-manager review",
        "",
        "A senior-recruiter pass flagged what a hiring manager would catch; the",
        "experience bullets and summary were then rewritten in Google XYZ form",
        "(accomplished X, measured by Y, by doing Z) with these keywords woven in.",
        "Skills, projects, and education were left untouched.",
        "",
    ]
    for job_id, review in judge_reviews.items():
        entry = packages.get(job_id)
        title = f"{entry[0].company} · {entry[0].title}" if entry else job_id
        lines.append(f"## {title}")
        flags = review.get("red_flags") or []
        added = review.get("keywords_added") or []
        lines.append("**Red flags fixed:**")
        lines += [f"- {f}" for f in flags] or ["- (none flagged)"]
        lines.append(f"**Keywords woven in:** {', '.join(added) if added else '(none)'}")
        if review.get("summary"):
            lines.append(f"**Summary rewritten to:** {review['summary']}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    fname = Path(path).name
    for d in handoff_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(text, encoding="utf-8")
    logger.info("Judge reviewed %d resumes -> %s", len(judge_reviews), path)


def _write_ats_report(queue: list[dict], path: str, handoff_dirs: list | None = None) -> None:
    """
    Per-job ATS keyword-match: score + which posting skills the resume is
    MISSING (the actionable part). Written to output/ and each handoff dir.
    """
    if not queue:
        return
    handoff_dirs = handoff_dirs or []
    scored = [q for q in queue if q.get("ats_score") is not None]
    lines = [
        "# ATS keyword match",
        "",
        "Of the skills each posting names, how many appear in your tailored",
        "resume. `missing` = the posting asks for it and your resume doesn't --",
        "add it (only if you truly have it; the grounding check still applies).",
        "",
    ]
    for q in sorted(scored, key=lambda x: x["ats_score"], reverse=True):
        missing = ", ".join(q.get("ats_missing") or []) or "none 🎉"
        lines += [
            f"## {q['ats_score']}%  —  {q['company']} · {q['title']}",
            f"- missing: {missing}",
            "",
        ]
    text = "\n".join(lines) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    fname = Path(path).name
    for d in handoff_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(text, encoding="utf-8")
    if scored:
        avg = round(sum(q["ats_score"] for q in scored) / len(scored))
        logger.info("ATS match: avg %d%% across %d resumes -> %s", avg, len(scored), path)


def _write_apply_index(queue: list[dict], path: str, handoff_dirs: list | None = None,
                       country_priority: list | None = None) -> None:
    """
    A copy-paste-friendly index of every generated package: score, company,
    role, the apply link to open, and which files to upload / paste. This is
    the "here are the links" page for applying by hand.
    """
    if not queue:
        return
    handoff_dirs = handoff_dirs or []
    # When copies go to the Desktop, reference the flat Desktop filenames
    # (company_jobid_resume.pdf) so each row matches the file sitting next to
    # the index. Otherwise reference the repo output/ paths.
    desktop_names = bool(handoff_dirs)
    def _refs(q: dict) -> tuple[str, str]:
        if desktop_names and q.get("handoff_folder"):
            folder = q["handoff_folder"]                 # <Country>/<company>__<role>__<id>
            return f"{folder}/resume.pdf", (f"{folder}/cover_letter.txt" if q.get("cover_letter_txt") else "-")
        return q["resume_pdf"], (q.get("cover_letter_txt") or "-")

    # Grouped by country (in the configured fill-priority order -- US first --
    # then any other country alphabetically), then by match score within each.
    pidx = {c: i for i, c in enumerate(country_priority or [])}
    ordered = sorted(queue, key=lambda x: (
        pidx.get(x.get("country", "Remote"), len(pidx)),
        x.get("country", "Remote"),
        -(x.get("ats_score") or 0), -(x.get("score") or 0),
    ))
    lines = [
        "# Apply index",
        "",
        "For each row: open the apply link, upload the resume PDF, and paste the",
        "cover-letter .txt into the form. Grouped by country, then match score.",
        "",
        "| ATS | Country | Company | Role | Apply link | Resume PDF | Cover letter (.txt) |",
        "|--:|---|---|---|---|---|---|",
    ]
    for q in ordered:
        resume_ref, cover_ref = _refs(q)
        ats = f"{q['ats_score']}%" if q.get("ats_score") is not None else "-"
        lines.append(
            f"| {ats} | {q.get('country','Remote')} | {q['company']} | {q['title']} | "
            f"<{q['apply_url']}> | `{resume_ref}` | `{cover_ref}` |"
        )
    text = "\n".join(lines) + "\n"

    # Plain-text companion: bare apply links, one per line, grouped under a header
    # per country, so you can copy-paste a URL straight out (no markdown).
    txt_lines = [
        f"APPLY LINKS -- {len(queue)} jobs, grouped by country, then match score.",
        "Open each link, upload the resume PDF, paste the cover-letter .txt.",
        "",
    ]
    current_country = None
    for q in ordered:
        country = q.get("country", "Remote")
        if country != current_country:
            current_country = country
            txt_lines += ["=" * 68, f"  {country}", "=" * 68, ""]
        resume_ref, cover_ref = _refs(q)
        ats = f"{q['ats_score']}%" if q.get("ats_score") is not None else "-"
        txt_lines += [
            f"[{ats}] {q['company']} -- {q['title']}",
            q["apply_url"],                      # bare URL on its own line = easy to copy
            f"  resume: {resume_ref}",
            f"  cover:  {cover_ref}",
            "",
        ]
    txt = "\n".join(txt_lines) + "\n"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    txt_path = Path(path).with_suffix(".txt")
    txt_path.write_text(txt, encoding="utf-8")
    fname, txt_fname = Path(path).name, txt_path.name
    for d in handoff_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(text, encoding="utf-8")
        (d / txt_fname).write_text(txt, encoding="utf-8")
    where = str(txt_path) + (f" (+ {len(handoff_dirs)} handoff dir)" if handoff_dirs else "")
    logger.info("Wrote apply index (%d jobs) -> %s and %s", len(queue), path, where)


def _write_referrals(entries: list[dict], out_dir: str) -> None:
    """
    One PLAIN-TEXT file per company (<out_dir>/<company>.txt), each holding every
    role drafted for that company. Plain text (no markdown) so the note and DM
    can be copied straight into LinkedIn, and the profile/job URL sits bare on
    its own line. Deliberately not automated -- see README on LinkedIn's own
    weekly/monthly caps, far lower than this pipeline's throughput.
    """
    if not entries:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_company: dict[str, list[dict]] = {}
    for e in entries:
        by_company.setdefault(e["company"], []).append(e)

    bar = "=" * 68
    for company, company_entries in sorted(by_company.items()):
        lines = [
            f"LinkedIn referral drafts -- {company}",
            "",
            "Send these MANUALLY. Connection-request notes are capped at 300",
            "characters by LinkedIn; the count is shown so you can verify before",
            "pasting.",
            "",
        ]
        for e in sorted(company_entries, key=lambda x: x["title"]):
            lines += [
                bar,
                e["title"],
                e["url"],                       # bare URL on its own line = easy to copy
                "",
                f"Connection note ({e['note_len']}/300 chars):",
                e["note"],
                "",
                "Follow-up DM (after they accept):",
                e["dm"],
                "",
            ]
        lines.append(bar)
        (out / f"{_slug(company)}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(
        "Wrote %d referral drafts (.txt) across %d companies -> %s/",
        len(entries), len(by_company), out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke test: run scan -> render -> outputs with stub content, no DeepSeek calls / no API key.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap how many jobs to generate this run (overrides limits.max_jobs_per_run).")
    args = parser.parse_args()
    run(args.config, dry_run=args.dry_run, limit=args.limit)
