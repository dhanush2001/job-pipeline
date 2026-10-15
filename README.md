# Job Application Pipeline

Scans job postings, tailors your resume per posting with DeepSeek, renders it
to PDF from a LaTeX template, scores it against the posting for ATS keyword
match, and drops a ready-to-upload folder on your Desktop for each job. It
never touches an application form — everything it produces is for you to
review and submit by hand.

## What it does, end to end

```
scan_jobs.py -> match.py -> tailor_resume.py -> judge.py -> render_resume.py -> handoff.py
(public APIs)  (keyword     (DeepSeek writes   (recruiter/  (Jinja2 -> LaTeX   (per-job folder:
                filter +     tailored resume    XYZ rewrite  -> one-page PDF)    resume.pdf,
                ATS rank)     + cover letter)    of bullets)                     cover_letter,
                                                                                 missing_skills.md)
```

Every run produces:

- **`output/apply_index.md`** — one row per job: ATS score, company, role,
  apply link, and which files to upload. Your checklist.
- **A per-job folder** copied into `job-applications/<Country>/` under each path
  in `output.handoff_dirs` (Desktop by default) — one country folder per market
  you target, with a `<Company>__<Role>` folder inside holding `resume.pdf`,
  `cover_letter.pdf` / `.txt`, and `missing_skills.md`. Roles located in
  countries outside `matching.target_countries` are dropped before tailoring.
- **`output/ats_report.md`** — per-job keyword match and the keywords you're
  missing.
- **`output/judge_review.md`** — what the recruiter/hiring-manager pass flagged
  and rewrote.
- **`output/grounding_review.md`** — any number or claim that couldn't be
  traced to your real content (read this before sending anything).
- **Referral drafts** — one plain-text (`.txt`) file per company in `output/referrals/`.

There is no browser step and nothing is ever submitted. The pipeline calls
only public JSON APIs (Greenhouse, Lever, Ashby, SmartRecruiters, Workable,
Adzuna, GitHub) and the DeepSeek API, so it runs anywhere, including GitHub
Actions.

## Three ways to feed it jobs

1. **Scanned automatically** — list company slugs under `sources:` in
   `config.yaml` and run `python -m src.main`.
2. **Paste one job** — `python -m src.tailor_one`, then paste a company, title,
   and description. Produces the same per-job folder.
3. **Paste many jobs into a file** — `python -m src.tailor_worklist --new 5`
   writes a file with paste slots; paste a whole job posting into each slot
   (separate them with a `===` line), then
   `python -m src.tailor_worklist output/needs_description.txt` tailors them all.
   You don't type the company or title — they're pulled from each posting
   automatically (add a `Company:` / `Title:` line at the top of a slot to
   override). The same command also reads the structured `### JOB` worklist that
   a scan writes for postings it couldn't fetch; the format is auto-detected.

## Why these job sources

Greenhouse, Lever, Ashby, SmartRecruiters, and Workable all have public,
documented JSON APIs — no login, no scraping, no ToS problem. Calling them is
the intended use; there's no "bot" to be flagged as. LinkedIn and Indeed offer
nothing comparable and prohibit automated scraping in their terms, so Adzuna (a
legitimate aggregator API, free tier) covers general-board postings instead.

## Referrals: what scales and what doesn't

**Generation scales fine. Referral outreach does not**, and the ceiling isn't a
design choice — it's LinkedIn's own limit. Referral drafts are written for you;
sending them is manual and deliberately so.

| | Realistic ceiling |
|---|---|
| Tailored resumes + cover letters generated | 100+/day, ~$2.50/month |
| LinkedIn connection requests | ~100/**week** (up to ~200 on Premium with strong account health) |
| Personalized notes, **Premium** | no monthly quota |
| Personalized notes, free account | ~5/**month** |

**On Premium, the note quota is not your constraint — the weekly connection cap
is.** Free accounts get roughly five personalized notes per month, which makes
per-company referral notes basically impossible. Premium removes that quota, so
you can attach a personalized note to every request. What still binds is the
weekly invitation limit: around 100/week standard, up to ~200/week for
established accounts with good acceptance rates. Acceptance rates fall as volume
climbs, so pushing the ceiling is self-defeating.

Practical shape at 100 applications/day (500/week): you cannot request a
referral for every one. Send notes for the ~20/day you care most about, which
lands comfortably inside the weekly cap, and let the rest be direct
applications. Referrals convert far better than cold applications — sending
those by hand is the right call, and it's also the only way the numbers work.

## ATS scoring and job ranking

After the keyword filter, jobs are re-ranked by how well your skills (master
resume + bullet bank) match each posting, so the ones generated first score
highest. Two knobs in `matching:` control selection:

- `min_jd_skills` — drop postings naming fewer than N skills (avoids non-tech
  roles that trivially hit 100%).
- `min_ats` — only keep postings you match ≥ this % (0 = keep all, just re-sort).
- `prefer_engineering_roles` — rank genuine SWE/ML/data/DevOps roles first and
  drop business/customer-facing roles that pure keyword-matching surfaces.

Each generated resume is scored against its posting; the matched and missing
keywords go into `output/ats_report.md` and each job's `missing_skills.md`. The
missing list is honest about what you'd need to add — it does **not** invent
skills you don't have.

## The judge pass

With `judge.enabled: true`, a second DeepSeek pass acts as a senior recruiter
and hiring manager: it flags red flags and missing keywords, then rewrites the
experience bullets in Google's XYZ form (accomplished X, measured by Y, by
doing Z). It only weaves in keywords that are already supported by your
evidence — it never introduces a skill or a metric you don't have. See
`output/judge_review.md` for what it caught. It doubles per-job API cost (still
cheap on `fast`).

## Cover letters

One DeepSeek call produces the tailored resume, cover letter, and referral
drafts together. This is a cost decision: the input (master resume + GitHub
repos + job description) is ~5,900 tokens and separate calls would re-send it
each time. Each letter is written as a PDF **and** a `.txt` beside it — the
text version is the one you'll use most, since most ATS forms have a paste-in
textarea rather than an upload field.

## Optimizing for quality

Three levers, in the order they actually matter:

**1. The prompt (biggest, free).** The system prompt frames the model as a
recruiter, and makes it fill in `job_analysis` FIRST — hard requirements, the
posting's own terminology, and honest `gaps` where you have no supporting
evidence. Writing that analysis before the content forces tailoring to be about
*this* job instead of generic polish. It includes good/bad bullet examples and
explicit anti-patterns ("leveraged cutting-edge technologies"), and it forbids
keyword-stuffing, which fails the first phone screen even when it passes the ATS
filter.

**2. Quality profile.** `quality.profile` bundles model and reasoning:

| profile | model | reasoning | $/month @ 100/day |
|---|---|---|---|
| `fast` | V4 Flash | disabled | $2.52 |
| `balanced` | V4 Flash | low | $3.10 |
| `high` | **V4 Pro** | high | **$14.09** |

`high` is the strongest tier and high-effort thinking lets it work through the
requirement-to-evidence mapping properly. Note `high` sets `max_tokens: 16000` —
thinking traces plus a full JSON payload truncate at smaller limits, and
truncated JSON fails the parse entirely.

**3. Grounding verification (`src/quality.py`).** The real failure mode of LLM
resume tailoring isn't clumsy phrasing, it's invented facts — "reduced p99
latency by 47%" reads beautifully and is a serious problem if your master resume
never said 47%. The prompt forbids it; this checks it mechanically, because at
100/day nobody reads every line. Every number and proper noun in the generated
content (resume **and** cover letter) is traced back to your master resume, your
GitHub repos, or the job posting. Anything untraceable lands in
`output/grounding_review.md`, numbers first. Clean packages don't appear at all.
It flags rather than blocks — but check every flagged number before sending.

## Thinking mode: set it explicitly

**DeepSeek V4 enables thinking by default at `high` effort.** Send no thinking
parameter and you get high-effort chain-of-thought on every call, billed at the
output rate. Thinking mode also ignores `temperature`, `top_p`,
`presence_penalty` and `frequency_penalty` — per DeepSeek's docs these do not
error, they simply have no effect. This pipeline therefore always sends an
explicit toggle, defaulting to `reasoning: "disabled"`, and only sends
`temperature` when it can actually take effect.

| `reasoning` | $/month | vs disabled |
|---|---|---|
| `disabled` (this pipeline's default) | $2.52 | — |
| `low` | $3.10 | +$0.58 |
| `high` (**the API's own default**) | $4.69 | +$2.18 |
| `max` | $8.32 | +$5.81 |

If you are optimizing for cost, `disabled` is right — tailoring is selection and
rewriting, not a reasoning problem. If you are optimizing for quality, the
`high` profile turns thinking on deliberately at the Pro tier.

## Prompt caching (don't reorder the prompt)

`_build_user_prompt` puts stable content (master resume, GitHub repos, bullet
bank) first and the per-job description last. DeepSeek bills cached prefixes at
$0.007/1M versus $0.22/1M for a miss, and caching keys on a shared prefix — so
with this ordering ~92% of each prompt is cacheable across a run. Measured
effect at 100 jobs/day: **$4.76/month → $2.52/month**. There's a test asserting
the order. A separate local on-disk response cache (`deepseek.cache`) skips the
call entirely on re-runs while you iterate on the template.

## GitHub-driven projects

If you set `github.username`, the pipeline pulls your real repositories (name,
description, language, topics, stars, last-push date, README excerpt) once per
run and hands them to the tailoring step. The model picks the **2-4 most
relevant repos per role** and writes bullets grounded in that data — so a
backend posting and an ML posting get different Projects sections. The tailored
list *replaces* the `projects:` block in your master resume; if the fetch fails
or returns nothing, the master list is used unchanged.

**Set `GITHUB_TOKEN`.** Unauthenticated GitHub API access is 60 requests/hour
*per IP*, and this module spends one request on the repo list plus one per
README. On shared CI runners that limit is routinely already exhausted. The
workflow passes Actions' automatic `secrets.GITHUB_TOKEN`, which is enough for
public repos; use a fine-grained PAT for private repos.

## Setup

**See [SETUP.md](SETUP.md) for the step-by-step checklist.** The short version:

1. **DeepSeek API key** — platform.deepseek.com. `export DEEPSEEK_API_KEY=...`
2. **(Optional) Adzuna key** — developer.adzuna.com, free tier. Only for the
   general-board source.
3. **Adapt your LaTeX template** to LaTeX-safe Jinja2 delimiters: `\VAR{...}`
   for values, `\BLOCK{if ...}...\BLOCK{endif}` for logic.
4. **Fill in your resume**: copy `data/master_resume.example.yaml` to
   `data/master_resume.yaml`. Give every experience entry a stable `id` — that's
   how tailored bullets map back to the right job.
5. **Copy `config.example.yaml` to `config.yaml`**, fill in contact info,
   company slugs per ATS platform, and matching keywords.
6. **Install**: `pip install -r requirements.txt` (LaTeX too:
   `texlive-latex-base texlive-latex-extra latexmk`).
7. **Set `PIPELINE_CONTACT`** to your email (optional but courteous).

Run `python -m src.doctor` first — it reports what's configured, what's
missing, and what the next run would do, without spending any API credits.

## Running it

**Locally first**, to confirm it produces sensible resumes:
```
export DEEPSEEK_API_KEY=...
python -m src.doctor                                 # preflight, no cost
python -m src.main --config config.yaml --limit 5    # small batch
```
Read two or three generated PDFs and `output/grounding_review.md` before
trusting it unattended. Then open `output/apply_index.md`, click each apply
link, and upload the files from the per-job Desktop folder yourself.

**On a schedule** via `.github/workflows/daily-scan.yml`:
1. Push to a **private** GitHub repo (it holds job-search config and personal info).
2. Settings → Secrets and variables → Actions: add `DEEPSEEK_API_KEY`, and
   optionally `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`, `PIPELINE_CONTACT`.
3. Runs weekdays at 13:37 UTC; also triggerable manually.
4. Download the run artifact (tailored PDFs + reports) and apply from it.

## Known limitations

- **Some postings can't be fetched** (Workday's per-tenant markup, Adzuna links
  that land on arbitrary sites). Those show up in `output/needs_description.txt`
  — paste the description and run `python -m src.tailor_worklist`.
- **Spot-check the LLM.** DeepSeek is instructed never to invent experience, and
  grounding checks it mechanically, but read a few tailored resumes before
  running unattended for a week.

## Tests

All offline — no network, no API keys:
```
python tests/test_scan_and_match.py      # API parsing, match/rank, ATS re-rank, role tiers, work-auth
python tests/test_locations.py           # country detection + target-country filtering
python tests/test_new_sources.py         # Ashby / SmartRecruiters / Workable parsers
python tests/test_ats_score.py           # keyword scoring, missing-skills report
python tests/test_judge.py               # judge parsing, keyword grounding
python tests/test_github_projects.py     # repo parsing, skills reconcile, one-page fitter
python tests/test_handoff_worklist.py    # per-job folder layout, worklist parse/generate
python tests/test_cover_letter_and_referral.py  # letters, note cap, cache prefix ordering
python tests/test_dry_run.py             # dry-run makes no API calls
python tests/test_response_cache.py      # on-disk response cache
python tests/test_thinking_mode.py       # thinking toggle, effort mapping, temperature
python tests/test_quality.py             # fabrication detection, grounding report
```
