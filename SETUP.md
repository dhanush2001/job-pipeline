# SETUP CHECKLIST

Work top to bottom. Steps 1–6 are required; nothing runs without them.

---

## 1. DeepSeek API key  — REQUIRED
platform.deepseek.com → create key. Locally: `export DEEPSEEK_API_KEY=...`

## 2. Your resume content  — REQUIRED
```
cp data/master_resume.example.yaml data/master_resume.yaml
```
Replace **all** of it. It currently says Jane Doe at Acme Corp.

**Give every experience entry a stable `id`** (e.g. `acme-intern-2025`).
That id is how tailored bullets get matched back to the right job — if ids
change between runs, bullets land under the wrong employer.

Put your real metrics in here. The grounding checker only accepts numbers it
can trace to this file, your repos, or the posting — so a number that belongs
on your resume but isn't in this file will get flagged every single run.

## 3. Your LaTeX template  — REQUIRED
`templates/resume.tex.j2` is a complete, working reference that renders the
sample resume end to end — you can use it as-is. To adapt your own `.tex`,
convert its placeholders to the LaTeX-safe delimiters this repo uses (Jinja2's
default `{{ }}`/`{% %}` collide with LaTeX):
- `{{ value }}` → `\VAR{value}`
- `{% if x %}...{% endif %}` → `\BLOCK{if x}...\BLOCK{endif}`

Converting an existing `.tex` is the fiddliest step; start from `resume.tex.j2`
and swap in your preamble/section layout a piece at a time.

## 4. config.yaml  — REQUIRED
```
cp config.example.yaml config.yaml
```
Fill in:
- `candidate:` — your real name, email, phone, LinkedIn (currently Jane Doe)
- `github.username` — **currently blank, so GitHub projects are OFF**
- `sources.*` — company slugs you want tracked, per ATS platform
  (`greenhouse`, `lever`, `ashby`, `smartrecruiters`, `workable`). The slug is
  the token in the careers URL: `boards.greenhouse.io/stripe` → `stripe`.
  Currently `examplecorp` / `anotherexample`, which don't exist.
- `matching.*` — tune keywords to your actual target roles
- `sources.adzuna` — set `countries` / `queries` to your targets (needs a free
  Adzuna key; see Optional below)

## 5. GitHub repo  — REQUIRED
Push to a **private** repo. `config.yaml` and `data/master_resume.yaml` are
deliberately committed (Actions needs them), so the repo holds your contact
details and resume. Private is not optional.

## 6. Actions secrets  — REQUIRED for scheduled runs
Settings → Secrets and variables → Actions:
- `DEEPSEEK_API_KEY` (required)
- `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` (optional — general job boards)
- `PIPELINE_CONTACT` (optional — your email, sent in the User-Agent)

`GITHUB_TOKEN` is automatic; don't create it. Only needed if you want
**private** repos considered, in which case add a fine-grained PAT instead.

---

## First run — do this before trusting anything

```
pip install -r requirements.txt

# Small batch first
python -m src.main --config config.yaml --limit 5
```

Then check, in this order:

1. **Open a generated PDF.** Does it look like your resume? This is where a
   broken template conversion shows up.
2. **Read `output/grounding_review.md`.** Anything flagged that you did NOT
   write? That's a fabrication — do not send it.
3. **Read `output/apply_index.md`.** One row per job: score, company, role,
   apply link, and which resume/cover-letter files to upload. This is your
   checklist — open each link and submit the files yourself.
4. **Read one cover letter.** Generic filler means the prompt needs tuning.

Finished PDFs are also copied into a `job-applications/` folder on your
Desktop (see `output.handoff_dirs` in config), one subfolder per job, each
with `resume.pdf`, `cover_letter.pdf`, and `missing_skills.md`.

### Pasting jobs yourself

For postings the scanners can't fetch (or ones you found elsewhere):

```
python -m src.tailor_one          # paste ONE company + title + description
python -m src.tailor_worklist --new 5   # MANY: write a file with 5 paste slots
python -m src.tailor_worklist output/needs_description.txt   # tailor them all
```

For the many-jobs file, paste a whole job posting into each slot and separate
them with a line of `===`. You don't type the company or title — they're pulled
from each posting automatically (add a `Company:` / `Title:` line at the top of
a slot to override). Each job still gets its own Desktop folder.

---

## Known-unverified — expect to fix these on first real run

Testing here was offline, so these are the parts most likely to break:

| What | Why it's unverified |
|---|---|
| Greenhouse / Lever / Ashby / SmartRecruiters / Workable parsers | Tested against fixtures, not every live board shape |
| **Lever parser** | No test at all — least trustworthy code in the repo |
| GitHub repo parser | Live call was rate-limited (403); fixture is structural only |
| Output quality | Compare `fast` vs `high` on 5 jobs each before trusting a profile |

None of this is hypothetical-caution — it's the specific list of things I
could not run.

---

## Optional

- **Adzuna key** (developer.adzuna.com, free) for general job boards. Without
  it you only get the Greenhouse/Lever companies you list by name.
- **`quality.profile`** is `high` (V4 Pro + thinking, ~$14/mo at 100/day).
  Drop to `fast` (~$2.50) if quality is fine without it.
- **Cron time** is 13:37 UTC weekdays. It's deliberately off-the-hour and
  outside DeepSeek's peak-pricing windows (01:00–04:00, 06:00–10:00 UTC).
