"""
tailor_resume.py
-----------------
Generates a complete application package per job in ONE DeepSeek call:
tailored resume content, a cover letter, and LinkedIn referral drafts.

Why one call instead of four: the expensive part of the prompt is the input
(master resume + GitHub repos + job description ~ 6k tokens). Splitting into
separate calls for resume / cover letter / referral would re-send all of that
each time and roughly quadruple input spend. One call reuses it.

PROMPT ORDER IS DELIBERATE. DeepSeek bills cached input prefixes at a small
fraction of the miss rate, and caching works on a shared *prefix*. So the
stable content (system prompt, master resume, GitHub repos) goes FIRST and
the per-job content (the job description) goes LAST. Across a 100-job run the
stable prefix is identical every time, so only the tail is a cache miss.
Reordering these keys would silently multiply your bill.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

from src import usage

logger = logging.getLogger(__name__)

DEEPSEEK_API_BASE = "https://api.deepseek.com"
REQUEST_TIMEOUT = 90

# The old "deepseek-chat" / "deepseek-reasoner" aliases were retired in July
# 2026 and folded into the V4 family. "deepseek-v4-flash" is the stable route
# and the cheap tier; "deepseek-v4-pro" is ~3x the price and unnecessary here.
DEFAULT_MODEL = "deepseek-v4-flash"

# --- Thinking mode ---------------------------------------------------------
# CRITICAL DEFAULT: DeepSeek V4 has thinking mode ENABLED BY DEFAULT at "high"
# effort. If you send no thinking parameter at all, every call silently
# generates a chain-of-thought trace billed at the OUTPUT rate ($0.66/1M) and
# adds substantial latency. At 100 calls/day that is real money for no gain on
# a writing-and-selection task.
#
# Also: thinking mode IGNORES temperature, top_p, presence_penalty and
# frequency_penalty. Per DeepSeek's docs these do not error -- they simply
# have no effect -- so a temperature you set alongside thinking is a silent
# no-op. This module therefore only sends temperature when thinking is off.
#
# Chat Completions accepts reasoning_effort of low / high / max
# (medium and xhigh are accepted but both map to high).
VALID_EFFORTS = {"low", "high", "max"}
EFFORT_ALIASES = {"medium": "high", "xhigh": "high"}

# LinkedIn hard caps. The connection-request note is 300 characters on every
# plan; published analyses of large invite datasets put the best-performing
# range well below that, around 120-180, because a note that fills the cap
# reads like an automated pitch. We target the shorter range and validate
# against the hard cap in code, because the model will overshoot.
LINKEDIN_NOTE_HARD_LIMIT = 300
LINKEDIN_NOTE_TARGET = 180
LINKEDIN_DM_LIMIT = 8000

# The résumé shows only the few projects most relevant to each role; more than
# this crowds the page and pushes the resume onto a second page. Enforced in
# code because the prompt's count is a request the model regularly overshoots.
MAX_PROJECTS = 3

SYSTEM_PROMPT = """You are a senior technical recruiter who has screened
thousands of new-grad and early-career engineering resumes. You know what
survives a six-second scan and an ATS keyword filter, and what gets cut.

You receive: a candidate's master resume (JSON), optionally their real GitHub
repositories, optionally a "bullet_bank" (pre-written candidate bullets grouped
by project and angle), and a target job posting.

## Using the bullet_bank
The bullet_bank is a MENU, not finished copy. You MAY pick from it, rephrase it,
and merge it with resume/repo evidence to match this posting's stack. Rules:
- It is a legitimate source of facts: numbers and technologies in it are the
  candidate's real work and may be used.
- NEVER output a placeholder token. Text like "[X]", "[N]", "[~Y%]", "[your
  CPU]", "[your key result]" is an instruction to the candidate, not data. If a
  bullet's value depends on a placeholder you cannot fill from a real source,
  drop that bullet rather than emitting the bracket.
- OBEY the bank's honesty notes. Where it says a project was "used / extended /
  applied" (not built from scratch), keep that framing; do not upgrade it.
- Ignore its meta-commentary (legends, warnings, "how to adapt" section) --
  those are notes to the candidate, never resume content.

## Step 1 — analyze before you write
Fill in "job_analysis" FIRST. Identify what this posting actually rewards:
the hard requirements, the implied seniority, the domain, and the specific
terminology the posting uses. Writing this before the resume content forces
the tailoring to be about THIS job rather than generic polish.

## Step 2 — map evidence to requirements
For each requirement you identified, find the strongest concrete evidence in
the master resume or GitHub repos. If there is no evidence for a requirement,
say so in "gaps" — do not paper over it with vague language.

## GROUNDING RULE — the one that matters most
Every factual claim must trace to the master resume, the GitHub repo data, or
the job posting's own vocabulary. Specifically:
- NEVER invent a metric. If the source says "reduced deploy time", do not
  write "reduced deploy time by 40%". A fabricated number is the single
  fastest way to lose an offer, and it is checked programmatically after you.
- NEVER invent employers, titles, dates, team sizes, or user counts.
- NEVER claim a technology the candidate has no evidence of using.
- You MAY rephrase, reorder, re-scope, and change emphasis freely.
- If evidence is thin, write something honest and general rather than
  something specific and false.

## What good bullets look like
GOOD: "Built an internal deploy tool that cut deploy time, used by the
platform team" — concrete, traceable, scoped.
BAD: "Leveraged cutting-edge technologies to drive impactful outcomes" —
says nothing.
BAD: "Improved system performance by 35%" when no such number is in the
source — fabricated.

Lead each bullet with a strong past-tense verb. Prefer the posting's own
terminology where it honestly describes the candidate's work (that is ATS
keyword alignment). Do not keyword-stuff: a skills list padded with
technologies the candidate has never touched fails the first phone screen.

## Emphasis — bold the keywords (REQUIRED)
In every experience and project bullet, wrap the 2-4 most important technical
terms in **markdown bold** — the concrete skills, tools, and results a recruiter
scans for and that match this posting (e.g. "Built a **FastAPI** service exposing
**REST APIs** backed by **PostgreSQL**"). This emphasis is rendered in the PDF.
Rules: bold specific nouns (languages, frameworks, systems, metrics), never whole
clauses or verbs; at most ~4 bolded spans per bullet; do not bold in the summary,
cover letter, or LinkedIn drafts — bullets only.

## Cover letter
3 short paragraphs, 200-300 words. Paragraph 1: which role and one sentence
on why this specific company/team. Paragraph 2: the single strongest piece of
evidence mapped to their top requirement — concrete, not a resume rehash.
Paragraph 3: brief close. No "I am passionate about", no restating the job
description back at them, no invented enthusiasm about products you were not
told about.

## Length — the resume MUST fit on ONE page
This is a strict layout constraint. Keep every bullet to a single line — about
15-22 words, one idea. Prefer 3 tight bullets over 4 loose ones. Do not pad.
Concise, scannable bullets beat long ones on a six-second scan anyway.

## Projects
Select the 2-3 projects most relevant to THIS role and return them ordered
most-relevant first — never more than 3. Prefer the candidate's GitHub
repositories when they are provided; fall back to the master resume's projects
otherwise. Return only the project name (copied verbatim from the source so it
can be matched back to it) and tailored bullets: dates and tech stack are
filled in from the source data automatically, so do NOT invent them here.

## LinkedIn drafts
referral_note is a CONNECTION REQUEST note: HARD LIMIT 300 characters, aim
120-180. Its only job is to earn the accept — name the role, be human, do not
pitch. referral_dm is the follow-up after they accept, 90-150 words: ask
whether they would be open to referring you, give one concrete relevant
detail, and make it easy to decline.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{
  "job_analysis": {
    "hard_requirements": ["..."],
    "key_terminology": ["..."],
    "seniority_signal": "...",
    "gaps": ["requirements with no supporting evidence"]
  },
  "summary": "AT MOST 2 sentences (~2 resume lines), tailored to this role; lead with the single strongest relevant qualification. No fluff.",
  "skills": ["most relevant first"],
  "experience": [{"id": "<master resume id>", "bullets": ["..."]}],
  "projects": [{"name": "...", "bullets": ["..."]}],
  "cover_letter": "...",
  "referral_note": "...",
  "referral_dm": "..."
}"""


@dataclasses.dataclass
class ApplicationPackage:
    summary: str
    skills: list[str]
    experience_bullets: dict[str, list[str]]
    projects: list[dict[str, Any]]
    # Defaulted so callers that only need resume tailoring keep working.
    cover_letter: str = ""
    referral_note: str = ""
    referral_dm: str = ""
    note_was_truncated: bool = False
    job_analysis: dict[str, Any] = dataclasses.field(default_factory=dict)

    def groundable_texts(self) -> list[str]:
        """
        Every free-text field the model generated, for the grounding checker.
        Includes the cover letter and LinkedIn drafts, not just the resume
        body: a fabricated metric is exactly as damaging in a cover letter as
        in a resume bullet, so all generated prose gets verified.
        """
        texts = [self.summary, self.cover_letter, self.referral_note, self.referral_dm]
        for bullets in self.experience_bullets.values():
            texts.extend(bullets)
        for proj in self.projects:
            texts.extend(proj.get("bullets", []))
        return [t for t in texts if t]


# Backwards-compatible alias: earlier code referred to TailoredContent.
TailoredContent = ApplicationPackage


def _normalize_projects(
    raw_projects: list[dict[str, Any]], max_projects: int = MAX_PROJECTS
) -> list[dict[str, Any]]:
    """
    Keep only named projects, preserve the model's relevance ordering, and cap
    the count. The "2-4 relevant projects" promise is enforced here rather than
    trusted to the prompt, which the model regularly overshoots.
    """
    projects = [
        {"name": p.get("name", ""), "bullets": p.get("bullets", [])}
        for p in raw_projects
        if p.get("name")
    ]
    return projects[:max_projects]


def _build_user_prompt(
    master_resume: dict[str, Any],
    job_title: str,
    company: str,
    job_description: str,
    github_repos: list[dict[str, Any]] | None = None,
    bullet_bank: str = "",
) -> str:
    """
    Key order matters for prompt caching -- stable content first, per-job
    content last. See the module docstring. master_resume, github_repos, and
    the bullet_bank are identical on every call, so they form the cached prefix;
    only target_job changes per posting and therefore goes last.
    """
    payload: dict[str, Any] = {"master_resume": master_resume}
    if github_repos:
        payload["github_repos"] = github_repos
    if bullet_bank:
        payload["bullet_bank"] = bullet_bank
    payload["target_job"] = {
        "title": job_title,
        "company": company,
        "description": job_description[:6000],
    }
    return json.dumps(payload)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
    return text.strip()


def _enforce_note_limit(note: str) -> tuple[str, bool]:
    """
    LinkedIn blocks the send outright past 300 characters, so a too-long note
    is useless rather than merely imperfect. Truncate at a sentence boundary
    where possible so the result still reads like a finished thought.
    """
    note = " ".join(note.split())
    if len(note) <= LINKEDIN_NOTE_HARD_LIMIT:
        return note, False

    cut = note[:LINKEDIN_NOTE_HARD_LIMIT]
    for sep in (". ", "! ", "? "):
        idx = cut.rfind(sep)
        if idx > LINKEDIN_NOTE_HARD_LIMIT * 0.5:
            return cut[: idx + 1].strip(), True
    return cut[: LINKEDIN_NOTE_HARD_LIMIT - 1].rstrip() + "…", True


# --- Local response cache -------------------------------------------------
# On-disk cache of DeepSeek responses, keyed by the EXACT request (model,
# settings, resume, repos, bullet bank, job description). Opt-in, for local
# iteration: when you re-run after tweaking the template or rendering, cached
# jobs return instantly and cost nothing. This is separate from DeepSeek's
# automatic server-side prompt caching -- that trims the bill within one run;
# this skips the call entirely on the next run. Any input change misses.


def _cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_read(cache_dir: str, key: str) -> str | None:
    path = Path(cache_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    except (ValueError, KeyError, OSError):
        return None  # treat a corrupt entry as a miss


def _cache_write(cache_dir: str, key: str, content: str) -> None:
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename so a killed run never leaves a partial
    # file that a later run reads back as a bogus cache hit.
    tmp = directory / f"{key}.json.tmp{os.getpid()}"
    tmp.write_text(json.dumps({"content": content}), encoding="utf-8")
    os.replace(tmp, directory / f"{key}.json")


def _thinking_params(reasoning: str, temperature: float) -> dict[str, Any]:
    """
    Build the thinking-mode parameters, and only send temperature when it can
    actually take effect. Always sends an explicit thinking toggle -- omitting
    it means "enabled at high effort", which is not a default you want to
    inherit by accident on a 100-call run.
    """
    reasoning = (reasoning or "disabled").lower()
    reasoning = EFFORT_ALIASES.get(reasoning, reasoning)

    if reasoning in ("disabled", "off", "none", "non-think"):
        # Thinking off -> temperature is honored.
        return {"thinking": {"type": "disabled"}, "temperature": temperature}

    if reasoning not in VALID_EFFORTS:
        logger.warning(
            "Unknown reasoning value %r -- falling back to disabled. "
            "Valid: disabled, low, high, max.", reasoning,
        )
        return {"thinking": {"type": "disabled"}, "temperature": temperature}

    # Thinking on -> temperature would be silently ignored, so don't send it.
    logger.debug("Thinking mode %s enabled; temperature ignored by the API", reasoning)
    return {"thinking": {"type": "enabled"}, "reasoning_effort": reasoning}


def generate_application_package(
    master_resume: dict[str, Any],
    job_title: str,
    company: str,
    job_description: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    github_repos: list[dict[str, Any]] | None = None,
    bullet_bank: str = "",
    max_retries: int = 2,
    reasoning: str = "disabled",
    temperature: float = 0.4,
    max_tokens: int | None = None,
    cache_enabled: bool = False,
    cache_dir: str = ".cache/deepseek",
) -> ApplicationPackage | None:
    """
    reasoning: "disabled" (default) or one of low / high / max.

    Default is disabled because this is a selection-and-rewriting task, not a
    reasoning problem -- and because leaving it unset would silently mean
    "high" (see module constants above).
    """
    if not api_key:
        logger.error("No DeepSeek API key configured")
        return None

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(
                    master_resume, job_title, company, job_description,
                    github_repos, bullet_bank,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    payload.update(_thinking_params(reasoning, temperature))
    if max_tokens:
        payload["max_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    cache_key = _cache_key(payload) if cache_enabled else None

    raw = None
    if cache_key:
        raw = _cache_read(cache_dir, cache_key)
        if raw is not None:
            logger.info("Cache hit for %s @ %s -- skipping API call", job_title, company)

    if raw is None:
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    f"{DEEPSEEK_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    logger.warning("DeepSeek rate limited (attempt %d) for %s", attempt + 1, job_title)
                    continue
                resp.raise_for_status()
                data = resp.json()
                usage.record(data.get("usage"))
                raw = data["choices"][0]["message"]["content"]
                if cache_key:
                    _cache_write(cache_dir, cache_key, raw)
                break
            except (requests.RequestException, KeyError, IndexError) as exc:
                logger.warning(
                    "DeepSeek call failed (attempt %d/%d) for %s @ %s: %s",
                    attempt + 1, max_retries + 1, job_title, company, exc,
                )

    if raw is None:
        logger.error("Giving up on %s @ %s after %d attempts", job_title, company, max_retries + 1)
        return None

    try:
        parsed = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        logger.error("Unparseable DeepSeek response: %s\nRaw: %s", exc, raw[:500])
        return None

    experience_bullets = {
        entry.get("id"): entry.get("bullets", [])
        for entry in parsed.get("experience", [])
        if entry.get("id")
    }
    projects = _normalize_projects(parsed.get("projects", []))
    note, truncated = _enforce_note_limit(parsed.get("referral_note", ""))
    if truncated:
        logger.info("Referral note for %s exceeded 300 chars and was trimmed", company)

    return ApplicationPackage(
        job_analysis=parsed.get("job_analysis", {}) or {},
        summary=parsed.get("summary", ""),
        skills=parsed.get("skills", []),
        experience_bullets=experience_bullets,
        projects=projects,
        cover_letter=parsed.get("cover_letter", "").strip(),
        referral_note=note,
        referral_dm=parsed.get("referral_dm", "").strip()[:LINKEDIN_DM_LIMIT],
        note_was_truncated=truncated,
    )


# Backwards-compatible alias for the older function name.
tailor_resume = generate_application_package


_EXTRACT_PROMPT = (
    "You extract two fields from a pasted job posting. Return ONLY JSON: "
    '{"company": "<hiring company>", "title": "<the role/job title>"}. '
    "Use the exact company and title as written in the posting. If the company "
    'is not stated, use "Unknown". Do not add commentary.'
)


def extract_company_title(
    posting_text: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int | None = 200,
) -> tuple[str, str] | None:
    """
    Pull (company, title) out of a raw pasted posting with one cheap,
    no-thinking DeepSeek call. Returns None on any failure so the caller can
    fall back to a heuristic. Trims the posting so this stays a tiny request.
    """
    if not api_key or not posting_text.strip():
        return None
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _EXTRACT_PROMPT},
            {"role": "user", "content": posting_text.strip()[:4000]},
        ],
        "response_format": {"type": "json_object"},
    }
    payload.update(_thinking_params("disabled", 0.0))
    if max_tokens:
        payload["max_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers=headers, json=payload, timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        usage.record(data.get("usage"))
        parsed = json.loads(_strip_fences(data["choices"][0]["message"]["content"]))
    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Company/title extraction failed: %s", exc)
        return None
    company = str(parsed.get("company", "")).strip()
    title = str(parsed.get("title", "")).strip()
    if not title:
        return None
    return (company or "Unknown", title)


def stub_application_package(
    master_resume: dict[str, Any],
    job_title: str,
    company: str,
) -> ApplicationPackage:
    """
    A deterministic, offline stand-in for a real DeepSeek call -- used by
    `python -m src.main --dry-run` to smoke-test the whole pipeline with no API
    key and no cost.

    It draws ONLY on the master resume and the posting's own title/company, so
    it flows through merge -> render -> grounding untouched and produces real
    PDFs without inventing anything. It does NOT tailor -- skills are left empty
    so the merge keeps the master's grouped skills, and bullets are the master's
    verbatim. The cover letter is clearly marked as placeholder text.
    """
    experience_bullets = {
        entry.get("id"): entry.get("bullets", [])
        for entry in master_resume.get("experience", [])
        if entry.get("id")
    }
    projects = _normalize_projects(master_resume.get("projects", []))
    cover_letter = (
        f"I am writing to apply for the {job_title} role at {company}. "
        "This is placeholder text produced by --dry-run to smoke-test rendering; "
        "run without the flag to generate a tailored cover letter."
    )
    note = (
        f"Hi, I just applied for the {job_title} role at {company} and would "
        "love to connect. [dry-run placeholder]"
    )
    dm = (
        f"Thanks for connecting. I recently applied for the {job_title} role at "
        f"{company} and wondered whether you might be open to referring me. "
        "[dry-run placeholder]"
    )
    return ApplicationPackage(
        job_analysis={"gaps": []},
        summary=master_resume.get("summary", "") or "",
        skills=[],  # empty -> merge preserves the master's grouped skills
        experience_bullets=experience_bullets,
        projects=projects,
        cover_letter=cover_letter,
        referral_note=note,
        referral_dm=dm,
    )
