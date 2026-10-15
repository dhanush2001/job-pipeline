"""
judge.py -- a second-pass "judge" agent that reviews and rewrites the tailored
resume the way a senior recruiter and then a hiring manager would.

One DeepSeek call, two lenses:
  PASS 1 (senior technical recruiter): find the red flags a hiring manager would
          hold against the resume, and the keywords the posting wants that the
          candidate can legitimately add.
  PASS 2 (hiring manager scanning 200 resumes): rewrite the EXPERIENCE bullets
          in Google XYZ form -- "Accomplished [X] as measured by [Y], by doing
          [Z]" -- and the SUMMARY, weaving in TRUE keywords, cutting the red
          flags, front-loaded so the first few words stop the scroll. Skills,
          projects, and education are intentionally left untouched.

Grounding still runs afterward: the judge is told to add only skills/metrics the
candidate's evidence supports, and any invented number is flagged downstream.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

from src import usage
from src.tailor_resume import (
    DEEPSEEK_API_BASE,
    REQUEST_TIMEOUT,
    _strip_fences,
    _thinking_params,
)

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are two experts reviewing one candidate's resume
against one job posting. You receive the candidate's EVIDENCE (master resume,
GitHub repos, bullet bank), the resume's CURRENT experience bullets, the target
job, and the keywords the posting wants that are currently missing.

## PASS 1 — senior technical recruiter
Identify:
- red_flags: what a hiring manager would notice and hold against this resume --
  vague duty-listing with no outcome, passive voice, buzzword stuffing, clichés
  ("team player", "hard worker"), inconsistent tense, missing scope/impact,
  over-claiming, or bullets that don't say what actually happened.
- keywords the posting wants that are MISSING but the candidate can legitimately
  claim from their evidence. Do NOT list a keyword the evidence doesn't support.

## PASS 2 — hiring manager scanning 200 resumes, rewrite the EXPERIENCE bullets
Rewrite every experience bullet using the Google XYZ formula:
"Accomplished [X] as measured by [Y], by doing [Z]." Rules:
- Lead with a strong past-tense verb and the OUTCOME. The first 4-5 words must
  stop a recruiter mid-scroll -- put the result up front, not the task.
- Include [Y] (a number/metric) ONLY when a real one exists in the candidate's
  evidence. NEVER invent a metric. No real number? Drop Y -- "Accomplished X by
  doing Z" is honest and still strong. A fabricated number loses the offer.
- The `missing_keywords` you are given are ALREADY verified against the
  candidate's evidence -- weave in the ones that fit naturally. Do NOT introduce
  any other tool, framework, cloud, or skill that isn't already in the
  candidate's evidence. If the posting wants something the candidate lacks, that
  is a gap -- leave it out; do not paper over it.
- Cut every red flag: no passive voice, no duty-listing, no filler.
- One line per bullet, ~15-24 words. Keep the same bullet ids. 3 bullets max.
- Wrap the 2-4 most important technical terms per bullet in **markdown bold**
  (specific tools/systems/metrics, never whole clauses or verbs; ~4 spans max).
  This emphasis is rendered in the PDF, so the rewritten bullets must keep it.

### Also rewrite the SUMMARY
Rewrite `current_summary` to the same stop-the-scroll standard: AT MOST 2
sentences (~2 resume lines), leading with the single strongest qualification for
THIS role and naturally carrying the posting's language. Same grounding and
keyword rules as the bullets. No bold in the summary. If `current_summary` is
empty, return "" -- do NOT invent one.

### Sections you MUST leave untouched
Rewrite ONLY the experience bullets and the summary. Do NOT return or alter
skills, projects, or education. A hiring manager skims those sections, so they
are intentionally skipped here; touching them only risks breaking grounding.

## GROUNDING (non-negotiable)
Every fact must trace to the candidate's evidence or the posting's own
vocabulary. Numbers are checked programmatically after you -- an untraceable
number is the fastest way to lose an offer.

Respond with ONLY this JSON, no markdown fences, no commentary:
{
  "red_flags": ["short, specific issues a hiring manager would flag"],
  "keywords_added": ["keywords you wove in (each supported by the evidence)"],
  "summary": "rewritten summary (<=2 sentences), or \"\" if there was none",
  "experience": [{"id": "<id from current_experience>", "bullets": ["XYZ bullet", "..."]}]
}"""


def judge_and_rewrite(
    master_resume: dict[str, Any],
    github_repos: list[dict[str, Any]] | None,
    bullet_bank: str,
    job_title: str,
    company: str,
    job_description: str,
    experience_bullets: dict[str, list[str]],
    missing_keywords: list[str],
    api_key: str,
    model: str,
    current_summary: str = "",
    reasoning: str = "disabled",
    temperature: float = 0.4,
    max_tokens: int | None = None,
    max_retries: int = 2,
) -> dict | None:
    """
    Returns {"red_flags": [...], "keywords_added": [...], "summary": str,
    "experience": {id: [bullets]}} or None on failure (caller keeps the
    original bullets/summary). "summary" is "" when there was none to rewrite.
    """
    if not api_key or not experience_bullets:
        return None

    user = {
        "candidate_evidence": {
            "master_resume": master_resume,
            "github_repos": github_repos or [],
            "bullet_bank": bullet_bank or "",
        },
        "current_summary": current_summary or "",
        "current_experience": [{"id": k, "bullets": v} for k, v in experience_bullets.items()],
        "target_job": {"title": job_title, "company": company, "description": job_description[:6000]},
        "missing_keywords": missing_keywords,
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user)},
        ],
        "response_format": {"type": "json_object"},
    }
    payload.update(_thinking_params(reasoning, temperature))
    if max_tokens:
        payload["max_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    raw = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers=headers, json=payload, timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                logger.warning("Judge rate limited (attempt %d) for %s", attempt + 1, job_title)
                continue
            resp.raise_for_status()
            data = resp.json()
            usage.record(data.get("usage"))
            raw = data["choices"][0]["message"]["content"]
            break
        except (requests.RequestException, KeyError, IndexError) as exc:
            logger.warning("Judge call failed (attempt %d) for %s @ %s: %s",
                           attempt + 1, job_title, company, exc)

    if raw is None:
        return None
    try:
        parsed = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        logger.error("Judge response unparseable: %s", exc)
        return None

    experience = {
        e.get("id"): e.get("bullets", [])
        for e in parsed.get("experience", [])
        if e.get("id") and e.get("bullets")
    }
    return {
        "red_flags": parsed.get("red_flags", []) or [],
        "keywords_added": parsed.get("keywords_added", []) or [],
        "summary": (parsed.get("summary") or "").strip(),
        "experience": experience,
    }
