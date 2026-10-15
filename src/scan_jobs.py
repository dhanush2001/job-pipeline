"""
scan_jobs.py
------------
Pulls job postings from two kinds of sources:

1. ATS public JSON APIs (Greenhouse, Lever, Ashby, SmartRecruiters, Workable) —
   no auth needed, stable, ToS-friendly. These are the endpoints the companies'
   own career pages call, so hitting them directly is the same traffic a browser
   would generate, just structured. Each is a separate `sources.<ats>` list of
   company slugs in config.yaml; leave a list empty to skip that platform.

   NOTE: not every ATS exposes descriptions in its list endpoint. SmartRecruiters
   requires a per-posting detail fetch (N+1 requests) to get the text that
   matching and tailoring need; the others return it inline.

2. Adzuna's job search API as the "general board" source. Adzuna aggregates
   postings from LinkedIn/Indeed/company sites and re-publishes them under its
   own API with a free tier + API key. This is used INSTEAD of scraping
   LinkedIn/Indeed directly: both explicitly prohibit automated scraping in
   their Terms of Service and run aggressive bot detection, so a direct
   scraper would be fragile (breaks constantly) and risks the account tied to
   it. Adzuna gives comparable coverage without that risk.

Every fetch_* function returns a list of Job objects with a normalized schema
so downstream code (match.py, main.py) doesn't care which source a posting
came from.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Optional

from src.locations import ADZUNA_COUNTRY_NAME

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15  # seconds

# Identify the client honestly rather than sending python-requests' default UA
# or impersonating a browser. These are public APIs where a real contact point
# is the courteous thing to send — set PIPELINE_CONTACT to your email.
CONTACT = os.environ.get("PIPELINE_CONTACT", "")
USER_AGENT = (
    f"job-application-pipeline/1.0 (personal job search; {CONTACT})"
    if CONTACT
    else "job-application-pipeline/1.0 (personal job search)"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

# Small pause between API calls so a long company list doesn't arrive as a burst.
POLITE_DELAY_SECONDS = 1.0

# Adzuna's free tier throttles bursts with transient 503/429s; retry a few times
# with exponential backoff before giving up on a page.
ADZUNA_MAX_RETRIES = 3


@dataclasses.dataclass
class Job:
    id: str  # stable unique id, namespaced by source so ids never collide
    source: str  # greenhouse | lever | ashby | smartrecruiters | workable | adzuna
    company: str
    title: str
    location: str
    url: str  # link a human would open
    apply_url: str  # link to open to apply (shown in the apply index)
    description: str  # plain-text or HTML job description, used for tailoring + matching
    country: str = ""  # canonical country name; set from the Adzuna query, else inferred from location


def fetch_greenhouse_jobs(company_slug: str) -> list[Job]:
    """
    Greenhouse's public board API. company_slug is the token in a company's
    Greenhouse URL, e.g. for boards.greenhouse.io/stripe it's "stripe".
    Docs: https://developers.greenhouse.io/job-board.html
    """
    url = f"https://boards-api.greenhouse.io/v1/boards/{company_slug}/jobs?content=true"
    jobs: list[Job] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("Greenhouse fetch failed for %s: %s", company_slug, exc)
        return jobs

    for item in payload.get("jobs", []):
        jobs.append(
            Job(
                id=f"greenhouse:{company_slug}:{item['id']}",
                source="greenhouse",
                company=company_slug,
                title=item.get("title", ""),
                location=(item.get("location") or {}).get("name", ""),
                url=item.get("absolute_url", ""),
                apply_url=item.get("absolute_url", ""),
                description=item.get("content", ""),
            )
        )
    return jobs


def fetch_lever_jobs(company_slug: str) -> list[Job]:
    """
    Lever's public postings API. company_slug is the token in a company's
    Lever URL, e.g. for jobs.lever.co/netflix it's "netflix".
    Docs: https://github.com/lever/postings-api
    """
    url = f"https://api.lever.co/v0/postings/{company_slug}?mode=json"
    jobs: list[Job] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("Lever fetch failed for %s: %s", company_slug, exc)
        return jobs

    for item in payload:
        categories = item.get("categories", {})
        jobs.append(
            Job(
                id=f"lever:{company_slug}:{item['id']}",
                source="lever",
                company=company_slug,
                title=item.get("text", ""),
                location=categories.get("location", ""),
                url=item.get("hostedUrl", ""),
                apply_url=item.get("applyUrl", item.get("hostedUrl", "")),
                description=item.get("descriptionPlain", item.get("description", "")),
            )
        )
    return jobs


def fetch_ashby_jobs(board_name: str) -> list[Job]:
    """
    Ashby's public job-board posting API. board_name is the token in a company's
    Ashby URL, e.g. for jobs.ashbyhq.com/ramp it's "ramp". Descriptions come
    back inline (descriptionPlain), so no per-posting fetch is needed.
    Docs: https://developers.ashbyhq.com/docs/public-job-posting-api
    """
    url = f"https://api.ashbyhq.com/posting-api/job-board/{board_name}"
    jobs: list[Job] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("Ashby fetch failed for %s: %s", board_name, exc)
        return jobs

    for item in payload.get("jobs", []):
        job_url = item.get("jobUrl", "")
        jobs.append(
            Job(
                id=f"ashby:{board_name}:{item.get('id', '')}",
                source="ashby",
                company=board_name,
                title=item.get("title", ""),
                location=item.get("location") or "",
                url=job_url,
                apply_url=item.get("applyUrl", job_url),
                description=item.get("descriptionPlain") or item.get("descriptionHtml", ""),
            )
        )
    return jobs


def fetch_workable_jobs(account_slug: str) -> list[Job]:
    """
    Workable's public account widget API. account_slug is the token in a
    company's Workable URL, e.g. for apply.workable.com/acme it's "acme".
    `details=true` pulls the job description inline (otherwise only titles).
    """
    url = f"https://apply.workable.com/api/v1/widget/accounts/{account_slug}"
    jobs: list[Job] = []
    try:
        resp = requests.get(url, headers=HEADERS, params={"details": "true"}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("Workable fetch failed for %s: %s", account_slug, exc)
        return jobs

    for item in payload.get("jobs", []):
        loc = item.get("location") or {}
        location = loc.get("location_str") or ", ".join(
            p for p in (loc.get("city"), loc.get("region"), loc.get("country")) if p
        )
        shortcode = item.get("shortcode") or item.get("id", "")
        description = " ".join(
            p for p in (item.get("description", ""), item.get("requirements", "")) if p
        )
        job_url = item.get("url") or item.get("shortlink", "")
        jobs.append(
            Job(
                id=f"workable:{account_slug}:{shortcode}",
                source="workable",
                company=account_slug,
                title=item.get("title", ""),
                location=location,
                url=job_url,
                apply_url=item.get("application_url", job_url),
                description=description,
            )
        )
    return jobs


def _smartrecruiters_description(company_id: str, posting_id: str) -> str:
    """Fetch one posting's detail and join its text sections. Best-effort."""
    url = f"https://api.smartrecruiters.com/v1/companies/{company_id}/postings/{posting_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        sections = (resp.json().get("jobAd") or {}).get("sections") or {}
    except (requests.RequestException, AttributeError, ValueError) as exc:
        logger.debug("SmartRecruiters detail fetch failed for %s/%s: %s", company_id, posting_id, exc)
        return ""
    parts = []
    for key in ("jobDescription", "qualifications", "additionalInformation"):
        text = (sections.get(key) or {}).get("text", "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def fetch_smartrecruiters_jobs(company_id: str, max_postings: int = 100) -> list[Job]:
    """
    SmartRecruiters' public postings API. company_id is the identifier in a
    company's SmartRecruiters URL, e.g. for jobs.smartrecruiters.com/Square it's
    "Square". The list endpoint carries no description, so this fetches each
    posting's detail (N+1 requests) for the text matching/tailoring need --
    capped at max_postings to bound the request count on huge boards.
    Docs: https://dev.smartrecruiters.com/customer-api/posting-api/
    """
    base = f"https://api.smartrecruiters.com/v1/companies/{company_id}/postings"
    jobs: list[Job] = []
    summaries: list[dict] = []
    offset, page_size = 0, 100
    while len(summaries) < max_postings:
        try:
            resp = requests.get(
                base, headers=HEADERS,
                params={"limit": page_size, "offset": offset}, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            logger.warning("SmartRecruiters fetch failed for %s: %s", company_id, exc)
            break
        content = payload.get("content", [])
        if not content:
            break
        summaries.extend(content)
        offset += page_size
        if offset >= payload.get("totalFound", 0):
            break

    for item in summaries[:max_postings]:
        posting_id = item.get("id", "")
        loc = item.get("location") or {}
        location = ", ".join(
            p for p in (loc.get("city"), loc.get("region"), loc.get("country")) if p
        ) or ("Remote" if loc.get("remote") else "")
        page = f"https://jobs.smartrecruiters.com/{company_id}/{posting_id}"
        jobs.append(
            Job(
                id=f"smartrecruiters:{company_id}:{posting_id}",
                source="smartrecruiters",
                company=company_id,
                title=item.get("name", ""),
                location=location,
                url=page,
                apply_url=page,
                description=_smartrecruiters_description(company_id, posting_id) or item.get("name", ""),
            )
        )
    return jobs


def fetch_adzuna_jobs(
    keywords: str,
    location: str,
    app_id: str,
    app_key: str,
    country: str = "us",
    results_per_page: int = 20,
    max_pages: int = 2,
) -> list[Job]:
    """
    Adzuna aggregator API — used as the general-board source in place of
    scraping LinkedIn/Indeed directly. Free tier requires an app_id/app_key
    from https://developer.adzuna.com/.
    """
    jobs: list[Job] = []
    if not app_id or not app_key:
        logger.warning("Adzuna skipped: no app_id/app_key configured")
        return jobs

    for page in range(1, max_pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": keywords,
            "where": location,
            "results_per_page": results_per_page,
            "content-type": "application/json",
        }
        # Retry transient failures (503/429/5xx and network errors) with backoff
        # -- a scan fires many Adzuna requests in a burst, and the free tier
        # throttles with 503s that clear on a short wait. An auth error (401/403)
        # is NOT transient: stop the whole source and tell the user to check keys.
        payload = None
        for attempt in range(ADZUNA_MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                if resp.status_code in (401, 403):
                    logger.error("Adzuna auth failed (%s) -- check ADZUNA_APP_ID/APP_KEY.",
                                 resp.status_code)
                    return jobs
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{resp.status_code} {resp.reason}")
                resp.raise_for_status()
                payload = resp.json()
                break
            except requests.RequestException as exc:
                if attempt < ADZUNA_MAX_RETRIES:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning("Adzuna %s/%r page %s failed (%s) -- retry %d/%d in %ds",
                                   country, keywords, page, exc, attempt + 1, ADZUNA_MAX_RETRIES, wait)
                    time.sleep(wait)
                else:
                    logger.warning("Adzuna %s/%r page %s gave up after %d retries: %s",
                                   country, keywords, page, ADZUNA_MAX_RETRIES, exc)
        if payload is None:
            break

        time.sleep(POLITE_DELAY_SECONDS)

        results = payload.get("results", [])
        if not results:
            break

        country_name = ADZUNA_COUNTRY_NAME.get(country, country.upper())
        for item in results:
            jobs.append(
                Job(
                    id=f"adzuna:{item['id']}",
                    source="adzuna",
                    company=(item.get("company") or {}).get("display_name", ""),
                    title=item.get("title", ""),
                    location=(item.get("location") or {}).get("display_name", ""),
                    url=item.get("redirect_url", ""),
                    apply_url=item.get("redirect_url", ""),
                    description=item.get("description", ""),
                    country=country_name,  # authoritative: the API was queried per-country
                )
            )
    return jobs


def scan_all(config: dict) -> list[Job]:
    """Run every configured source and return the combined, deduped job list."""
    jobs: list[Job] = []

    for slug in config.get("sources", {}).get("greenhouse", []):
        jobs.extend(fetch_greenhouse_jobs(slug))
        time.sleep(POLITE_DELAY_SECONDS)

    for slug in config.get("sources", {}).get("lever", []):
        jobs.extend(fetch_lever_jobs(slug))
        time.sleep(POLITE_DELAY_SECONDS)

    for slug in config.get("sources", {}).get("ashby", []):
        jobs.extend(fetch_ashby_jobs(slug))
        time.sleep(POLITE_DELAY_SECONDS)

    for company_id in config.get("sources", {}).get("smartrecruiters", []):
        jobs.extend(fetch_smartrecruiters_jobs(company_id))
        time.sleep(POLITE_DELAY_SECONDS)

    for slug in config.get("sources", {}).get("workable", []):
        jobs.extend(fetch_workable_jobs(slug))
        time.sleep(POLITE_DELAY_SECONDS)

    adzuna_cfg = config.get("sources", {}).get("adzuna")
    if adzuna_cfg:
        app_id = os.environ.get("ADZUNA_APP_ID", adzuna_cfg.get("app_id", ""))
        app_key = os.environ.get("ADZUNA_APP_KEY", adzuna_cfg.get("app_key", ""))
        location = adzuna_cfg.get("location", "")
        country = adzuna_cfg.get("country", "us")
        max_pages = adzuna_cfg.get("max_pages", 2)
        # Accept either a single `keywords` string or a list of `queries` -- each
        # query is a separate aggregator search, so a list casts the widest net
        # across role families (this is the closest thing to "all jobs posted").
        queries = adzuna_cfg.get("queries") or (
            [adzuna_cfg["keywords"]] if adzuna_cfg.get("keywords") else []
        )
        # Run every query against every targeted country (same free keys work
        # across all Adzuna countries). `countries` is a list of country codes;
        # falls back to the single `country` for backward compatibility.
        countries = adzuna_cfg.get("countries") or [country]
        for cc in countries:
            for kw in queries:
                jobs.extend(fetch_adzuna_jobs(
                    keywords=kw, location=location, app_id=app_id, app_key=app_key,
                    country=cc, max_pages=max_pages,
                ))
                time.sleep(POLITE_DELAY_SECONDS)

    seen: set[str] = set()
    deduped: list[Job] = []
    for job in jobs:
        if job.id in seen:
            continue
        seen.add(job.id)
        deduped.append(job)

    logger.info("Scanned %d jobs across all sources (%d unique)", len(jobs), len(deduped))
    return deduped
