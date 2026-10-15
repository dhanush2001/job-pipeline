"""
github_projects.py
-------------------
Pulls your actual repositories from the GitHub API so the tailoring step can
pick the projects most relevant to each role, instead of shipping the same
static Projects section to every employer.

Auth: works unauthenticated (60 requests/hour, public repos only). Set
GITHUB_TOKEN for 5,000 requests/hour and access to your private repos. A
read-only fine-grained token with just "public_repo" (or "repo" if you want
private ones considered) is plenty -- do not use a broad-scope token.

What gets sent to the LLM: name, description, language, topics, star count,
last-push date, and a truncated README excerpt. READMEs are capped hard
because they're the one field that can be enormous, and input tokens are
what you pay for.
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT = 15
README_CHAR_LIMIT = 1200  # keeps prompt cost bounded; see cost notes in README


@dataclasses.dataclass
class Repo:
    name: str
    description: str
    language: str
    topics: list[str]
    stars: int
    pushed_at: str
    url: str
    readme_excerpt: str = ""

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "topics": self.topics,
            "stars": self.stars,
            "last_pushed": self.pushed_at,
            "url": self.url,
            "readme_excerpt": self.readme_excerpt,
        }


def _check_rate_limit(resp: requests.Response) -> None:
    """
    Turn GitHub's 403-on-rate-limit into a clear, actionable log line.

    This matters more than it looks: the unauthenticated limit is 60
    requests/hour PER IP, and this module spends 1 request on the repo list
    plus 1 per README. On a GitHub Actions runner the IP is shared Azure
    infrastructure, so unauthenticated calls hit this limit routinely --
    verified the hard way during development, on a shared sandbox IP that
    was already exhausted by unrelated traffic. Always set GITHUB_TOKEN in
    CI; the workflow passes the automatic secrets.GITHUB_TOKEN for this.
    """
    remaining = resp.headers.get("x-ratelimit-remaining")
    if resp.status_code == 403 and remaining == "0":
        limit = resp.headers.get("x-ratelimit-limit", "?")
        reset = resp.headers.get("x-ratelimit-reset", "")
        logger.error(
            "GitHub API rate limit exhausted (limit=%s/hour, resets at epoch %s). "
            "Set GITHUB_TOKEN to raise this from 60 to 5,000 requests/hour.",
            limit, reset,
        )


def _headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "job-application-pipeline/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_readme_excerpt(owner: str, repo_name: str, token: str | None) -> str:
    """Best-effort README fetch. Missing READMEs are normal, not an error."""
    url = f"{GITHUB_API}/repos/{owner}/{repo_name}/readme"
    try:
        resp = requests.get(url, headers=_headers(token), timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return ""
        _check_rate_limit(resp)
        resp.raise_for_status()
        content = resp.json().get("content", "")
        decoded = base64.b64decode(content).decode("utf-8", errors="replace")
    except (requests.RequestException, ValueError) as exc:
        logger.debug("README fetch failed for %s/%s: %s", owner, repo_name, exc)
        return ""

    # Strip badge/image lines -- they're noise that costs input tokens.
    lines = [ln for ln in decoded.splitlines() if not ln.strip().startswith("![")]
    return "\n".join(lines)[:README_CHAR_LIMIT].strip()


def fetch_repos(
    username: str,
    token: str | None = None,
    max_repos: int = 12,
    include_forks: bool = False,
    include_readmes: bool = True,
) -> list[Repo]:
    """
    Fetch a user's repos, newest-pushed first. Forks are excluded by default
    since they're generally not your work and dilute the signal.
    """
    if not username:
        logger.warning("No GitHub username configured -- skipping repo fetch")
        return []

    token = token or os.environ.get("GITHUB_TOKEN")
    url = f"{GITHUB_API}/users/{username}/repos"
    params = {"sort": "pushed", "direction": "desc", "per_page": 100, "type": "owner"}

    try:
        resp = requests.get(url, headers=_headers(token), params=params, timeout=REQUEST_TIMEOUT)
        _check_rate_limit(resp)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("GitHub repo fetch failed for %s: %s", username, exc)
        return []

    repos: list[Repo] = []
    for item in payload:
        if item.get("fork") and not include_forks:
            continue
        if item.get("archived"):
            continue
        repos.append(
            Repo(
                name=item.get("name", ""),
                description=item.get("description") or "",
                language=item.get("language") or "",
                topics=item.get("topics", []) or [],
                stars=item.get("stargazers_count", 0),
                pushed_at=(item.get("pushed_at") or "")[:10],
                url=item.get("html_url", ""),
            )
        )
        if len(repos) >= max_repos:
            break

    if include_readmes:
        for repo in repos:
            repo.readme_excerpt = fetch_readme_excerpt(username, repo.name, token)

    logger.info("Fetched %d GitHub repos for %s", len(repos), username)
    return repos
