"""
Tests for the Ashby / SmartRecruiters / Workable fetchers.

These stub scan_jobs.requests.get with a tiny URL router that serves saved
fixtures, so the REAL fetch_* functions run offline -- including
SmartRecruiters' N+1 per-posting detail fetch. No network, no keys.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.scan_jobs as sj

FIX = Path(__file__).parent / "fixtures"


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise sj.requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def _install_router(routes):
    """routes: list of (url_predicate, fixture_filename). First match wins."""
    def fake_get(url, headers=None, params=None, timeout=None):
        for predicate, fixture in routes:
            if predicate(url):
                return _FakeResp(json.loads((FIX / fixture).read_text()))
        return _FakeResp({}, status=404)

    original = sj.requests.get
    sj.requests.get = fake_get
    return original


def test_ashby_parsing():
    original = _install_router([(lambda u: "ashby" in u, "ashby_response.json")])
    try:
        jobs = sj.fetch_ashby_jobs("acme")
    finally:
        sj.requests.get = original

    assert len(jobs) == 2
    j = jobs[0]
    assert j.source == "ashby"
    assert j.id == "ashby:acme:a1"
    assert j.title == "Software Engineer, New Grad"
    assert j.location == "Remote - US"
    assert j.apply_url.endswith("/application")
    assert "distributed systems" in j.description
    print("test_ashby_parsing: PASS")


def test_workable_parsing():
    original = _install_router([(lambda u: "workable" in u, "workable_response.json")])
    try:
        jobs = sj.fetch_workable_jobs("acme")
    finally:
        sj.requests.get = original

    assert len(jobs) == 1
    j = jobs[0]
    assert j.source == "workable"
    assert j.id == "workable:acme:W1"
    assert j.location == "Austin, TX, United States"
    assert j.apply_url.endswith("/apply")
    # description = job description + requirements joined
    assert "distributed systems" in j.description and "Computer Science" in j.description
    print("test_workable_parsing: PASS")


def test_smartrecruiters_parsing_with_detail_fetch():
    # List endpoint ends in /postings; detail endpoint is /postings/<id>.
    routes = [
        (lambda u: u.endswith("/postings"), "smartrecruiters_postings.json"),
        (lambda u: "/postings/" in u, "smartrecruiters_detail.json"),
    ]
    original = _install_router(routes)
    try:
        jobs = sj.fetch_smartrecruiters_jobs("Acme")
    finally:
        sj.requests.get = original

    assert len(jobs) == 1
    j = jobs[0]
    assert j.source == "smartrecruiters"
    assert j.id == "smartrecruiters:Acme:SR1"
    assert j.title == "Software Engineer New Grad"
    assert j.location == "Seattle, WA, us"
    assert j.url == "https://jobs.smartrecruiters.com/Acme/SR1"
    # description came from the N+1 detail fetch, not the list summary
    assert "Python" in j.description and "computer science degree" in j.description
    print("test_smartrecruiters_parsing_with_detail_fetch: PASS")


def test_scan_all_wires_new_sources():
    routes = [
        (lambda u: "ashby" in u, "ashby_response.json"),
        (lambda u: "workable" in u, "workable_response.json"),
        (lambda u: u.endswith("/postings"), "smartrecruiters_postings.json"),
        (lambda u: "/postings/" in u, "smartrecruiters_detail.json"),
    ]
    original_get = _install_router(routes)
    original_sleep = sj.time.sleep
    sj.time.sleep = lambda *_: None  # don't wait POLITE_DELAY per slug
    try:
        config = {"sources": {"ashby": ["acme"], "smartrecruiters": ["Acme"], "workable": ["acme"]}}
        jobs = sj.scan_all(config)
    finally:
        sj.requests.get = original_get
        sj.time.sleep = original_sleep

    sources = {j.source for j in jobs}
    assert sources == {"ashby", "smartrecruiters", "workable"}, sources
    # 2 ashby + 1 smartrecruiters + 1 workable
    assert len(jobs) == 4, len(jobs)
    print("test_scan_all_wires_new_sources: PASS")


if __name__ == "__main__":
    test_ashby_parsing()
    test_workable_parsing()
    test_smartrecruiters_parsing_with_detail_fetch()
    test_scan_all_wires_new_sources()
    print("\nAll new-source tests passed.")
