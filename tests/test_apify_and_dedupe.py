"""
Apify source (actor dataset rows -> Jobs, with differing field names) and
cross-board de-duplication. Offline: requests.post is stubbed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.scan_jobs as sj
from src.scan_jobs import Job, apify_item_to_job, dedupe_postings


def _job(id, source, company, title, location="San Francisco, CA", apply_url="", desc="d"):
    return Job(id=id, source=source, company=company, title=title, location=location,
               url=apply_url or f"https://x/{id}", apply_url=apply_url or f"https://x/{id}",
               description=desc)


def test_apify_rows_map_across_actor_field_names():
    linkedin = {"id": "4012345678", "title": "Software Engineer, New Grad", "companyName": "Acme",
                "location": "New York, NY", "link": "https://www.linkedin.com/jobs/view/4012345678",
                "applyUrl": "https://boards.greenhouse.io/acme/jobs/7654321", "descriptionText": "Python"}
    indeed = {"positionName": "ML Engineer", "company": {"name": "Beta"}, "location": "Remote",
              "url": "https://www.indeed.com/viewjob?jk=abc", "description": "<p>PyTorch</p>"}
    a = apify_item_to_job(linkedin, "curious/linkedin-jobs")
    b = apify_item_to_job(indeed, "misc~indeed-scraper")
    assert a.id == "apify:linkedin-jobs:4012345678" and a.company == "Acme"
    assert a.apply_url.endswith("/jobs/7654321") and a.description == "Python"
    assert b.company == "Beta" and b.title == "ML Engineer" and b.id.startswith("apify:indeed-scraper:")
    assert apify_item_to_job({"title": "no link"}, "x/y") is None
    print("test_apify_rows_map_across_actor_field_names: PASS")


def test_fetch_apify_handles_auth_errors_and_missing_token():
    class R:
        def __init__(self, code, body):
            self.status_code, self._body = code, body
        def json(self):
            return self._body
        def raise_for_status(self):
            if self.status_code >= 400:
                raise sj.requests.HTTPError(str(self.status_code))
    orig = sj.requests.post
    try:
        assert sj.fetch_apify_jobs("a/b", {}, token="") == []
        sj.requests.post = lambda *a, **k: R(401, {})
        assert sj.fetch_apify_jobs("a/b", {}, token="t") == []
        sj.requests.post = lambda *a, **k: R(200, [{"title": "SWE", "url": "https://l/1", "id": "1"}])
        jobs = sj.fetch_apify_jobs("a/b", {"q": "swe"}, token="t")
        assert [j.id for j in jobs] == ["apify:b:1"]
    finally:
        sj.requests.post = orig
    print("test_fetch_apify_handles_auth_errors_and_missing_token: PASS")


def test_same_posting_on_two_boards_is_kept_once():
    gh = _job("greenhouse:7654321", "greenhouse", "Acme", "Software Engineer, New Grad",
              apply_url="https://boards.greenhouse.io/acme/jobs/7654321", desc="full description")
    # LinkedIn copy via Apify: different id, apply link points at the same Greenhouse req.
    li = _job("apify:linkedin:4012345678", "apify", "ACME", "Software Engineer - New Grad",
              location="New York, NY", apply_url="https://boards.greenhouse.io/acme/jobs/7654321")
    # Adzuna copy: different id and link, same company/title/city.
    az = _job("adzuna:99", "adzuna", "acme", "Software Engineer, New Grad", desc="short")
    kept = dedupe_postings([gh, li, az])
    assert kept == [gh], [j.id for j in kept]
    print("test_same_posting_on_two_boards_is_kept_once: PASS")


def test_aggregator_copies_keep_longer_description():
    a = _job("adzuna:1", "adzuna", "Beta", "ML Engineer", desc="short")
    b = _job("apify:li:2", "apify", "Beta", "ML Engineer", desc="a much longer description")
    kept = dedupe_postings([a, b])
    assert [j.id for j in kept] == ["apify:li:2"], [j.id for j in kept]
    print("test_aggregator_copies_keep_longer_description: PASS")


def test_distinct_reqs_are_not_merged():
    # Same company board, same title, different req ids -> two real openings.
    r1 = _job("greenhouse:1", "greenhouse", "Acme", "Software Engineer")
    r2 = _job("greenhouse:2", "greenhouse", "Acme", "Software Engineer")
    # Different city on an aggregator -> a different posting.
    other = _job("adzuna:3", "adzuna", "Acme", "Software Engineer", location="Austin, TX")
    kept = dedupe_postings([r1, r2, other])
    assert len(kept) == 3, [j.id for j in kept]
    print("test_distinct_reqs_are_not_merged: PASS")


if __name__ == "__main__":
    test_apify_rows_map_across_actor_field_names()
    test_fetch_apify_handles_auth_errors_and_missing_token()
    test_same_posting_on_two_boards_is_kept_once()
    test_aggregator_copies_keep_longer_description()
    test_distinct_reqs_are_not_merged()
    print("\nAll Apify / dedupe tests passed.")
