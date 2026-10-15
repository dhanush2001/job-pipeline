"""
Tests for the local on-disk response cache (deepseek.cache).

The cache is opt-in and keyed on the exact request, so a re-run with identical
inputs must reuse the saved response and make NO second API call. These tests
stub requests.post with a call counter -- no API key or network needed.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.tailor_resume as tr

# A minimal but valid package payload the parser accepts.
_CONTENT = json.dumps({
    "summary": "Tailored summary.",
    "skills": ["Python"],
    "experience": [],
    "projects": [{"name": "P", "bullets": ["b"]}],
    "cover_letter": "cover",
    "referral_note": "note",
    "referral_dm": "dm",
})


class _FakeResp:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _counting_post():
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return _FakeResp(_CONTENT)

    return calls, fake_post


_BASE = dict(
    master_resume={"name": "X"},
    job_title="SWE",
    company="Acme",
    job_description="job desc",
    api_key="test-key",
)


def test_cache_reuses_response_without_second_call():
    calls, fake_post = _counting_post()
    original = tr.requests.post
    tr.requests.post = fake_post
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            kwargs = {**_BASE, "cache_enabled": True, "cache_dir": cache_dir}
            first = tr.generate_application_package(**kwargs)
            second = tr.generate_application_package(**kwargs)

            assert calls["n"] == 1, f"Expected 1 API call, made {calls['n']}"
            assert first.summary == second.summary == "Tailored summary."
            assert len(list(Path(cache_dir).glob("*.json"))) == 1, "One cache file expected"
    finally:
        tr.requests.post = original
    print("test_cache_reuses_response_without_second_call: PASS")


def test_cache_disabled_calls_every_time():
    calls, fake_post = _counting_post()
    original = tr.requests.post
    tr.requests.post = fake_post
    try:
        tr.generate_application_package(**_BASE)  # cache off by default
        tr.generate_application_package(**_BASE)
        assert calls["n"] == 2, f"Cache is off, expected 2 API calls, made {calls['n']}"
    finally:
        tr.requests.post = original
    print("test_cache_disabled_calls_every_time: PASS")


def test_different_job_misses_cache():
    calls, fake_post = _counting_post()
    original = tr.requests.post
    tr.requests.post = fake_post
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            common = {"cache_enabled": True, "cache_dir": cache_dir}
            tr.generate_application_package(**{**_BASE, **common})
            # A different job description must key differently -> a real call.
            tr.generate_application_package(
                **{**_BASE, "job_description": "a totally different posting", **common}
            )
            assert calls["n"] == 2, f"Different input should miss; made {calls['n']} calls"
    finally:
        tr.requests.post = original
    print("test_different_job_misses_cache: PASS")


if __name__ == "__main__":
    test_cache_reuses_response_without_second_call()
    test_cache_disabled_calls_every_time()
    test_different_job_misses_cache()
    print("\nAll response-cache tests passed.")
