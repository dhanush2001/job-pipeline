"""Tests for the judge agent -- stubs the API so it runs offline."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.judge as judge
import src.llm_client as llm

_RESPONSE = json.dumps({
    "red_flags": ["Bullet lists duties, not outcomes", "Passive voice in bullet 2"],
    "keywords_added": ["Docker", "FastAPI"],
    "summary": "Backend-leaning CS grad who ships FastAPI services on Docker, with REST APIs used daily in production.",
    "experience": [
        {"id": "acme-intern-2022",
         "bullets": ["Shipped an admin dashboard used by a 12-person team, by building a React + Java Servlet app on REST APIs",
                     "Cut manual data entry by owning end-to-end UI-to-database flow via REST APIs"]}],
})


class _FakeResp:
    """A streamed (SSE) chat response, content split mid-JSON across chunks."""
    status_code = 200
    headers: dict = {}
    def __init__(self, content): self._c = content
    def iter_lines(self, decode_unicode=True):
        half = len(self._c) // 2
        for piece in (self._c[:half], self._c[half:]):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]})
        yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        yield "data: [DONE]"
    def close(self): pass


def test_judge_parses_and_rewrites():
    calls = {"n": 0, "body": None}
    def fake_post(*a, **k):
        calls["n"] += 1
        calls["body"] = json.loads(k["json"]["messages"][1]["content"])
        return _FakeResp(_RESPONSE)
    orig = llm.requests.post
    llm.requests.post = fake_post
    try:
        out = judge.judge_and_rewrite(
            master_resume={"name": "X"}, github_repos=[], bullet_bank="",
            job_title="SWE", company="Acme", job_description="Need Docker, FastAPI.",
            experience_bullets={"acme-intern-2022": ["did stuff on a dashboard"]},
            missing_keywords=["docker", "fastapi"], api_key="k", model="deepseek-flash",
            current_summary="CS grad with backend experience.")
    finally:
        llm.requests.post = orig

    assert calls["n"] == 1
    # The current summary must be sent so the judge can rewrite it.
    assert calls["body"]["current_summary"] == "CS grad with backend experience."
    assert out["red_flags"] and "duties" in out["red_flags"][0].lower()
    assert out["keywords_added"] == ["Docker", "FastAPI"]
    assert out["summary"].startswith("Backend-leaning CS grad")
    assert "acme-intern-2022" in out["experience"]
    assert len(out["experience"]["acme-intern-2022"]) == 2
    print("test_judge_parses_and_rewrites: PASS")


def test_judge_summary_empty_when_absent():
    """No summary key in the response -> out['summary'] is '' (not None)."""
    resp = json.dumps({"red_flags": [], "keywords_added": [],
                       "experience": [{"id": "e1", "bullets": ["Did a thing"]}]})
    orig = llm.requests.post
    llm.requests.post = lambda *a, **k: _FakeResp(resp)
    try:
        out = judge.judge_and_rewrite(
            master_resume={}, github_repos=[], bullet_bank="", job_title="t",
            company="c", job_description="jd", experience_bullets={"e1": ["b"]},
            missing_keywords=[], api_key="k", model="m")
    finally:
        llm.requests.post = orig
    assert out["summary"] == ""
    print("test_judge_summary_empty_when_absent: PASS")


def test_judge_null_experience_does_not_raise():
    """`"experience": null` used to raise TypeError in the judge too."""
    resp = json.dumps({"red_flags": None, "keywords_added": None, "experience": None})
    orig = llm.requests.post
    llm.requests.post = lambda *a, **k: _FakeResp(resp)
    try:
        out = judge.judge_and_rewrite(
            master_resume={}, github_repos=[], bullet_bank="", job_title="t",
            company="c", job_description="jd", experience_bullets={"e1": ["b"]},
            missing_keywords=[], api_key="k", model="m")
    finally:
        llm.requests.post = orig
    assert out == {"red_flags": [], "keywords_added": [], "summary": "", "experience": {}}, out
    print("test_judge_null_experience_does_not_raise: PASS")


def test_judge_skips_without_key_or_bullets():
    assert judge.judge_and_rewrite({}, [], "", "t", "c", "jd", {"id": ["b"]}, [], api_key="", model="m") is None
    assert judge.judge_and_rewrite({}, [], "", "t", "c", "jd", {}, [], api_key="k", model="m") is None
    print("test_judge_skips_without_key_or_bullets: PASS")


if __name__ == "__main__":
    test_judge_parses_and_rewrites()
    test_judge_summary_empty_when_absent()
    test_judge_null_experience_does_not_raise()
    test_judge_skips_without_key_or_bullets()
    print("\nAll judge tests passed.")
