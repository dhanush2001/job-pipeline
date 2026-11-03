"""
Tests for llm_client -- the streamed, deadline-bounded DeepSeek client. HTTP and
sleep are stubbed, so every failure mode runs offline and instantly: stalls,
rate limits with Retry-After, fatal errors (no retry), truncation, the hard
per-call deadline, and the diagnostics report.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

import src.llm_client as llm

PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class _Stream:
    """Fake streamed response. `chunks` are content pieces; `finish` the final reason."""
    def __init__(self, chunks=("{\"ok\": ", "true}"), finish="stop", status=200,
                 headers=None, body="", raise_after=None, usage=None):
        self.status_code, self.headers, self.text = status, headers or {}, body
        self._chunks, self._finish, self._raise_after, self._usage = chunks, finish, raise_after, usage

    def iter_lines(self, decode_unicode=True):
        for i, c in enumerate(self._chunks):
            if self._raise_after is not None and i == self._raise_after:
                raise requests.exceptions.ConnectionError("Read timed out. (read timeout=60)")
            yield "data: " + json.dumps({"choices": [{"delta": {"content": c}}]})
            yield ""  # SSE keep-alive blank line
        yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": self._finish}]})
        if self._usage:
            yield "data: " + json.dumps({"choices": [], "usage": self._usage})
        yield "data: [DONE]"

    def close(self):
        pass


def _run(responses, **settings):
    """Serve `responses` in order from a fake requests.post; record sleeps."""
    seq, sleeps = list(responses), []
    orig_post, orig_sleep, orig_settings = llm.requests.post, llm.time.sleep, llm.settings()
    llm.requests.post = lambda *a, **k: seq.pop(0) if not isinstance(seq[0], Exception) else (_ for _ in ()).throw(seq.pop(0))
    llm.time.sleep = sleeps.append
    llm.configure(settings)
    llm.reset()
    try:
        return llm.chat_json(PAYLOAD, "key", label="t"), sleeps
    finally:
        llm.requests.post, llm.time.sleep = orig_post, orig_sleep
        llm._SETTINGS.update(orig_settings)


def test_streamed_chunks_are_reassembled():
    res, _ = _run([_Stream(usage={"prompt_tokens": 7, "completion_tokens": 3})])
    assert res.ok and res.content == '{"ok": true}' and res.attempts == 1, res
    assert res.usage == {"prompt_tokens": 7, "completion_tokens": 3}
    print("test_streamed_chunks_are_reassembled: PASS")


def test_rate_limit_backs_off_honouring_retry_after_then_succeeds():
    res, sleeps = _run([_Stream(status=429, headers={"Retry-After": "3"}, body="slow down"), _Stream()])
    assert res.ok and res.attempts == 2, res
    assert sleeps == [3.0], f"should wait exactly Retry-After: {sleeps}"
    print("test_rate_limit_backs_off_honouring_retry_after_then_succeeds: PASS")


def test_stall_is_retried():
    res, sleeps = _run([_Stream(raise_after=1), _Stream()])
    assert res.ok and res.attempts == 2 and len(sleeps) == 1, res
    print("test_stall_is_retried: PASS")


def test_fatal_errors_fail_fast_without_retry():
    for status, reason in [(402, "no_balance"), (401, "auth"), (400, "bad_request")]:
        res, sleeps = _run([_Stream(status=status, body='{"error":{"message":"Insufficient Balance"}}')])
        assert res.reason == reason and res.attempts == 1 and sleeps == [], (status, res)
        assert "Insufficient Balance" in res.detail, "server's message must surface"
    print("test_fatal_errors_fail_fast_without_retry: PASS")


def test_truncation_is_reported_not_passed_on():
    res, _ = _run([_Stream(chunks=('{"summary": "half',), finish="length")])
    assert res.reason == "truncated" and not res.ok, res
    print("test_truncation_is_reported_not_passed_on: PASS")


def test_gives_up_after_max_retries():
    res, sleeps = _run([_Stream(status=503)] * 3, max_retries=2)
    assert res.reason == "http_503" and res.attempts == 3 and len(sleeps) == 2, res
    print("test_gives_up_after_max_retries: PASS")


def test_hard_deadline_stops_a_slow_stream():
    """A stream that keeps trickling (never stalls) is still cut at call_deadline."""
    clock = {"t": 0.0}
    orig = llm.time.monotonic
    def tick():
        clock["t"] += 50.0          # each check costs 50 "seconds"
        return clock["t"]
    llm.time.monotonic = tick
    try:
        res, _ = _run([_Stream(chunks=tuple("x" * 20))], call_deadline=200)
    finally:
        llm.time.monotonic = orig
    assert res.reason == "deadline" and not res.ok, res
    print("test_hard_deadline_stops_a_slow_stream: PASS")


class _KeepAliveOnly:
    """What DeepSeek does under overload: HTTP 200, then only keep-alives."""
    status_code, headers, text = 200, {}, ""
    def iter_lines(self, decode_unicode=True):
        for _ in range(100):
            yield ": keep-alive"
            yield ""
    def close(self):
        pass


def test_keepalive_only_is_queued_and_trips_breaker():
    clock = {"t": 0.0}
    orig = llm.time.monotonic
    def tick():
        clock["t"] += 12.0           # keep-alives arrive every ~12s
        return clock["t"]
    llm.time.monotonic = tick
    try:
        res, sleeps = _run([_KeepAliveOnly()], queue_timeout=90, call_deadline=300)
        assert res.reason == "queued" and res.attempts == 1 and sleeps == [], res
        assert "keep-alives only" in res.detail
        assert not llm.api_unavailable(), "one queued call must not trip the breaker"
        # Two more queued calls in a row -> breaker trips (threshold 3).
        orig_post = llm.requests.post
        llm.requests.post = lambda *a, **k: _KeepAliveOnly()
        try:
            llm.chat_json(PAYLOAD, "key", label="b")
            llm.chat_json(PAYLOAD, "key", label="c")
        finally:
            llm.requests.post = orig_post
        assert llm.api_unavailable()
    finally:
        llm.time.monotonic = orig
        llm.reset()
    assert not llm.api_unavailable(), "reset() clears the breaker"
    print("test_keepalive_only_is_queued_and_trips_breaker: PASS")


def test_no_key_and_diagnostics_report():
    llm.reset()
    assert llm.chat_json(PAYLOAD, "", label="x").reason == "no_key"
    _run([_Stream()])                       # resets, then one ok call
    llm._record(llm.CallResult(None, "stalled", "job B", 3, 120.0, detail="no data for 60s"))
    with tempfile.TemporaryDirectory() as d:
        line = llm.write_report(f"{d}/diag.md")
        text = Path(f"{d}/diag.md").read_text()
    assert "1/2 ok" in line and "stalled=1" in line, line
    assert "job B" in text and "no data for 60s" in text
    print("test_no_key_and_diagnostics_report: PASS")


if __name__ == "__main__":
    test_streamed_chunks_are_reassembled()
    test_rate_limit_backs_off_honouring_retry_after_then_succeeds()
    test_stall_is_retried()
    test_fatal_errors_fail_fast_without_retry()
    test_truncation_is_reported_not_passed_on()
    test_gives_up_after_max_retries()
    test_hard_deadline_stops_a_slow_stream()
    test_keepalive_only_is_queued_and_trips_breaker()
    test_no_key_and_diagnostics_report()
    print("\nAll llm_client tests passed.")
