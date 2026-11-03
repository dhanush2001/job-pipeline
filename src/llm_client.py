"""
llm_client.py
-------------
One place that talks to the DeepSeek chat API (OpenAI-compatible), built so a
run can't silently stall and every failure says WHY.

Why streaming: a non-streaming request sends nothing until the whole answer is
generated, so a plain read timeout can't tell "still generating a long JSON"
from "the connection is dead". Streaming keeps bytes flowing while the model
works, so:

  * stall_timeout  -- fires only when NOTHING arrives for that long (a real stall)
  * queue_timeout  -- DeepSeek, when overloaded, answers HTTP 200 and then sends
                      only ": keep-alive" lines (for up to ~10 min). Keep-alives
                      defeat any byte-based timeout, so this bounds the wait for
                      the first REAL data and reports the call as "queued".
  * call_deadline  -- hard wall-clock budget per call, retries included
  * connect_timeout-- fail fast when the API is unreachable

Retries back off exponentially (honouring Retry-After) on 429 / 5xx / network /
stall. Errors a retry can't fix -- bad key (401), no balance (402), malformed
request (400/422) -- fail immediately with the server's own message. A response
cut off by max_tokens is reported as truncated, not passed on as broken JSON.
A circuit breaker stops a run from launching more work once DeepSeek keeps
queueing requests without starting them.

Every call is recorded (outcome, elapsed, attempts, tokens) so a run ends with
output/ai_diagnostics.md instead of a silent gap.
`python -m src.llm_client --probe` checks key/balance/latency before a big run.

A model named claude-* goes to Anthropic's Messages API instead (official SDK,
ANTHROPIC_API_KEY). Callers keep building the same DeepSeek-shaped payload;
_anthropic_call translates it, so tailoring, the judge and project selection
switch provider by changing only the model name.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import os

import requests

from src import usage

logger = logging.getLogger(__name__)

DEEPSEEK_API_BASE = "https://api.deepseek.com"

# Defaults; a run overrides them from config.yaml's `deepseek:` block via configure().
_SETTINGS: dict[str, float] = {
    "connect_timeout": 10.0,   # seconds to establish the connection
    "stall_timeout": 60.0,     # seconds with NO bytes received before we call it stalled
    "queue_timeout": 90.0,     # seconds to wait for the first real data (see module doc)
    "call_deadline": 300.0,    # hard wall-clock budget per call, retries included
    "max_retries": 2,          # retries after the first attempt
    "breaker_threshold": 3,    # consecutive "queued" calls before the breaker trips
}

# HTTP statuses where retrying can't help: say so immediately.
_FATAL = {400: "bad_request", 401: "auth", 402: "no_balance", 403: "forbidden", 422: "bad_request"}


# Optional `anthropic.effort` from config.yaml: overrides the effort mapped from
# the quality profile for every Claude call.
_SETTINGS_ANTHROPIC: dict[str, str] = {}


def configure(deepseek_cfg: dict | None, anthropic_cfg: dict | None = None) -> None:
    """Apply timeouts from config.yaml (`deepseek.connect_timeout`, etc.)."""
    for key in _SETTINGS:
        if deepseek_cfg and deepseek_cfg.get(key) is not None:
            _SETTINGS[key] = float(deepseek_cfg[key])
    if anthropic_cfg and anthropic_cfg.get("effort"):
        _SETTINGS_ANTHROPIC["effort"] = str(anthropic_cfg["effort"])


def settings() -> dict[str, float]:
    return dict(_SETTINGS)


@dataclasses.dataclass
class CallResult:
    content: str | None
    reason: str              # "ok" or why it failed (see report)
    label: str = ""
    attempts: int = 0
    elapsed_s: float = 0.0
    finish_reason: str | None = None
    detail: str = ""
    usage: dict | None = None

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


# ---- per-run state + diagnostics ----------------------------------------------
_LOCK = threading.Lock()
_RECORDS: list[CallResult] = []
_QUEUED_STREAK = {"n": 0}


def reset() -> None:
    with _LOCK:
        _RECORDS.clear()
        _QUEUED_STREAK["n"] = 0


def records() -> list[CallResult]:
    with _LOCK:
        return list(_RECORDS)


def api_unavailable() -> bool:
    """Circuit breaker: True after `breaker_threshold` consecutive calls that
    DeepSeek accepted but never started (reason "queued"). Callers should stop
    launching new work -- every further call would just wait out the queue."""
    return _QUEUED_STREAK["n"] >= int(_SETTINGS["breaker_threshold"])


def _record(res: CallResult) -> CallResult:
    with _LOCK:
        _RECORDS.append(res)
        if res.reason == "queued":
            _QUEUED_STREAK["n"] += 1
        elif res.ok:
            _QUEUED_STREAK["n"] = 0
    toks = ""
    if res.usage:
        toks = f", {res.usage.get('prompt_tokens', '?')} in / {res.usage.get('completion_tokens', '?')} out"
    msg = (f"AI call [{res.label}] {res.reason} in {res.elapsed_s:.1f}s "
           f"({res.attempts} attempt{'s' if res.attempts != 1 else ''}{toks})")
    if res.ok:
        logger.info(msg)
    else:
        logger.warning("%s -- %s", msg, res.detail or "no detail")
    return res


def is_anthropic(model: str | None) -> bool:
    return str(model or "").startswith("claude-")


def has_key(api_key: str, model: str | None) -> bool:
    """Whether the provider that serves `model` has a key to call it with."""
    return bool(os.environ.get("ANTHROPIC_API_KEY")) if is_anthropic(model) else bool(api_key)


# DeepSeek reasoning setting -> Claude effort. Thinking can't be turned off on
# Claude Opus 5.5 / Sonnet 5.5 (a 400), so "disabled" becomes the lowest effort.
_EFFORT = {"disabled": "low", "low": "low", "medium": "medium", "high": "high", "max": "max"}


def _anthropic_call(payload: dict[str, Any], label: str) -> CallResult:
    """chat_json for claude-* models. Same contract: never raises, says why."""
    import anthropic

    start = time.monotonic()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _record(CallResult(None, "no_key", label, detail="ANTHROPIC_API_KEY not set"))

    system = "\n\n".join(m["content"] for m in payload["messages"] if m["role"] == "system")
    messages = []
    for m in payload["messages"]:
        if m["role"] != "user":
            continue
        text = m["content"]
        # The user prompt is JSON whose first keys (resume, repos, bullet bank)
        # are the same for every job; only target_job onward changes. Cache that
        # prefix so job 2..N read it at a tenth of the input price.
        cut = text.find('"target_job"')
        if cut > 0:
            blocks = [{"type": "text", "text": text[:cut], "cache_control": {"type": "ephemeral"}},
                      {"type": "text", "text": text[cut:]}]
        else:
            blocks = [{"type": "text", "text": text}]
        messages.append({"role": "user", "content": blocks})

    thinking = payload.get("thinking") or {}
    reasoning = "disabled" if thinking.get("type") == "disabled" else payload.get("reasoning_effort", "high")
    effort = _SETTINGS_ANTHROPIC.get("effort") or _EFFORT.get(reasoning, "high")

    client = anthropic.Anthropic(
        timeout=anthropic.Timeout(_SETTINGS["call_deadline"], connect=_SETTINGS["connect_timeout"],
                                  read=_SETTINGS["stall_timeout"]),
        max_retries=int(_SETTINGS["max_retries"]),
    )
    attempts = 1
    try:
        with client.beta.messages.stream(
            model=payload["model"],
            max_tokens=max(int(payload.get("max_tokens") or 0), 32000),
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            # On a safety decline the API re-runs the request on a suitable model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            final = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        return _record(CallResult(None, "auth", label, attempts, time.monotonic() - start, None, str(exc)))
    except anthropic.PermissionDeniedError as exc:
        return _record(CallResult(None, "forbidden", label, attempts, time.monotonic() - start, None, str(exc)))
    except anthropic.BadRequestError as exc:
        why = "no_balance" if "credit balance" in str(exc).lower() else "bad_request"
        return _record(CallResult(None, why, label, attempts, time.monotonic() - start, None, str(exc)))
    except anthropic.RateLimitError as exc:
        return _record(CallResult(None, "rate_limited", label, attempts, time.monotonic() - start, None, str(exc)))
    except anthropic.APITimeoutError as exc:
        return _record(CallResult(None, "deadline", label, attempts, time.monotonic() - start, None, str(exc)))
    except anthropic.APIStatusError as exc:
        return _record(CallResult(None, f"http_{exc.status_code}", label, attempts, time.monotonic() - start,
                                  None, str(exc)))
    except anthropic.APIConnectionError as exc:
        return _record(CallResult(None, "network", label, attempts, time.monotonic() - start, None, str(exc)))

    u = final.usage
    read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
    write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    used = {"prompt_tokens": u.input_tokens + read + write, "completion_tokens": u.output_tokens,
            "prompt_cache_hit_tokens": read, "prompt_cache_miss_tokens": u.input_tokens,
            "cache_write_tokens": write}
    usage.record(used, model=payload["model"])
    elapsed = time.monotonic() - start
    content = "".join(b.text for b in final.content if b.type == "text")
    if final.stop_reason == "max_tokens":
        return _record(CallResult(content, "truncated", label, attempts, elapsed, "length",
                                  "hit max_tokens before finishing", used))
    if final.stop_reason == "refusal":
        return _record(CallResult(None, "refused", label, attempts, elapsed, "refusal",
                                  str(getattr(final, "stop_details", "") or ""), used))
    if not content.strip():
        return _record(CallResult(None, "empty", label, attempts, elapsed, final.stop_reason,
                                  "model returned no content", used))
    return _record(CallResult(content, "ok", label, attempts, elapsed, final.stop_reason, "", used))


# ---- the call -----------------------------------------------------------------
class _HTTPError(Exception):
    def __init__(self, status: int, body: str, retry_after: str | None = None):
        super().__init__(f"HTTP {status}: {body}")
        self.status, self.body, self.retry_after = status, body, retry_after


class _Deadline(Exception):
    pass


class _Queued(Exception):
    """Accepted (HTTP 200) but only keep-alives arrived -- never started generating."""
    def __init__(self, waited: float):
        super().__init__(f"accepted but not started after {waited:.0f}s (keep-alives only)")
        self.waited = waited


def _backoff(attempt: int, retry_after: str | None, remaining: float) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), remaining))
        except ValueError:
            pass
    return max(0.0, min(2 ** attempt + random.uniform(0, 1), remaining))


def _stream_once(body: dict, headers: dict, deadline_at: float) -> tuple[str, str | None, dict | None]:
    """One streaming attempt. Returns (content, finish_reason, usage)."""
    connect, stall = _SETTINGS["connect_timeout"], _SETTINGS["stall_timeout"]
    resp = requests.post(f"{DEEPSEEK_API_BASE}/chat/completions", headers=headers,
                         json=body, stream=True, timeout=(connect, stall))
    try:
        if resp.status_code != 200:
            text = ""
            try:
                text = resp.text[:300]
            except Exception:  # noqa: BLE001
                pass
            raise _HTTPError(resp.status_code, text, resp.headers.get("Retry-After"))
        parts: list[str] = []
        finish, used = None, None
        started, got_data = time.monotonic(), False
        for line in resp.iter_lines(decode_unicode=True):
            now = time.monotonic()
            if now > deadline_at:
                raise _Deadline() if got_data else _Queued(now - started)
            if not line or not line.startswith("data:"):
                # Blank lines / ": keep-alive": the server is holding the request.
                if not got_data and now - started > _SETTINGS["queue_timeout"]:
                    raise _Queued(now - started)
                continue
            got_data = True
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                used = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    parts.append(delta["content"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        return "".join(parts), finish, used
    finally:
        resp.close()


def chat_json(payload: dict[str, Any], api_key: str, label: str = "") -> CallResult:
    """Send a chat request (streamed) and return the full text content plus a
    diagnosis. Never raises -- check `.ok` / `.reason`."""
    if is_anthropic(payload.get("model")):
        return _anthropic_call(payload, label)
    if not api_key:
        return _record(CallResult(None, "no_key", label, detail="DEEPSEEK_API_KEY not set"))

    body = dict(payload)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    start = time.monotonic()
    deadline_at = start + _SETTINGS["call_deadline"]
    max_attempts = int(_SETTINGS["max_retries"]) + 1
    reason, detail = "failed", ""
    attempt = 0

    for attempt in range(max_attempts):
        retry_after = None
        try:
            content, finish, used = _stream_once(body, headers, deadline_at)
            usage.record(used, model=payload.get("model"))
            elapsed = time.monotonic() - start
            if finish == "length":
                return _record(CallResult(content, "truncated", label, attempt + 1, elapsed, finish,
                                          "hit max_tokens before finishing -- raise deepseek.max_tokens", used))
            if not (content or "").strip():
                return _record(CallResult(None, "empty", label, attempt + 1, elapsed, finish,
                                          "model returned no content", used))
            return _record(CallResult(content, "ok", label, attempt + 1, elapsed, finish, "", used))
        except _HTTPError as exc:
            if exc.status in _FATAL:
                return _record(CallResult(None, _FATAL[exc.status], label, attempt + 1,
                                          time.monotonic() - start, None, str(exc)))
            reason = "rate_limited" if exc.status == 429 else f"http_{exc.status}"
            detail, retry_after = str(exc), exc.retry_after
        except _Deadline:
            return _record(CallResult(None, "deadline", label, attempt + 1, time.monotonic() - start, None,
                                      f"exceeded call_deadline={_SETTINGS['call_deadline']:.0f}s"))
        except _Queued as exc:
            # Re-queuing rarely helps during an overload and each retry costs
            # another queue_timeout -- fail fast; the circuit breaker decides next.
            return _record(CallResult(None, "queued", label, attempt + 1, time.monotonic() - start, None,
                                      f"{exc} -- DeepSeek is overloaded; try later"))
        except requests.exceptions.ConnectTimeout as exc:
            reason, detail = "connect_timeout", str(exc)
        except requests.RequestException as exc:
            text = str(exc)
            stalled = "timed out" in text.lower() or "read timeout" in text.lower()
            reason = "stalled" if stalled else "network"
            detail = (f"no data for {_SETTINGS['stall_timeout']:.0f}s" if stalled else text)
        except (json.JSONDecodeError, ValueError) as exc:
            reason, detail = "bad_stream", str(exc)

        remaining = deadline_at - time.monotonic()
        if attempt + 1 >= max_attempts or remaining <= 1:
            break
        wait = _backoff(attempt, retry_after, remaining - 1)
        logger.info("AI call [%s] attempt %d %s; retrying in %.1fs", label, attempt + 1, reason, wait)
        time.sleep(wait)

    return _record(CallResult(None, reason, label, attempt + 1, time.monotonic() - start, None, detail))


# ---- reporting ------------------------------------------------------------------
def summary_line() -> str:
    recs = records()
    if not recs:
        return "AI calls: none"
    ok = [r for r in recs if r.ok]
    fails: dict[str, int] = {}
    for r in recs:
        if not r.ok:
            fails[r.reason] = fails.get(r.reason, 0) + 1
    lat = sorted(r.elapsed_s for r in ok)
    lat_s = (f"; latency p50 {statistics.median(lat):.0f}s, p95 {lat[int(0.95 * (len(lat) - 1))]:.0f}s, "
             f"max {lat[-1]:.0f}s") if lat else ""
    fail_s = (", failed: " + ", ".join(f"{k}={v}" for k, v in sorted(fails.items()))) if fails else ""
    return f"AI calls: {len(ok)}/{len(recs)} ok{fail_s}{lat_s}"


def write_report(path: str) -> str:
    """Markdown diagnostics for the run. Returns the one-line summary."""
    recs = records()
    line = summary_line()
    lines = ["# AI call diagnostics", "", line, "",
             f"Settings: connect {_SETTINGS['connect_timeout']:.0f}s, stall {_SETTINGS['stall_timeout']:.0f}s, "
             f"queue {_SETTINGS['queue_timeout']:.0f}s, deadline {_SETTINGS['call_deadline']:.0f}s/call, "
             f"retries {int(_SETTINGS['max_retries'])}", ""]
    if api_unavailable():
        lines += ["**Circuit breaker tripped:** DeepSeek accepted requests but never started them "
                  f"({int(_SETTINGS['breaker_threshold'])}+ in a row). Remaining jobs were not launched "
                  "and stay unseen, so they'll be picked up next run.", ""]
    failed = [r for r in recs if not r.ok]
    if failed:
        lines += ["## Failures", "", "| call | reason | attempts | elapsed | detail |", "|---|---|---|---|---|"]
        for r in failed:
            lines.append(f"| {r.label} | {r.reason} | {r.attempts} | {r.elapsed_s:.0f}s | "
                         f"{(r.detail or '').replace('|', '/')[:160]} |")
        lines.append("")
    slow = sorted((r for r in recs if r.ok), key=lambda r: -r.elapsed_s)[:5]
    if slow:
        lines += ["## Slowest successful calls", ""]
        lines += [f"- {r.label}: {r.elapsed_s:.0f}s" for r in slow]
        lines.append("")
    lines += ["## Reasons", "",
              "- **queued**: HTTP 200 then only keep-alives for queue_timeout -- DeepSeek is "
              "overloaded and never started the request. Retry later.",
              "- **stalled**: no bytes for stall_timeout -- retried with backoff",
              "- **deadline**: whole call exceeded call_deadline -- raise it, or use a faster profile",
              "- **truncated**: hit max_tokens -- raise deepseek.max_tokens",
              "- **rate_limited**: 429 -- lower deepseek.concurrency",
              "- **no_balance / auth**: top up / check DEEPSEEK_API_KEY (ANTHROPIC_API_KEY for claude-*) "
              "-- nothing works until fixed",
              "- **refused**: Claude declined even after its fallback model -- rare; check the posting text"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return line


def probe(api_key: str, model: str = "deepseek-flash") -> CallResult:
    """One tiny live call: confirms key, balance, connectivity, and latency."""
    payload = {
        "model": model,
        # DeepSeek requires the word "json" in the prompt when response_format is json_object.
        "messages": [{"role": "user", "content": 'Reply with exactly this json: {"ok": true}'}],
        "response_format": {"type": "json_object"},
        "max_tokens": 20,
        "thinking": {"type": "disabled"},
    }
    return chat_json(payload, api_key, label="probe")


if __name__ == "__main__":
    import argparse
    import os
    import sys

    import yaml

    import src  # noqa: F401 -- loads .env

    ap = argparse.ArgumentParser(description="DeepSeek connectivity / latency probe.")
    ap.add_argument("--probe", action="store_true", help="make one tiny live call")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if Path(args.config).exists():
        configure(yaml.safe_load(Path(args.config).read_text()).get("deepseek", {}))
    res = probe(os.environ.get("DEEPSEEK_API_KEY", ""))
    print(f"\nProbe: {res.reason} in {res.elapsed_s:.1f}s" + (f" -- {res.detail}" if res.detail else ""),
          flush=True)
    sys.exit(0 if res.ok else 1)
