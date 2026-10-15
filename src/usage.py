"""
usage.py -- accumulate DeepSeek token usage for one run and estimate cost.

Token COUNTS are exact: they come straight from each API response's `usage`
object (prompt_tokens / completion_tokens, and DeepSeek's cache-hit/miss split
when present). Only the per-token PRICES are estimates -- set `deepseek.pricing`
in config.yaml to your real platform.deepseek.com rates to make cost exact.

A module-level accumulator collects every call in the current process; main.py
resets it at the start of a run and reports + logs it at the end, so you get
per-session numbers plus a running total across sessions.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path


@dataclasses.dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0       # prompt tokens (= cache hit + miss)
    output_tokens: int = 0      # completion tokens
    cache_hit_tokens: int = 0   # input billed at the cheaper cached rate
    cache_miss_tokens: int = 0  # input billed at the full rate

    def add(self, u: dict | None) -> None:
        if not u:
            return
        self.calls += 1
        prompt = int(u.get("prompt_tokens", 0) or 0)
        self.input_tokens += prompt
        self.output_tokens += int(u.get("completion_tokens", 0) or 0)
        hit = u.get("prompt_cache_hit_tokens")
        miss = u.get("prompt_cache_miss_tokens")
        if hit is not None or miss is not None:
            self.cache_hit_tokens += int(hit or 0)
            self.cache_miss_tokens += int(miss or 0)
        else:
            self.cache_miss_tokens += prompt  # no split reported -> count all as miss

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, pricing: dict) -> float:
        """$ cost from per-1M-token rates. Uncached input is billed when the API
        doesn't report a cache split (cache_miss holds the whole prompt then)."""
        hit = (self.cache_hit_tokens / 1_000_000) * pricing.get("cache_hit_input_per_1m", 0.0)
        miss = (self.cache_miss_tokens / 1_000_000) * pricing.get("cache_miss_input_per_1m", 0.0)
        out = (self.output_tokens / 1_000_000) * pricing.get("output_per_1m", 0.0)
        return hit + miss + out


# ESTIMATE rates ($ per 1M tokens). DeepSeek's real rates vary by model and
# peak/off-peak -- override via config `deepseek.pricing`. These are only used
# to turn the exact token counts into a ballpark dollar figure.
DEFAULT_PRICING = {
    "cache_hit_input_per_1m": 0.07,
    "cache_miss_input_per_1m": 0.27,
    "output_per_1m": 1.10,
}

_RUN = Usage()


def reset() -> None:
    global _RUN
    _RUN = Usage()


def record(usage_dict: dict | None) -> None:
    """Called by each API wrapper with the response's `usage` object."""
    _RUN.add(usage_dict)


def current() -> Usage:
    return _RUN


def session_summary(pricing: dict | None = None) -> dict:
    p = pricing or DEFAULT_PRICING
    u = _RUN
    return {
        "calls": u.calls,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
        "cache_hit_tokens": u.cache_hit_tokens,
        "cache_miss_tokens": u.cache_miss_tokens,
        "estimated_cost_usd": round(u.cost_usd(p), 4),
    }


def append_session_log(log_path: str, pricing: dict | None = None, extra: dict | None = None) -> dict:
    """
    Append this session's usage to a JSONL history file and return
    {"session": {...}, "totals": {...}} where totals sum every logged session.
    One line per run, so you get per-session numbers and a running grand total.
    """
    s = session_summary(pricing)
    s["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    if extra:
        s.update(extra)

    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(s) + "\n")

    totals = {"sessions": 0, "calls": 0, "input_tokens": 0, "output_tokens": 0,
              "total_tokens": 0, "estimated_cost_usd": 0.0}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        totals["sessions"] += 1
        for k in ("calls", "input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"):
            totals[k] += row.get(k, 0) or 0
    totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 4)
    return {"session": s, "totals": totals}


def finalize(config: dict, logger, extra: dict | None = None) -> dict:
    """Append this run to the usage log, write the .txt report, log a one-line
    summary. Shared by the scan flow and the paste flows so they report the same
    way and feed the same running total. No-op-safe when no calls were made."""
    ds = config.get("deepseek", {}) or {}
    out = config.get("output", {}) or {}
    pricing = ds.get("pricing") or DEFAULT_PRICING
    log_path = out.get("usage_log", "data/usage_log.jsonl")
    report_path = out.get("token_usage", "output/token_usage.txt")
    result = append_session_log(log_path, pricing, extra=extra)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(format_report(result), encoding="utf-8")
    s, t = result["session"], result["totals"]
    logger.info("Tokens this run: %s in + %s out = %s total / %d calls; est. $%.4f  (%s)",
                f"{s['input_tokens']:,}", f"{s['output_tokens']:,}", f"{s['total_tokens']:,}",
                s["calls"], s["estimated_cost_usd"], report_path)
    logger.info("Tokens all %d runs: %s total; est. $%.4f  (%s)",
                t["sessions"], f"{t['total_tokens']:,}", t["estimated_cost_usd"], log_path)
    return result


def format_report(result: dict) -> str:
    """Human-readable .txt report of this session + the running total."""
    s, t = result["session"], result["totals"]
    return "\n".join([
        "# DeepSeek token usage",
        "",
        "Token counts are exact (from the API). Cost is an ESTIMATE -- set your",
        "real rates under deepseek.pricing in config.yaml to make it exact.",
        "",
        f"This session ({s.get('timestamp','')}):",
        f"  API calls:      {s['calls']}",
        f"  Input tokens:   {s['input_tokens']:,}  (cache hit {s['cache_hit_tokens']:,} / miss {s['cache_miss_tokens']:,})",
        f"  Output tokens:  {s['output_tokens']:,}",
        f"  Total tokens:   {s['total_tokens']:,}",
        f"  Est. cost:      ${s['estimated_cost_usd']:.4f}",
        "",
        f"All sessions so far ({t['sessions']} runs):",
        f"  Calls:          {t['calls']}",
        f"  Input tokens:   {t['input_tokens']:,}",
        f"  Output tokens:  {t['output_tokens']:,}",
        f"  Total tokens:   {t['total_tokens']:,}",
        f"  Est. cost:      ${t['estimated_cost_usd']:.4f}",
        "",
    ]) + "\n"
