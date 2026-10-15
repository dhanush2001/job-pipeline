"""Tests for token-usage accounting + cost estimation (src/usage.py)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import usage

PRICING = {"cache_hit_input_per_1m": 0.10, "cache_miss_input_per_1m": 1.00, "output_per_1m": 2.00}


def test_counts_and_cost_with_cache_split():
    usage.reset()
    # Two calls, each reporting DeepSeek's cache hit/miss split.
    usage.record({"prompt_tokens": 1000, "completion_tokens": 500,
                  "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200})
    usage.record({"prompt_tokens": 1000, "completion_tokens": 500,
                  "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200})
    s = usage.session_summary(PRICING)
    assert s["calls"] == 2
    assert s["input_tokens"] == 2000 and s["output_tokens"] == 1000
    assert s["total_tokens"] == 3000
    assert s["cache_hit_tokens"] == 1600 and s["cache_miss_tokens"] == 400
    # cost = 1600/1e6*0.10 + 400/1e6*1.00 + 1000/1e6*2.00
    expected = round(1600/1e6*0.10 + 400/1e6*1.00 + 1000/1e6*2.00, 4)
    assert s["estimated_cost_usd"] == expected, (s["estimated_cost_usd"], expected)
    print("test_counts_and_cost_with_cache_split: PASS")


def test_no_cache_split_counts_all_input_as_miss():
    usage.reset()
    usage.record({"prompt_tokens": 1000, "completion_tokens": 200})  # no hit/miss fields
    usage.record(None)   # ignored
    s = usage.session_summary(PRICING)
    assert s["calls"] == 1                      # None didn't count as a call
    assert s["cache_miss_tokens"] == 1000 and s["cache_hit_tokens"] == 0
    print("test_no_cache_split_counts_all_input_as_miss: PASS")


def test_session_log_accumulates_running_total():
    usage.reset()
    with tempfile.TemporaryDirectory() as tmp:
        log = str(Path(tmp) / "usage_log.jsonl")
        usage.record({"prompt_tokens": 100, "completion_tokens": 50,
                      "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 100})
        r1 = usage.append_session_log(log, PRICING, extra={"jobs_tailored": 1})
        assert r1["totals"]["sessions"] == 1
        assert r1["session"]["jobs_tailored"] == 1

        usage.reset()
        usage.record({"prompt_tokens": 300, "completion_tokens": 100,
                      "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 300})
        r2 = usage.append_session_log(log, PRICING)
        assert r2["totals"]["sessions"] == 2
        assert r2["totals"]["input_tokens"] == 400      # 100 + 300 across both runs
        assert r2["totals"]["output_tokens"] == 150
        assert "DeepSeek token usage" in usage.format_report(r2)
    print("test_session_log_accumulates_running_total: PASS")


if __name__ == "__main__":
    test_counts_and_cost_with_cache_split()
    test_no_cache_split_counts_all_input_as_miss()
    test_session_log_accumulates_running_total()
    print("\nAll usage tests passed.")
