"""The claude-* path of llm_client: request translation, usage, failure reasons.
Uses a fake SDK client -- no network, no key needed beyond a dummy env var."""
import types

import anthropic

from src import llm_client, usage


class _FakeStream:
    def __init__(self, final):
        self.final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.final


def _fake_client(captured, stop="end_turn", text='{"ok": true}'):
    final = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="thinking", thinking=""),
                 types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop, stop_details=None,
        usage=types.SimpleNamespace(input_tokens=500, output_tokens=2000,
                                    cache_read_input_tokens=15000, cache_creation_input_tokens=0))

    class Client:
        def __init__(self, **kw):
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(stream=self._stream))

        def _stream(self, **kw):
            captured.update(kw)
            return _FakeStream(final)
    return Client


def _payload(**extra):
    p = {"model": "claude-sonnet-5-5",
         "messages": [{"role": "system", "content": "Respond with only JSON."},
                      {"role": "user", "content": '{"master_resume": {"a": 1}, "target_job": {"t": 2}}'}],
         "response_format": {"type": "json_object"}, "temperature": 0.4,
         "thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    p.update(extra)
    return p


def test_payload_is_translated_for_claude(monkeypatch):
    captured = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(captured))
    usage.reset()
    res = llm_client.chat_json(_payload(), api_key="", label="t")
    assert res.ok and res.content == '{"ok": true}'
    # No temperature / response_format (400s on these models); adaptive thinking + effort instead.
    assert "temperature" not in captured and "response_format" not in captured
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"] == {"effort": "high"}
    assert captured["fallbacks"] == "default"
    # The shared resume prefix is cached; the per-job part is not.
    first, second = captured["messages"][0]["content"]
    assert first["text"].startswith('{"master_resume"') and first["cache_control"]
    assert second["text"].startswith('"target_job"') and "cache_control" not in second
    assert captured["system"][0]["text"] == "Respond with only JSON."
    u = usage.current()
    assert (u.input_tokens, u.cache_hit_tokens, u.output_tokens) == (15500, 15000, 2000)


def test_disabled_thinking_becomes_low_effort(monkeypatch):
    captured = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(captured))
    llm_client.chat_json(_payload(thinking={"type": "disabled"}), api_key="", label="t")
    assert captured["output_config"] == {"effort": "low"} and captured["thinking"] == {"type": "adaptive"}


def test_truncation_and_missing_key_are_reported(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client({}, stop="max_tokens"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_client.chat_json(_payload(), api_key="deepseek-key", label="t").reason == "no_key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert llm_client.chat_json(_payload(), api_key="", label="t").reason == "truncated"
    assert llm_client.has_key("", "claude-opus-5-5") and not llm_client.has_key("", "deepseek-flash")


def test_each_model_is_priced_at_its_own_rates():
    usage.reset()
    usage.record({"prompt_tokens": 1_000_000, "completion_tokens": 0}, model="deepseek-flash")
    usage.record({"prompt_tokens": 1_000_000, "completion_tokens": 0,
                  "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1_000_000},
                 model="claude-sonnet-5-5")
    s = usage.session_summary({"cache_miss_input_per_1m": 0.30},
                              {"claude-sonnet-5-5": {"cache_miss_input_per_1m": 2.00}})
    assert s["estimated_cost_usd"] == 2.30
