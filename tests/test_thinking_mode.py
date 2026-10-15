"""
Tests for DeepSeek thinking-mode parameters.

These matter because the API's defaults are the expensive ones: omitting the
thinking parameter means "enabled at high effort", and temperature is silently
ignored whenever thinking is on. Both are easy to get wrong and neither fails
loudly, so they get asserted rather than assumed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tailor_resume import _thinking_params


def test_disabled_sends_explicit_toggle_and_temperature():
    p = _thinking_params("disabled", 0.4)
    assert p["thinking"] == {"type": "disabled"}, "Must send an EXPLICIT disable"
    assert p["temperature"] == 0.4, "Temperature is honored when thinking is off"
    assert "reasoning_effort" not in p
    print("test_disabled_sends_explicit_toggle_and_temperature: PASS")


def test_enabled_omits_temperature():
    """Temperature has no effect in thinking mode; sending it is misleading."""
    for effort in ("low", "high", "max"):
        p = _thinking_params(effort, 0.4)
        assert p["thinking"] == {"type": "enabled"}
        assert p["reasoning_effort"] == effort
        assert "temperature" not in p, f"temperature sent with reasoning={effort}"
    print("test_enabled_omits_temperature: PASS")


def test_alias_efforts_map_to_high():
    for alias in ("medium", "xhigh"):
        p = _thinking_params(alias, 0.4)
        assert p["reasoning_effort"] == "high", f"{alias} should map to high"
    print("test_alias_efforts_map_to_high: PASS")


def test_unknown_value_falls_back_to_disabled():
    """A typo must not silently inherit the API's expensive 'high' default."""
    p = _thinking_params("hihg", 0.4)
    assert p["thinking"] == {"type": "disabled"}
    assert p["temperature"] == 0.4
    print("test_unknown_value_falls_back_to_disabled: PASS")


def test_empty_and_none_default_to_disabled():
    for value in ("", None, "off", "none"):
        p = _thinking_params(value, 0.4)
        assert p["thinking"] == {"type": "disabled"}, f"{value!r} should disable"
    print("test_empty_and_none_default_to_disabled: PASS")


def test_case_insensitive():
    p = _thinking_params("HIGH", 0.4)
    assert p["reasoning_effort"] == "high"
    print("test_case_insensitive: PASS")


def test_payload_never_omits_thinking_key():
    """
    The whole point: an absent thinking key means enabled/high. Every path
    must return one.
    """
    for value in ("disabled", "low", "high", "max", "medium", "garbage", "", None):
        p = _thinking_params(value, 0.4)
        assert "thinking" in p, f"thinking key missing for {value!r} -> defaults to high"
    print("test_payload_never_omits_thinking_key: PASS")


if __name__ == "__main__":
    test_disabled_sends_explicit_toggle_and_temperature()
    test_enabled_omits_temperature()
    test_alias_efforts_map_to_high()
    test_unknown_value_falls_back_to_disabled()
    test_empty_and_none_default_to_disabled()
    test_case_insensitive()
    test_payload_never_omits_thinking_key()
    print("\nAll thinking-mode tests passed.")
