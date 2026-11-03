"""
Tests for _parse_package -- turning a raw model response into an
ApplicationPackage. The point is robustness: a syntactically-valid response with
the wrong types (e.g. {"experience": null}) must degrade to a safe package or
None, never raise -- otherwise one bad response aborts the whole batch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tailor_resume import _parse_package


def test_null_fields_do_not_raise():
    """`{"experience": null}` used to raise TypeError (iterating None). It must
    now coerce to empty and return a valid package."""
    pkg = _parse_package('{"experience": null, "projects": null, "skills": null, '
                         '"summary": null, "cover_letter": null, "referral_note": null}')
    assert pkg is not None
    assert pkg.experience_bullets == {}
    assert pkg.projects == []
    assert pkg.skills == []
    assert pkg.summary == ""
    assert pkg.cover_letter == ""
    print("test_null_fields_do_not_raise: PASS")


def test_missing_fields_default():
    pkg = _parse_package('{}')
    assert pkg is not None and pkg.experience_bullets == {} and pkg.skills == []
    print("test_missing_fields_default: PASS")


def test_non_dict_response_returns_none():
    assert _parse_package('[1, 2, 3]') is None
    assert _parse_package('"just a string"') is None
    assert _parse_package('42') is None
    print("test_non_dict_response_returns_none: PASS")


def test_unparseable_returns_none():
    assert _parse_package('{not valid json') is None
    print("test_unparseable_returns_none: PASS")


def test_malformed_experience_entries_are_skipped():
    """Entries that aren't dicts, or lack an id, are dropped rather than raising."""
    pkg = _parse_package('{"experience": [null, "oops", {"bullets": ["x"]}, '
                         '{"id": "job-1", "bullets": ["real"]}]}')
    assert pkg is not None
    assert pkg.experience_bullets == {"job-1": ["real"]}
    print("test_malformed_experience_entries_are_skipped: PASS")


def test_wellformed_response_parses():
    raw = ('{"experience": [{"id": "x", "bullets": ["a", "b"]}], '
           '"skills": ["Python"], "summary": "hi", "cover_letter": "  cl  "}')
    pkg = _parse_package(raw)
    assert pkg.experience_bullets == {"x": ["a", "b"]}
    assert pkg.skills == ["Python"]
    assert pkg.summary == "hi"
    assert pkg.cover_letter == "cl"
    print("test_wellformed_response_parses: PASS")


def test_nested_gaps_are_flattened():
    """Live failure: "gaps" held a nested list and '; '.join() crashed the run."""
    from src.tailor_resume import _normalize_analysis
    a = _normalize_analysis({"gaps": ["No Go", ["No Rust", "No k8s"], None, "", 3],
                             "seniority_signal": "new grad"})
    assert a["gaps"] == ["No Go", "No Rust", "No k8s", "3"], a
    assert "; ".join(a["gaps"]) and a["seniority_signal"] == "new grad"
    assert _normalize_analysis("oops") == {} and _normalize_analysis(None) == {}
    print("test_nested_gaps_are_flattened: PASS")


if __name__ == "__main__":
    test_nested_gaps_are_flattened()
    test_null_fields_do_not_raise()
    test_missing_fields_default()
    test_non_dict_response_returns_none()
    test_unparseable_returns_none()
    test_malformed_experience_entries_are_skipped()
    test_wellformed_response_parses()
    print("\nAll parse-package tests passed.")
