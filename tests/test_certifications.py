"""
Tests for role-gated certifications (e.g. the GUVI Java course should appear
only on Java/Spring postings, not on every resume). Deterministic, no API key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.render_resume import select_certifications

GUVI = {"name": "Java Programming", "issuer": "GUVI", "date": "2022",
        "include_if": ["java", "spring", "jvm", "j2ee", "servlet"]}
ALWAYS = {"name": "AWS Cloud Practitioner", "issuer": "AWS", "date": "2024"}


def test_conditional_cert_shows_on_matching_job():
    kept = select_certifications([GUVI], "Backend Engineer. We use Java and Spring Boot.")
    assert [c["name"] for c in kept] == ["Java Programming"]
    print("test_conditional_cert_shows_on_matching_job: PASS")


def test_conditional_cert_hidden_when_irrelevant():
    kept = select_certifications([GUVI], "ML Engineer. Python, PyTorch, and SQL.")
    assert kept == [], f"Java cert should be hidden: {kept}"
    print("test_conditional_cert_hidden_when_irrelevant: PASS")


def test_java_does_not_match_javascript():
    """The classic false positive: 'java' must not fire on 'JavaScript'."""
    kept = select_certifications([GUVI], "Frontend Engineer. React, TypeScript, JavaScript.")
    assert kept == [], f"'java' wrongly matched JavaScript: {kept}"
    print("test_java_does_not_match_javascript: PASS")


def test_trigger_is_case_insensitive():
    kept = select_certifications([GUVI], "Software Engineer working with JAVA and the JVM.")
    assert [c["name"] for c in kept] == ["Java Programming"]
    print("test_trigger_is_case_insensitive: PASS")


def test_unconditional_cert_always_kept():
    """A cert with no include_if shows regardless of the job (even with no text)."""
    assert [c["name"] for c in select_certifications([ALWAYS], "")] == ["AWS Cloud Practitioner"]
    assert [c["name"] for c in select_certifications([ALWAYS], "Java role")] == ["AWS Cloud Practitioner"]
    print("test_unconditional_cert_always_kept: PASS")


def test_no_jobtext_hides_conditional_but_keeps_unconditional():
    """A stub/dry render (no job text) shows only unconditional certs -- a
    conditional one needs a real signal, never appears by default."""
    kept = select_certifications([GUVI, ALWAYS], "")
    assert [c["name"] for c in kept] == ["AWS Cloud Practitioner"]
    print("test_no_jobtext_hides_conditional_but_keeps_unconditional: PASS")


def test_include_if_stripped_from_output():
    """The trigger list is internal plumbing and must not leak into the rendered
    resume data."""
    kept = select_certifications([GUVI], "Java backend role")
    assert "include_if" not in kept[0], "include_if leaked into render data"
    print("test_include_if_stripped_from_output: PASS")


def test_empty_and_missing_sections():
    assert select_certifications([], "Java") == []
    assert select_certifications(None, "Java") == []
    print("test_empty_and_missing_sections: PASS")


if __name__ == "__main__":
    test_conditional_cert_shows_on_matching_job()
    test_conditional_cert_hidden_when_irrelevant()
    test_java_does_not_match_javascript()
    test_trigger_is_case_insensitive()
    test_unconditional_cert_always_kept()
    test_no_jobtext_hides_conditional_but_keeps_unconditional()
    test_include_if_stripped_from_output()
    test_empty_and_missing_sections()
    print("\nAll certification tests passed.")
