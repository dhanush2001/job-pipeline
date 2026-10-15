"""Tests for the ATS keyword-match scorer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ats_score import score_resume


def test_match_rate_and_missing():
    jd = "We need Python, Kubernetes, SQL and machine learning experience."
    resume = {"skills": [{"category": "x", "entries": "Python, SQL, pandas"}],
              "summary": "Machine learning engineer."}
    r = score_resume(resume, jd)
    # posting skills: python, kubernetes, sql, machine learning -> resume has 3/4
    assert r["score"] == 75, r
    assert "kubernetes" in r["missing"] and "kubernetes" not in r["matched"]
    assert "python" in r["matched"]
    print("test_match_rate_and_missing: PASS")


def test_word_boundary_no_false_positive():
    # "java" must not be counted just because "javascript" is present
    r = score_resume({"summary": "I use JavaScript."}, "Need Java and JavaScript.")
    assert r["score"] == 50, r
    assert "java" in r["missing"] and "javascript" in r["matched"]
    print("test_word_boundary_no_false_positive: PASS")


def test_none_when_posting_has_no_skills():
    r = score_resume({"summary": "hello"}, "We value teamwork and clear communication.")
    assert r["score"] is None
    print("test_none_when_posting_has_no_skills: PASS")


if __name__ == "__main__":
    test_match_rate_and_missing()
    test_word_boundary_no_false_positive()
    test_none_when_posting_has_no_skills()
    print("\nAll ATS-score tests passed.")
