"""
Tests for the tailored-vs-master evaluator. Synthetic records, no API calls.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate_record, load_records, render_report, save_record
from src.quality import GroundingReport

MASTER = {
    "summary": "MS CS student.",
    "skills": [{"category": "Languages", "entries": "Python, SQL"}],
    "experience": [{"id": "acme", "company": "Acme",
                    "bullets": ["Shipped a React admin dashboard used daily by a 12-person team",
                                "Integrated the frontend with a Java Servlet backend via REST APIs"]}],
    "projects": [{"name": "Eval Harness", "bullets": ["Ran 15,000 experiments across 5 models"]},
                 {"name": "SynthEval", "bullets": ["Reduced evaluation error by 36%"]}],
}
CFG = {"graduation_year": 2026, "max_required_experience_years": 2}


def _record(desc, tailored, grounding=None, letter=""):
    return {"version": 1, "job": {"title": "SWE Intern", "company": "Stripe", "description": desc},
            "tailored": tailored, "ats": None,
            "grounding": grounding or {"unsupported_numbers": [], "mismatched_context": []},
            "cover_letter": letter, "judge": {}, "resume_pdf": "x.pdf"}


def test_flags_ineligible_posting():
    desc = "Python and SQL. Master's plus three (3) years of experience. Graduating fall 2027 or spring 2028."
    r = evaluate_record(_record(desc, MASTER), MASTER, CFG)
    assert r["verdict"] == "REVIEW"
    assert any("graduation window" in f for f in r["flags"]), r["flags"]
    assert any("3+ years" in f for f in r["flags"]), r["flags"]
    print("test_flags_ineligible_posting: PASS")


def test_labels_kept_rewritten_new_and_dropped_projects():
    tailored = {
        "summary": "MS CS student.",
        "experience": [{"id": "acme", "bullets": [
            "Shipped a React admin dashboard used daily by a 12-person team",            # kept
            "Owned end-to-end data flow by integrating the frontend with a **Java Servlet** "
            "backend via **REST APIs**",                                                # rewritten
            "Mentored three interns on Kubernetes deployments"]}],                      # new
        "projects": [{"name": "Eval Harness", "bullets": ["Ran 15,000 experiments across 5 models"]}],
    }
    r = evaluate_record(_record("Python, SQL, Java, REST APIs.", tailored), MASTER, CFG)
    assert r["counts"] == {"kept": 2, "rewritten": 1, "new": 1}, r["counts"]
    assert r["dropped_projects"] == ["SynthEval"]
    assert any("no measurable result" in s for s in r["style"])
    print("test_labels_kept_rewritten_new_and_dropped_projects: PASS")


def test_claim_flags_and_cover_letter_checks():
    g = {"unsupported_numbers": ["47%"], "mismatched_context": ["20% (in: ...)"]}
    letter = "I am writing to express my interest in this role. " + "word " * 420
    r = evaluate_record(_record("Python.", MASTER, g, letter), MASTER, CFG)
    assert any("UNVERIFIED NUMBERS: 47%" in f for f in r["flags"])
    assert any("different claim" in f for f in r["flags"])
    assert any("generic line" in s for s in r["style"])
    assert any("never names the company" in s for s in r["style"])
    assert any("words" in s for s in r["style"])
    print("test_claim_flags_and_cover_letter_checks: PASS")


def test_save_and_load_roundtrip_and_report():
    with tempfile.TemporaryDirectory() as d:
        pdf = f"{d}/stripe/123.pdf"
        rep = GroundingReport("Stripe", ["47%"], [], ["20% (in: x)"])
        path = save_record(pdf, {"title": "SWE", "company": "Stripe", "description": "Python."},
                           MASTER, {"score": 80, "matched": ["python"], "missing": ["sql"]}, rep)
        assert Path(path).exists() and path.endswith("123.json")
        Path(f"{d}/junk.json").write_text("{not json")         # ignored, not fatal
        recs = load_records(d)
        assert len(recs) == 1 and recs[0]["grounding"]["unsupported_numbers"] == ["47%"]
        report = render_report([evaluate_record(recs[0], MASTER, CFG)])
    assert "| REVIEW | Stripe |" in report and "UNVERIFIED NUMBERS" in report
    print("test_save_and_load_roundtrip_and_report: PASS")


if __name__ == "__main__":
    test_flags_ineligible_posting()
    test_labels_kept_rewritten_new_and_dropped_projects()
    test_claim_flags_and_cover_letter_checks()
    test_save_and_load_roundtrip_and_report()
    print("\nAll evaluation tests passed.")
