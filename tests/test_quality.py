"""
Tests for the grounding / fabrication checker.

The cases here are modeled on how LLM resume tailoring actually goes wrong:
plausible-sounding metrics and employer names that appear nowhere in the
source. These are deterministic and need no API key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.quality import build_source_corpus, verify_grounding, write_grounding_report
from src.tailor_resume import ApplicationPackage

MASTER = Path(__file__).parent.parent / "data" / "master_resume.example.yaml"


def _corpus(job_description: str = ""):
    master = yaml.safe_load(MASTER.read_text())
    repos = [{"name": "repo-radar", "description": "Agentic GitHub monitor",
              "language": "Python", "readme_excerpt": "Sends a daily digest via Slack."}]
    return build_source_corpus(master, repos, job_description)


def test_faithful_content_is_clean():
    """Content drawn from the master resume must not be flagged."""
    corpus = _corpus()
    texts = [
        "Built an internal tool that reduced deploy time by 30%",
        "Wrote integration tests raising coverage from 60% to 85%",
    ]
    report = verify_grounding(texts, corpus, "job-1")
    assert not report.unsupported_numbers, f"False positive: {report.unsupported_numbers}"
    assert report.severity == "clean", f"Expected clean, got {report.severity}"
    print("test_faithful_content_is_clean: PASS")


def test_invented_metric_is_caught():
    """The classic failure: a beautiful, entirely fabricated number."""
    corpus = _corpus()
    texts = ["Reduced p99 latency by 47% and scaled the service to 2,500,000 daily users"]
    report = verify_grounding(texts, corpus, "job-2")
    assert "47%" in report.unsupported_numbers, f"Missed 47%: {report.unsupported_numbers}"
    assert "2,500,000" in report.unsupported_numbers, f"Missed user count: {report.unsupported_numbers}"
    assert report.severity == "high"
    print(f"test_invented_metric_is_caught: PASS (flagged {report.unsupported_numbers})")


def test_invented_employer_is_caught():
    corpus = _corpus()
    texts = ["Led backend work at Palantir alongside the infrastructure team"]
    report = verify_grounding(texts, corpus, "job-3")
    assert any("Palantir" in e for e in report.unsupported_entities), \
        f"Missed invented employer: {report.unsupported_entities}"
    print("test_invented_employer_is_caught: PASS")


def test_calendar_years_are_not_flagged_as_metrics():
    """A year like '2027' (from a 'Summer 2027 intern' posting) is not a
    fabricated metric -- flagging it is pure noise. But a real metric with the
    same magnitude carrying a %/x suffix is still caught."""
    corpus = _corpus()
    report = verify_grounding(["Seeking a Summer 2027 internship; graduating 2026."], corpus, "job-yr")
    assert "2027" not in report.unsupported_numbers and "2026" not in report.unsupported_numbers
    assert report.unsupported_numbers == []
    # A suffixed number is still a metric and still flagged (not a bare year).
    r2 = verify_grounding(["Improved throughput by 87x"], corpus, "job-yr2")
    assert "87x" in r2.unsupported_numbers
    print("test_calendar_years_are_not_flagged_as_metrics: PASS")


def test_job_description_terms_are_not_fabrication():
    """
    Echoing a technology named in the posting is legitimate tailoring. If this
    flagged, the report would be noise and get ignored.
    """
    corpus = _corpus(job_description="We use Kubernetes and Terraform on AWS.")
    texts = ["Deployed services on Kubernetes using Terraform"]
    report = verify_grounding(texts, corpus, "job-4")
    assert not report.unsupported_entities, f"False positive: {report.unsupported_entities}"
    print("test_job_description_terms_are_not_fabrication: PASS")


def test_github_repo_content_counts_as_source():
    corpus = _corpus()
    texts = ["Built repo-radar, an agentic monitor that sends a daily digest via Slack"]
    report = verify_grounding(texts, corpus, "job-5")
    assert not report.unsupported_entities, f"Repo data should ground this: {report.unsupported_entities}"
    print("test_github_repo_content_counts_as_source: PASS")


def test_action_verbs_are_not_flagged_as_entities():
    corpus = _corpus()
    texts = ["Designed a service. Implemented caching. Migrated the database."]
    report = verify_grounding(texts, corpus, "job-6")
    assert not report.unsupported_entities, f"Verbs flagged as entities: {report.unsupported_entities}"
    print("test_action_verbs_are_not_flagged_as_entities: PASS")


def test_numbers_outrank_names_in_severity():
    corpus = _corpus()
    names_only = verify_grounding(["Worked with Datadog"], corpus, "a")
    with_number = verify_grounding(["Cut costs by 63%"], corpus, "b")
    assert names_only.severity == "low"
    assert with_number.severity == "high"
    print("test_numbers_outrank_names_in_severity: PASS")


def test_bullet_bank_numbers_count_as_source():
    """
    A metric copied from the bullet bank is real work, not fabrication -- once
    the bank is in the corpus, that number must not be flagged.
    """
    master = yaml.safe_load(MASTER.read_text())
    bank = "Improved open-ended clinical VQA accuracy by +5.8% on PathVQA."

    without_bank = build_source_corpus(master, None, "")
    with_bank = build_source_corpus(master, None, "", bank)

    text = ["Boosted open-ended VQA accuracy by 5.8% on the PathVQA benchmark"]
    assert "5.8%" in verify_grounding(text, without_bank, "x").unsupported_numbers, \
        "Sanity: without the bank the number should look unsupported"
    assert not verify_grounding(text, with_bank, "y").unsupported_numbers, \
        "With the bank in the corpus, its real metric must be grounded"
    print("test_bullet_bank_numbers_count_as_source: PASS")


def test_cover_letter_numbers_are_grounded():
    """
    A fabricated metric is just as damaging in a cover letter as in a resume
    bullet, so groundable_texts() must feed the cover letter and LinkedIn
    drafts into the checker -- not only the resume body.
    """
    corpus = _corpus()
    pkg = ApplicationPackage(
        summary="Graduate engineer focused on ML systems.",
        skills=["Python"],
        experience_bullets={},
        projects=[],
        cover_letter="In my last role I boosted revenue by 73% across the team.",
        referral_note="Hi, I applied for the New Grad SWE role and would love to connect.",
        referral_dm="Would you be open to referring me for the role?",
    )

    texts = pkg.groundable_texts()
    assert pkg.cover_letter in texts, "Cover letter must be checked"
    assert pkg.referral_note in texts and pkg.referral_dm in texts, "Drafts must be checked"

    report = verify_grounding(texts, corpus, "job-cl")
    assert "73%" in report.unsupported_numbers, \
        f"Fabricated cover-letter metric slipped through: {report.unsupported_numbers}"
    assert report.severity == "high"
    print("test_cover_letter_numbers_are_grounded: PASS")


def test_report_only_lists_flagged_jobs():
    corpus = _corpus()
    reports = [
        verify_grounding(["Built an internal tool that reduced deploy time by 30%"], corpus, "clean-job"),
        verify_grounding(["Improved throughput by 91%"], corpus, "dirty-job"),
    ]
    out = Path("/tmp/_grounding_test.md")
    count = write_grounding_report(reports, str(out))
    assert count == 1, f"Expected 1 flagged, got {count}"
    text = out.read_text()
    assert "dirty-job" in text
    assert "clean-job" not in text, "Clean jobs should not clutter the report"
    assert "91%" in text
    out.unlink(missing_ok=True)
    print("test_report_only_lists_flagged_jobs: PASS")


if __name__ == "__main__":
    test_faithful_content_is_clean()
    test_invented_metric_is_caught()
    test_invented_employer_is_caught()
    test_calendar_years_are_not_flagged_as_metrics()
    test_job_description_terms_are_not_fabrication()
    test_github_repo_content_counts_as_source()
    test_action_verbs_are_not_flagged_as_entities()
    test_numbers_outrank_names_in_severity()
    test_bullet_bank_numbers_count_as_source()
    test_cover_letter_numbers_are_grounded()
    test_report_only_lists_flagged_jobs()
    print("\nAll grounding tests passed.")
