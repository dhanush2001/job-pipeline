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
    # Abbreviated years in a LinkedIn draft ("graduating Dec '26") are not metrics either.
    assert verify_grounding(["Graduating Dec '26, open to roles."], corpus).unsupported_numbers == []
    # A suffixed number is still a metric and still flagged (not a bare year).
    r2 = verify_grounding(["Improved throughput by 87x"], corpus, "job-yr2")
    assert "87x" in r2.unsupported_numbers
    print("test_calendar_years_are_not_flagged_as_metrics: PASS")


def test_job_description_cannot_ground_a_number():
    """A metric that appears ONLY in the posting ("we aim to cut latency 47%") is
    the employer's target, not the candidate's achievement. Entities from the JD
    are fair to echo, but numbers must be grounded in the candidate's own sources."""
    candidate_only = _corpus()  # no JD
    full = _corpus(job_description="We aim to reduce p99 latency by 47%.")
    texts = ["Reduced p99 latency by 47%"]
    # Sanity: the full corpus (incl. JD) contains 47%, so default behaviour grounds it.
    assert "47%" not in verify_grounding(texts, full).unsupported_numbers
    # Correct behaviour: numbers checked against candidate-only sources -> flagged.
    report = verify_grounding(texts, full, number_corpus=candidate_only)
    assert "47%" in report.unsupported_numbers, report.unsupported_numbers
    print("test_job_description_cannot_ground_a_number: PASS")


def test_numbers_use_token_boundaries_not_substring():
    """'20%' must NOT be grounded just because the digits '20' sit inside an
    unrelated number like the year 2024 (the old substring bug)."""
    corpus = "worked on the project across 2024 and 2025"  # contains 2024, no standalone 20
    flagged = verify_grounding(["Improved conversion by 20%"], corpus, number_corpus=corpus)
    assert "20%" in flagged.unsupported_numbers, flagged.unsupported_numbers
    # A genuine standalone 20% still grounds it.
    ok = verify_grounding(["Improved conversion by 20%"], "we saw a 20% lift", number_corpus="we saw a 20% lift")
    assert "20%" not in ok.unsupported_numbers
    print("test_numbers_use_token_boundaries_not_substring: PASS")


def test_comma_formatted_numbers_are_grounded():
    """A thousands-separated metric in the source ('ran 15,000 experiments') must
    ground a '15,000' claim, even though the bare digits '15000' aren't a literal
    substring of '15,000'. Regression guard for the token-boundary fix."""
    corpus = "ran 15,000 experiments and processed 2,500,000 rows"
    r = verify_grounding(["delivered 15,000 experiments over 2,500,000 rows"],
                         corpus, number_corpus=corpus)
    assert r.unsupported_numbers == [], r.unsupported_numbers
    print("test_comma_formatted_numbers_are_grounded: PASS")


def test_number_must_mean_the_same_thing():
    """'Improved latency by 20%' is NOT supported by 'served 20 users': same
    digits, different kind of quantity. Same-kind numbers about a different
    subject are flagged for review, not passed silently."""
    users = "served 20 users on the internal dashboard"
    r = verify_grounding(["Improved latency by 20%"], users, number_corpus=users)
    assert "20%" in r.unsupported_numbers and r.severity == "high", r
    cost = "reduced cloud cost by 20% with autoscaling"
    r2 = verify_grounding(["Improved latency by 20%"], cost, number_corpus=cost)
    assert not r2.unsupported_numbers and r2.mismatched_context and r2.severity == "medium", r2
    same = "cut p99 latency by 20% with caching"
    r3 = verify_grounding(["Cut p99 latency by 20%"], same, number_corpus=same)
    assert r3.severity == "clean", r3
    mult = "3 kernels in total"
    assert "3x" in verify_grounding(["Sped up kernels 3x"], mult, number_corpus=mult).unsupported_numbers
    print("test_number_must_mean_the_same_thing: PASS")


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
    test_job_description_cannot_ground_a_number()
    test_numbers_use_token_boundaries_not_substring()
    test_comma_formatted_numbers_are_grounded()
    test_number_must_mean_the_same_thing()
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


def test_rewrite_that_adds_scope_is_flagged():
    """A real case: grounding passed, yet the judge turned 'used daily'
    into 'self-serve'. The claim must appear in that entry's own sources."""
    import types
    from src.quality import verify_claims
    master = {"experience": [{"id": "acme", "company": "Acme Corp",
                              "bullets": ["Shipped an admin dashboard non-technical staff used daily"]}],
              "projects": [{"name": "Matrix Engine: C", "bullets": ["Improved cache locality"]}]}
    pkg = types.SimpleNamespace(
        experience_bullets={"acme": ["Shipped a self-serve admin dashboard that improved staff workflows",
                                     "Shipped an admin dashboard non-technical staff used daily"]},
        projects=[{"name": "Matrix Engine", "bullets": ["Improved cache locality with tiling"]}],
        summary="", cover_letter="Your team owns the platform. I led the dashboard work.")
    found = verify_claims(pkg, master)
    assert any('"self-serve"' in f and f.startswith("Acme") for f in found), found
    # "Improved" in the Matrix project doesn't vouch for one at Acme...
    assert any('"improved"' in f and f.startswith("Acme") for f in found), found
    # ...but does vouch for the Matrix bullet itself.
    assert not any(f.startswith("Matrix") for f in found), found
    # Letters: only sentences about the candidate make claims.
    assert any('"led"' in f for f in found) and not any('"owns"' in f for f in found), found
