"""
Tests for the --dry-run stub generator.

The stub stands in for a DeepSeek call with no API key. It must (a) draw only on
the master resume + the posting's title/company so it never fabricates a metric,
and (b) carry the master's experience/projects so the rendered PDF is real.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile

import yaml

from src.main import load_seen, save_seen
from src.quality import build_source_corpus, verify_grounding
from src.tailor_resume import MAX_PROJECTS, stub_application_package

MASTER = Path(__file__).parent.parent / "data" / "master_resume.example.yaml"


def test_stub_is_grounded_no_invented_numbers():
    master = yaml.safe_load(MASTER.read_text())
    pkg = stub_application_package(master, "New Grad Software Engineer", "acme")

    corpus = build_source_corpus(master, None, "Hiring a New Grad Software Engineer.", "")
    report = verify_grounding(pkg.groundable_texts(), corpus, "acme")
    assert not report.unsupported_numbers, \
        f"Stub must never invent numbers: {report.unsupported_numbers}"
    print("test_stub_is_grounded_no_invented_numbers: PASS")


def test_stub_carries_master_content():
    master = yaml.safe_load(MASTER.read_text())
    pkg = stub_application_package(master, "SWE", "acme")

    # Experience bullets keyed by the master's stable ids.
    master_ids = {e["id"] for e in master.get("experience", []) if e.get("id")}
    assert set(pkg.experience_bullets) == master_ids, "Experience ids must match master"

    # Projects come from master, capped, skills left empty (merge keeps master's).
    assert 0 < len(pkg.projects) <= MAX_PROJECTS
    assert pkg.skills == [], "Stub leaves skills empty so master's grouped skills survive"
    assert pkg.cover_letter and pkg.referral_note and pkg.referral_dm, "Drafts must be populated"
    assert "SWE" in pkg.cover_letter and "acme" in pkg.cover_letter
    print("test_stub_carries_master_content: PASS")


def test_seen_log_survives_empty_or_corrupt():
    """An empty or half-written seen-log must NOT crash the run (it used to):
    load_seen returns an empty set, and save_seen writes atomically + round-trips."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seen.json"
        assert load_seen(str(p)) == set()          # missing file
        p.write_text("")                            # empty (interrupted save)
        assert load_seen(str(p)) == set()
        p.write_text("{ not json")                  # corrupt
        assert load_seen(str(p)) == set()
        save_seen(str(p), {"greenhouse:acme:1", "lever:beta:2"})
        assert load_seen(str(p)) == {"greenhouse:acme:1", "lever:beta:2"}
    print("test_seen_log_survives_empty_or_corrupt: PASS")


if __name__ == "__main__":
    test_stub_is_grounded_no_invented_numbers()
    test_stub_carries_master_content()
    test_seen_log_survives_empty_or_corrupt()
    print("\nAll dry-run tests passed.")
