"""Tests for country detection + targeting (src/locations.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.locations import country_of, is_targeted, REMOTE, OTHER

TARGETS = ["United States", "Canada", "United Kingdom", "Germany", "Singapore",
           "Hong Kong", "Taiwan", "France", "Ireland", "Switzerland", "Spain"]


def test_country_detection():
    cases = {
        # US: cities, state codes, bare US, remote-US -- and the Cambridge/Oxford trap
        "New York, NY": "United States", "San Francisco, CA": "United States",
        "Austin, TX": "United States", "Cambridge, MA": "United States",
        "Columbus, OH": "United States", "Remote - US": "United States",
        "Remote, United States": "United States",
        "Toronto, ON, Canada": "Canada", "Vancouver, BC": "Canada",
        "London, UK": "United Kingdom", "Manchester, England": "United Kingdom",
        "Berlin, Germany": "Germany", "München, Germany": "Germany",
        "Amsterdam, Netherlands": "Netherlands", "Sydney, Australia": "Australia",
        # newly recognized Asia + Europe hubs
        "Singapore": "Singapore", "Hong Kong": "Hong Kong", "Taipei, Taiwan": "Taiwan",
        "Paris, France": "France", "Dublin, Ireland": "Ireland",
        "Zurich, Switzerland": "Switzerland", "Stockholm, Sweden": "Sweden",
        "Madrid, Spain": "Spain", "Milan, Italy": "Italy", "Lisbon, Portugal": "Portugal",
        # still foreign / not pursued -> Other (dropped)
        "Bangalore, India": OTHER, "Tokyo, Japan": OTHER, "Dubai, UAE": OTHER,
        # no country -> Remote
        "Remote": REMOTE, "": REMOTE,
    }
    bad = {loc: (country_of(loc), exp) for loc, exp in cases.items() if country_of(loc) != exp}
    assert not bad, f"country_of mismatches: {bad}"
    print(f"test_country_detection: PASS ({len(cases)} cases)")


def test_is_targeted_keeps_targets_and_remote_drops_foreign():
    assert is_targeted("United States", TARGETS)
    assert is_targeted("Singapore", TARGETS)
    assert is_targeted("France", TARGETS)        # now a target
    assert is_targeted(REMOTE, TARGETS)          # country-agnostic remote kept
    assert not is_targeted(OTHER, TARGETS)       # foreign dropped
    assert not is_targeted("Japan", TARGETS)
    assert is_targeted("Japan", [])              # empty targets = keep everything
    print("test_is_targeted_keeps_targets_and_remote_drops_foreign: PASS")


if __name__ == "__main__":
    test_country_detection()
    test_is_targeted_keeps_targets_and_remote_drops_foreign()
    print("\nAll location tests passed.")
