"""
Tests for the post-AI guards, using the real failure cases found in the
Oct 8 run (skills credited to the wrong job, a summary repeating the phrase the
judge flagged, an in-progress project presented as finished). Offline.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guards import (apply_judge, flagged_phrases, guard_experience, guard_tailored,
                        overclaimed_in_progress, repair_summary)

MASTER = {
    "skills": [{"category": "Languages", "entries": "Python, C/C++, Java, TypeScript"}],
    "experience": [{"id": "acme", "company": "Acme", "title": "Software Development Intern",
                    "technologies": "Python, Java, TypeScript, React, REST APIs",
                    "bullets": ["Integrated a React frontend with a Java Servlet backend via REST APIs",
                                "Built a user-facing admin dashboard for a 12-person Admin Controls team",
                                "Participated in code reviews with senior engineers"]}],
    "projects": [
        {"name": "tiny-serve: LLM Inference Server", "dates": "August 2026 -- Present",
         "technologies": "Python, CUDA, C++",
         "bullets": ["Building an LLM inference server with a KV cache and paged attention"]},
        {"name": "Security Benchmark", "dates": "2026", "technologies": "Python",
         "bullets": ["Measured how LLMs approve insecure code; security evaluation harness"]},
    ],
}


def test_skill_from_a_project_is_not_credited_to_the_internship():
    judged = {"acme": [
        "Integrated a **React** frontend with a **Java Servlet** backend via **REST API**",   # fine (API ~ APIs)
        "Resolved issues in **code reviews**, strengthening **security** practices",         # security: project skill
        "Drove **optimization** of production-bound features through **code reviews**"]}  # optimization: invented
    pre = {"acme": ["Integrated a React frontend with a Java Servlet backend via REST APIs",
                    "Reviewed code with senior engineers", "Participated in code reviews and Git workflows"]}
    clean, notes = guard_experience(judged, MASTER, pre)
    out = clean["acme"]
    assert out[0] == judged["acme"][0]                      # untouched ("API" is implied by "REST APIs")
    assert out[1] == "Reviewed code with senior engineers"  # reverted to the clean pre-judge bullet
    joined = " ".join(out).lower()
    assert "security" not in joined and "optimization" not in joined and "git" not in joined
    assert len(notes) == 2 and "security" in notes[0] and "optimization" in notes[1]
    _assert_no_near_duplicates(out)
    print("test_skill_from_a_project_is_not_credited_to_the_internship: PASS")


def _assert_no_near_duplicates(bullets):
    from src.guards import _similar
    for i in range(len(bullets)):
        for j in range(i + 1, len(bullets)):
            assert _similar(bullets[i], bullets[j]) < 0.6, (bullets[i], bullets[j])


def test_replacement_never_duplicates_another_bullet():
    """The Nuro case: the judge already wrote a dashboard bullet; reverting a bad
    bullet to the original dashboard bullet would say the same thing twice."""
    judged = {"acme": [
        "Shipped a React admin dashboard to production for a 12-person Admin Controls team via REST APIs",
        "Built an end-to-end **data pipeline** connecting UI components to backend services",
    ]}
    pre = {"acme": ["Integrated a React frontend with a Java Servlet backend via REST APIs",
                    "Built a user-facing admin dashboard for a 12-person Admin Controls team"]}
    clean, notes = guard_experience(judged, MASTER, pre)
    out = clean["acme"]
    assert "data pipeline" not in " ".join(out).lower()
    _assert_no_near_duplicates(out)
    assert out[1] == "Integrated a React frontend with a Java Servlet backend via REST APIs", out
    print("test_replacement_never_duplicates_another_bullet: PASS")


def test_implied_skills_are_allowed():
    """TypeScript/React work implies JavaScript -- not a misplaced skill."""
    clean, notes = guard_experience({"acme": ["Built the **JavaScript** UI layer in React"]}, MASTER)
    assert notes == [] and clean["acme"] == ["Built the **JavaScript** UI layer in React"]
    print("test_implied_skills_are_allowed: PASS")


def test_flagged_phrase_is_removed_from_summary():
    flags = ["The summary leans on generic phrases like 'fast-moving, loosely-defined environments' "
             "rather than concrete qualifications."]
    phrases = flagged_phrases(flags)
    assert phrases == ["fast-moving, loosely-defined environments"]
    good = "MS CS candidate who ships React frontends and FastAPI services backed by PostgreSQL to production users."
    s, notes = repair_summary(good + " Ships features in fast-moving, loosely-defined environments.", phrases)
    assert s == good and notes
    # The Baseten case: dropping would leave filler -> keep the original and report instead.
    s_b, n_b = repair_summary("MS CS candidate with LLM serving and FastAPI work. Focused on performance.",
                              ["ms cs candidate"])
    assert s_b.startswith("MS CS candidate") and "left as-is" in n_b[0]
    # If every sentence is flagged, keep it rather than blank the summary -- and report.
    s2, notes2 = repair_summary("Thrives in fast-moving, loosely-defined environments.", phrases)
    assert s2.startswith("Thrives") and "left as-is" in notes2[0]
    print("test_flagged_phrase_is_removed_from_summary: PASS")


def test_summary_split_keeps_degree_abbreviations_intact():
    """Pinterest: splitting at "M.S." shipped "New-grad software engineer (M.S. Built ..."."""
    from src.guards import _sentences
    judged = ("New-grad software engineer (M.S. CS, December 2026) who shipped a production React/TypeScript "
              "admin dashboard on a Java REST backend and uses AI coding tools daily to write, debug, and "
              "document code. Built a Python LLM evaluation harness across 15,000 automated runs, "
              "quantifying a 12.7-point robustness gap in AI code review.")
    assert len(_sentences(judged)) == 2
    assert len(_sentences("Holds a Ph.D. in CS. Ships e.g. React apps. Done.")) == 3
    flags = ["The experience section is thin, which makes the 'shipped a production ...' framing feel thin.",
             "Summary opens with a credential ('New-grad software engineer (M.S. CS, December 2026)') instead "
             "of an achievement, and 'uses AI coding tools daily' is asserted without a shipped result attached."]
    phrases = flagged_phrases(flags)
    assert "shipped a production" in phrases and "uses ai coding tools daily" in phrases
    s, notes = repair_summary(judged, phrases)
    assert s.startswith("Built a Python LLM evaluation harness") and "(M.S." not in s and notes
    # With a repair available, the full summary is rewritten instead of losing the Acme sentence.
    fixed = ("Delivered a React and TypeScript admin dashboard on Java REST APIs at Acme, and built a Python "
             "LLM evaluation harness across 15,000 automated runs measuring a 12.7-point gap in AI code review.")
    pkg = SimpleNamespace(experience_bullets={}, summary="old")
    apply_judge(pkg, {"summary": judged, "red_flags": flags}, MASTER, lambda summ, names, fl: fixed)
    assert pkg.summary == fixed
    print("test_summary_split_keeps_degree_abbreviations_intact: PASS")


def test_in_progress_project_presented_as_finished_is_flagged():
    done = "MS CS candidate who builds and benchmarks LLM inference systems with a KV cache and paged attention."
    ongoing = "MS CS candidate building an LLM inference server with a KV cache. Strong in Python."
    assert overclaimed_in_progress(done, MASTER), "should flag"
    assert overclaimed_in_progress(ongoing, MASTER) == [], "progressive tense is fine"
    # The Haiku miss: "architected ... with paged attention" is finished-work language.
    assert overclaimed_in_progress("Performance engineer who architected batching layers with paged attention.", MASTER)
    print("test_in_progress_project_presented_as_finished_is_flagged: PASS")


def test_apply_judge_end_to_end_on_a_package():
    pkg = SimpleNamespace(experience_bullets={"acme": ["Integrated a React frontend via REST APIs"]},
                          summary="old")
    assert guard_tailored(pkg, MASTER) == []
    review = {"experience": {"acme": ["Hardened **security** of the React frontend"]},
              "summary": "Ships React frontends and Java Servlet backends over REST APIs to production users every single day. Thrives in fast-moving, loosely-defined environments.",
              "red_flags": ["generic: 'fast-moving, loosely-defined environments'"]}
    notes = apply_judge(pkg, review, MASTER)
    assert pkg.experience_bullets["acme"] == ["Integrated a React frontend via REST APIs"]
    assert pkg.summary == "Ships React frontends and Java Servlet backends over REST APIs to production users every single day."
    assert len(notes) == 2
    print("test_apply_judge_end_to_end_on_a_package: PASS")


def test_overclaim_is_fixed_not_just_flagged():
    from src.guards import fix_overclaim
    done = "Builds and benchmarks LLM inference systems with a KV cache and paged attention."
    ongoing = ("Building an LLM inference server with a KV cache, backed by strong Python and "
               "C++ systems work and React frontends shipped to production.")
    rest = "Strong Python and C++ engineer who ships FastAPI services and React frontends to production daily."
    # 1. Pre-judge summary passes every check -> use it.
    s, n = fix_overclaim(done + " " + rest, ongoing, MASTER)
    assert s == ongoing and n
    # 2. Repair call result passes -> use it (pre-judge also overclaims).
    s2, n2 = fix_overclaim(done + " " + rest, done, MASTER, [], lambda summ, names, fl: ongoing)
    assert s2 == ongoing and "progressive" in n2[0]
    # 3. No repair -> drop the offending sentence when a real summary remains.
    s3, _ = fix_overclaim(done + " " + rest, done, MASTER)
    assert s3 == rest
    # Filler is never accepted: a short, skill-less remainder keeps the original + report.
    s4, n4 = fix_overclaim(done + " Comfortable owning systems end to end.", done, MASTER,
                           [], lambda summ, names, fl: "Comfortable owning systems.")
    assert s4.startswith("Builds") and "reads as done" in n4[0]
    # A repair that still overclaims, or repeats a flagged phrase, is rejected.
    s5, _ = fix_overclaim(done + " " + rest, done, MASTER, ["shipped to production"],
                          lambda summ, names, fl: ongoing)
    assert s5 == rest, s5
    # "benchmarks" as a noun is not a claim of finished work.
    assert overclaimed_in_progress("Experience building LLM inference servers and C++ benchmarks.", MASTER) == []
    print("test_overclaim_is_fixed_not_just_flagged: PASS")


if __name__ == "__main__":
    test_skill_from_a_project_is_not_credited_to_the_internship()
    test_replacement_never_duplicates_another_bullet()
    test_implied_skills_are_allowed()
    test_flagged_phrase_is_removed_from_summary()
    test_in_progress_project_presented_as_finished_is_flagged()
    test_apply_judge_end_to_end_on_a_package()
    test_overclaim_is_fixed_not_just_flagged()
    print("\nAll guard tests passed.")
