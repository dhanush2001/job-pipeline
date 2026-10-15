"""
Tests for the GitHub-projects feature.

The repo-list parser is tested against a fixture captured from the documented
GitHub API response shape. NOTE: the live API call was rate-limited during
development (unauthenticated = 60 req/hour per IP), so this fixture is
structural, not captured from a verified live response -- treat the parser as
unconfirmed against real data until your first authenticated run.

The merge/render path IS fully verified: the tailored projects genuinely
reach the compiled PDF.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.render_resume import merge_tailored_content, render_resume
from src.tailor_resume import (
    MAX_PROJECTS,
    TailoredContent,
    _build_user_prompt,
    _normalize_projects,
)

FIXTURE = Path(__file__).parent / "fixtures" / "github_repos_response.json"


def test_repo_parsing_from_fixture():
    from src.github_projects import Repo

    payload = json.loads(FIXTURE.read_text())
    repos = []
    for item in payload:
        if item.get("fork"):
            continue
        if item.get("archived"):
            continue
        repos.append(
            Repo(
                name=item.get("name", ""),
                description=item.get("description") or "",
                language=item.get("language") or "",
                topics=item.get("topics", []) or [],
                stars=item.get("stargazers_count", 0),
                pushed_at=(item.get("pushed_at") or "")[:10],
                url=item.get("html_url", ""),
            )
        )

    # 4 in fixture; 1 fork and 1 archived must be dropped.
    assert len(repos) == 2, f"Expected 2 kept repos, got {len(repos)}"
    names = {r.name for r in repos}
    assert "forked-thing" not in names, "Forks should be excluded"
    assert "old-archived" not in names, "Archived repos should be excluded"

    repo = next(r for r in repos if r.name == "repo-radar")
    assert repo.language == "Python"
    assert repo.pushed_at == "2026-08-01", f"Date should be truncated to YYYY-MM-DD, got {repo.pushed_at}"
    assert "agents" in repo.topics

    # None description must become "" not the string "None"
    thin = next(r for r in repos if r.name == "thin-repo")
    assert thin.description == "", "Null description should normalize to empty string"

    print("test_repo_parsing_from_fixture: PASS")


def test_repos_reach_the_prompt():
    master = yaml.safe_load((Path(__file__).parent.parent / "data" / "master_resume.example.yaml").read_text())
    repos = [{"name": "repo-radar", "description": "Agentic GitHub monitor", "language": "Python"}]

    with_repos = _build_user_prompt(master, "SWE", "Acme", "job desc", repos)
    without = _build_user_prompt(master, "SWE", "Acme", "job desc", None)

    assert "github_repos" in with_repos
    assert "repo-radar" in with_repos
    assert "github_repos" not in without, "Key should be omitted entirely when no repos"

    print("test_repos_reach_the_prompt: PASS")


def test_tailored_projects_reach_the_pdf():
    """The real check: do role-specific projects actually land in the PDF?"""
    master = yaml.safe_load((Path(__file__).parent.parent / "data" / "master_resume.example.yaml").read_text())

    tailored = TailoredContent(
        summary="Tailored summary.",
        skills=["Python", "PyTorch"],
        experience_bullets={},
        projects=[
            {"name": "repo-radar", "bullets": ["Agentic system monitoring a research repo with a daily digest"]},
            {"name": "job-pipeline", "bullets": ["Scans ATS boards and tailors resumes per posting"]},
        ],
    )

    merged = merge_tailored_content(master, tailored)

    # Tailored projects must REPLACE the master list, not append to it.
    assert len(merged["projects"]) == 2
    assert merged["projects"][0]["name"] == "repo-radar"
    master_only = {p["name"] for p in master["projects"]}
    assert not (master_only & {p["name"] for p in merged["projects"]}), \
        "Master projects should have been replaced by tailored ones"

    out = Path(__file__).parent.parent / "output" / "resumes" / "_projects_test.pdf"
    assert render_resume(merged, "templates/resume.tex.j2", str(out)), "Render failed"

    # Confirm the tailored project text is actually in the compiled PDF.
    import subprocess
    text = subprocess.run(["pdftotext", str(out), "-"], capture_output=True, text=True).stdout
    assert "repo-radar" in text, "Tailored project name missing from PDF"
    assert "daily digest" in text, "Tailored project bullet missing from PDF"
    assert "Job Application Pipeline" not in text, "Master project leaked into tailored PDF"

    out.unlink(missing_ok=True)
    print("test_tailored_projects_reach_the_pdf: PASS")


def test_tailored_projects_keep_source_metadata():
    """
    The model returns only name + bullets. Dates and the tech stack must be
    re-attached from the grounded source (master projects or GitHub repos) so
    the tailored resume keeps the layout the master had -- otherwise the
    template's date/technologies fields render blank.
    """
    master = {
        "projects": [
            {
                "name": "LLaVA-Med: Medical Data Platform",
                "dates": "March 2026 -- Present",
                "technologies": "Python, FastAPI, PyTorch",
                "bullets": ["original bullet"],
            }
        ]
    }
    github_repos = [
        {"name": "repo-radar", "language": "Go", "last_pushed": "2026-08-01"}
    ]
    tailored = TailoredContent(
        summary="s",
        skills=[],
        experience_bullets={},
        projects=[
            # Name is trimmed vs the master -- fuzzy match must still find it.
            {"name": "LLaVA-Med", "bullets": ["tailored bullet"]},
            {"name": "repo-radar", "bullets": ["tailored repo bullet"]},
        ],
    )

    merged = merge_tailored_content(master, tailored, github_repos)
    by_name = {p["name"]: p for p in merged["projects"]}

    llava = by_name["LLaVA-Med"]
    assert llava["dates"] == "March 2026 -- Present", "Master project dates lost"
    assert llava["technologies"] == "Python, FastAPI, PyTorch", "Master tech stack lost"
    assert llava["bullets"] == ["tailored bullet"], "Tailored bullets must survive"

    radar = by_name["repo-radar"]
    assert radar["dates"] == "2026-08-01", "Repo last-pushed date not carried over"
    assert radar["technologies"] == "Go", "Repo language not carried over as tech stack"

    print("test_tailored_projects_keep_source_metadata: PASS")


def test_projects_capped():
    """Project count is enforced in code (one-page limit), not just requested."""
    raw = [{"name": f"proj-{i}", "bullets": ["b"]} for i in range(9)]
    kept = _normalize_projects(raw)
    assert len(kept) == MAX_PROJECTS == 3, f"Expected {MAX_PROJECTS}, got {len(kept)}"
    # Relevance ordering (the model's) must be preserved -- top ones kept.
    assert [p["name"] for p in kept] == ["proj-0", "proj-1", "proj-2"]
    # Unnamed entries are dropped rather than rendered as blank projects.
    assert _normalize_projects([{"bullets": ["b"]}, {"name": "keep", "bullets": []}]) == \
        [{"name": "keep", "bullets": []}]
    print("test_projects_capped: PASS")


def test_reconcile_skills_keeps_grouping():
    """A flat model skills list must NOT flatten grouped master skills (that
    printed a blank line per skill and overflowed the page). Groups are kept;
    entries are reordered within each group by the tailored priority."""
    from src.render_resume import _reconcile_skills
    master = [
        {"category": "Languages", "entries": "Python, SQL, JavaScript, Java"},
        {"category": "Backend & APIs", "entries": "Flask, FastAPI, REST APIs"},
    ]
    out = _reconcile_skills(master, ["FastAPI", "Python", "REST APIs"])
    assert isinstance(out[0], dict) and "category" in out[0], "grouping was lost"
    # tailored-first ordering within each group, all originals kept
    assert out[0]["entries"] == "Python, SQL, JavaScript, Java"  # Python moved first
    assert out[1]["entries"] == "FastAPI, REST APIs, Flask"      # tailored first, Flask kept
    # flat master stays flat
    assert _reconcile_skills(["Python", "AWS"], ["AWS", "React"]) == ["AWS", "React"]
    print("test_reconcile_skills_keeps_grouping: PASS")


def test_summary_not_injected_when_master_empty():
    """If the master leaves summary empty (to stay one page), a generated summary
    must NOT be added to the resume -- it only feeds the cover letter."""
    tailored = TailoredContent(summary="A generated summary.", skills=[],
                               experience_bullets={}, projects=[])
    # master with no summary section -> stays empty
    merged = merge_tailored_content({"summary": ""}, tailored)
    assert merged["summary"] == "", "generated summary leaked onto the resume"
    # master that opted into a summary -> gets the tailored one
    merged2 = merge_tailored_content({"summary": "old"}, tailored)
    assert merged2["summary"] == "A generated summary."
    # include_summary=True forces it on even when master is empty
    merged3 = merge_tailored_content({"summary": ""}, tailored, include_summary=True)
    assert merged3["summary"] == "A generated summary."
    print("test_summary_not_injected_when_master_empty: PASS")


def test_one_page_fitter_helpers():
    from src.render_resume import _pages_from_log, _one_page_variants
    assert _pages_from_log("Output written on resume.pdf (2 pages, 123 bytes).") == 2
    assert _pages_from_log("Output written on resume.pdf (1 page, 99 bytes).") == 1
    assert _pages_from_log("no page info") is None

    data = {"summary": "s",
            "experience": [{"bullets": ["a", "b", "c", "d"]}],
            "projects": [{"bullets": ["w", "x", "y", "z"]}, {"bullets": ["1"]},
                         {"bullets": ["2"]}, {"bullets": ["3"]}]}
    v = list(_one_page_variants(data))
    assert v[0] is data                                   # first is untouched
    assert v[1]["summary"] == ""                          # summary dropped next
    assert len(v[3]["projects"]) == 2                     # projects capped
    assert all(len(p["bullets"]) <= 2 for p in v[3]["projects"])  # bullets capped
    assert data["summary"] == "s"                         # original not mutated
    print("test_one_page_fitter_helpers: PASS")


def test_empty_projects_falls_back_to_master():
    master = yaml.safe_load((Path(__file__).parent.parent / "data" / "master_resume.example.yaml").read_text())
    tailored = TailoredContent(summary="s", skills=[], experience_bullets={}, projects=[])
    merged = merge_tailored_content(master, tailored)
    assert merged["projects"] == master["projects"], \
        "With no tailored projects, master projects must survive"
    print("test_empty_projects_falls_back_to_master: PASS")


if __name__ == "__main__":
    test_repo_parsing_from_fixture()
    test_repos_reach_the_prompt()
    test_tailored_projects_reach_the_pdf()
    test_tailored_projects_keep_source_metadata()
    test_projects_capped()
    test_empty_projects_falls_back_to_master()
    print("\nAll GitHub-projects tests passed.")
