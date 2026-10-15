"""
Tests for the handoff delivery + paste-proof fill-in worklist flow.

Covers: handoff dir resolution, the worklist writer/parser round-trip, that a
raw pasted description (colons, blank lines, bullets) parses cleanly, the
"is this filled in?" gate, and a real end-to-end dry-run of tailor_worklist
that renders a PDF (LaTeX required).
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import handoff
from src.main import _write_needs_description
from src.scan_jobs import Job
import src.tailor_worklist as wl


def _job(company, title, desc):
    return Job(id=f"x:{company}:{title}", source="ashby", company=company, title=title,
               location="", url="u", apply_url="a", description=desc)


def test_handoff_resolve_and_deliver():
    with tempfile.TemporaryDirectory() as tmp:
        dirs = handoff.resolve_dirs({}, override=tmp)
        assert dirs == [Path(tmp) / "job-applications"]
        assert handoff.resolve_dirs({"output": {"handoff_dirs": []}}) == []  # explicitly disabled
        handoff.resolve_dirs({})  # missing key -> defaults; just don't crash

        src_pdf = Path(tmp) / "r.pdf"; src_pdf.write_text("PDF")
        src_cover = Path(tmp) / "c.pdf"; src_cover.write_text("PDF")
        src_cover.with_suffix(".txt").write_text("cover text")
        out = handoff.deliver("Acme_SWE", src_pdf, src_cover, dirs,
                              extra_texts={"missing_skills.md": "gaps"})
        job_dir = dirs[0] / "Acme_SWE"      # each job gets its own folder
        assert (job_dir / "resume.pdf").exists()
        assert (job_dir / "cover_letter.pdf").exists()
        assert (job_dir / "cover_letter.txt").exists()
        assert (job_dir / "missing_skills.md").read_text() == "gaps"
        assert out == [job_dir / "resume.pdf"]
    print("test_handoff_resolve_and_deliver: PASS")


def test_worklist_writer_parser_roundtrip_and_gate():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "needs.txt"
        needs = [(_job("Acme", "SWE", ""), 0.5), (_job("Beta", "MLE", ""), 0.4)]
        assert _write_needs_description(needs, str(path), []) is True

        text = path.read_text()
        assert "### JOB 1" in text and "### JOB 2" in text
        assert "Company: Acme" in text and "Title: MLE" in text
        assert handoff.WORKLIST_PASTE_MARKER in text and handoff.WORKLIST_END_MARKER in text

        parsed = handoff.parse_worklist(text)
        assert len(parsed) == 2
        assert parsed[0]["company"] == "Acme" and parsed[1]["title"] == "MLE"
        assert all(wl._needs_filling(p["description"]) for p in parsed)  # all still placeholders

        assert wl._needs_filling(handoff.NEEDS_DESC_PLACEHOLDER) is True
        assert wl._needs_filling("") is True
        assert wl._needs_filling("too short") is True
        assert wl._needs_filling("A real, sufficiently long job description pasted by the user.") is False

        assert _write_needs_description([], str(Path(tmp) / "none.txt"), []) is False
    print("test_worklist_writer_parser_roundtrip_and_gate: PASS")


def test_parse_worklist_handles_raw_paste():
    """A raw job description -- colons, blank lines, bullets -- must parse cleanly."""
    text = handoff.render_worklist([{"company": "Acme", "title": "SWE: Payments", "url": "https://x"}])
    raw_jd = "About the role:\n\n- Build APIs\n- 5+ years? no: new grads welcome\nSalary: $100k"
    text = text.replace(handoff.NEEDS_DESC_PLACEHOLDER, raw_jd)

    parsed = handoff.parse_worklist(text)
    assert len(parsed) == 1
    j = parsed[0]
    assert j["company"] == "Acme"
    assert j["title"] == "SWE: Payments"   # colon in the title survives (split on first ':')
    assert j["url"] == "https://x"
    assert "Build APIs" in j["description"] and "Salary: $100k" in j["description"]
    assert not wl._needs_filling(j["description"])
    print("test_parse_worklist_handles_raw_paste: PASS")


def test_worklist_end_to_end_dry_run():
    """One filled entry + one placeholder -> exactly one PDF into the handoff dir."""
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        text = handoff.render_worklist([
            {"company": "Acme", "title": "New Grad SWE", "url": "https://x"},
            {"company": "Beta", "title": "MLE", "url": "https://y"},
        ])
        # Fill ONLY job 1 (replace first placeholder); leave job 2 as-is.
        text = text.replace(
            handoff.NEEDS_DESC_PLACEHOLDER,
            "We are hiring a new grad software engineer. Python, backend, distributed systems, testing.",
            1,
        )
        wl_path = Path(tmp) / "needs_description.txt"
        wl_path.write_text(text)

        args = argparse.Namespace(
            worklist=str(wl_path), config=str(repo / "config.yaml"), handoff=tmp,
            dry_run=True, new=None, out=None)
        assert wl.run(args) == 0

        job_dirs = [p for p in (Path(tmp) / "job-applications").iterdir() if p.is_dir()]
        names = {p.name for p in job_dirs}
        assert any(n.startswith("Acme__") for n in names), names   # Acme tailored -> its own folder
        assert not any(n.startswith("Beta") for n in names), names  # placeholder skipped
        acme = next(p for p in job_dirs if p.name.startswith("Acme__"))
        assert (acme / "resume.pdf").exists()
        assert (acme / "missing_skills.md").exists()
    print("test_worklist_end_to_end_dry_run: PASS")


def test_parse_pasted_jobs_simple_format():
    """Plain file: whole postings separated by ===, optional labels, comments ignored."""
    text = (
        "# paste postings below\n"
        "Senior nonsense line is fine\n"       # part of block 1 description, not a label
        "\n"
        "About the role: build backend APIs in Python.\n"
        "Requirements: 1+ years, testing, SQL.\n"
        "===\n"
        "Company: Beta Corp\n"
        "Title: ML Engineer\n"
        "URL: https://beta.example/job\n"
        "You will train models with PyTorch and deploy them.\n"
        "---\n"                                  # dashes also separate
        "<paste job posting 3 here>\n"           # unfilled placeholder slot
    )
    jobs = handoff.parse_pasted_jobs(text)
    assert len(jobs) == 3, jobs
    # Block 1: no labels -> company/title blank, whole thing is the description.
    assert jobs[0]["company"] == "" and jobs[0]["title"] == ""
    assert "backend APIs" in jobs[0]["description"]
    assert "nonsense line" in jobs[0]["description"]   # non-label lines are kept
    # Block 2: labels peeled off, description is the remainder.
    assert jobs[1]["company"] == "Beta Corp"
    assert jobs[1]["title"] == "ML Engineer"
    assert jobs[1]["url"] == "https://beta.example/job"
    assert jobs[1]["description"].startswith("You will train models")
    assert "Company:" not in jobs[1]["description"]
    # Block 3 is an unfilled placeholder -> caught by the fill gate.
    assert wl._needs_filling(jobs[2]["description"])
    assert handoff.looks_like_worklist(text) is False
    print("test_parse_pasted_jobs_simple_format: PASS")


def test_paste_flow_end_to_end_dry_run():
    """Paste two full postings (no labels) -> two folders, titles from the text."""
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        text = (
            "Software Engineer, New Grad\n"
            "Build backend services in Python and Go with PostgreSQL and Docker. "
            "New grads welcome; CI/CD and testing experience a plus.\n"
            "===\n"
            "Data Analyst\n"
            "Analyze usage data with SQL and Python, build dashboards, run A/B tests. "
            "Familiarity with pandas and statistics required.\n"
        )
        paste_path = Path(tmp) / "jobs.txt"
        paste_path.write_text(text)

        args = argparse.Namespace(
            worklist=str(paste_path), config=str(repo / "config.yaml"), handoff=tmp,
            dry_run=True, new=None, out=None)
        assert wl.run(args) == 0

        names = {p.name for p in (Path(tmp) / "job-applications").iterdir() if p.is_dir()}
        # Folder is <company>__<title>; heuristic title = first non-empty line.
        assert any("Software_Engineer_New_Grad" in n for n in names), names
        assert any("Data_Analyst" in n for n in names), names
    print("test_paste_flow_end_to_end_dry_run: PASS")


def test_new_template_generation():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub" / "jobs.txt"   # parent created on demand
        args = argparse.Namespace(worklist=None, new=3, out=str(out),
                                  config="config.yaml", handoff=None, dry_run=True)
        assert wl.run(args) == 0
        body = out.read_text()
        sep_lines = [ln for ln in body.splitlines() if ln.strip() == handoff.PASTE_SEPARATOR]
        assert len(sep_lines) == 2   # 3 slots -> 2 separator lines
        assert handoff.looks_like_worklist(body) is False
        # Refuses to clobber an existing file.
        assert wl.run(args) == 1
    print("test_new_template_generation: PASS")


if __name__ == "__main__":
    test_handoff_resolve_and_deliver()
    test_worklist_writer_parser_roundtrip_and_gate()
    test_parse_worklist_handles_raw_paste()
    test_parse_pasted_jobs_simple_format()
    test_new_template_generation()
    test_worklist_end_to_end_dry_run()
    test_paste_flow_end_to_end_dry_run()
    print("\nAll handoff/worklist tests passed.")
