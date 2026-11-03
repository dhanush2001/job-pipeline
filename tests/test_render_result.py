"""
Tests for render_resume's RenderResult: it must report when the one-page
constraint could not be met (not silently succeed), and return the content it
ACTUALLY rendered (possibly trimmed) so the caller scores the PDF, not the
pre-trim input. The LaTeX compiler is mocked, so these need no latexmk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.render_resume as rr_mod
from src.render_resume import render_resume

TEMPLATE = "templates/resume.tex.j2"
OUT = "/tmp/_render_result_test.pdf"

RESUME = {
    "name": "T. Candidate", "location": "Anytown", "phone": "1", "email": "e@x.com",
    "linkedin_handle": "h", "github_handle": "g",
    "summary": "A summary line that the one-page trimmer drops first.",
    "skills": [{"category": "Languages", "entries": "Python, SQL"}],
    "experience": [{"company": "C", "title": "Engineer", "dates": "2025",
                    "technologies": "Python", "location": "X",
                    "bullets": ["b1", "b2", "b3", "b4", "b5"]}],
    "projects": [{"name": "P", "dates": "2025", "technologies": "Python",
                  "bullets": ["p1", "p2", "p3", "p4"]}],
}


def _fake_compile(sequence):
    """Return (ok, pages) from `sequence` by call index; repeats the last."""
    calls = {"i": 0}

    def fake(tex_source, output_pdf_path):
        i = calls["i"]
        calls["i"] += 1
        return sequence[min(i, len(sequence) - 1)]
    return fake


def _with_compile(sequence, **kwargs):
    orig = rr_mod._compile_tex
    rr_mod._compile_tex = _fake_compile(sequence)
    try:
        return render_resume(RESUME, TEMPLATE, OUT, **kwargs)
    finally:
        rr_mod._compile_tex = orig


def test_one_page_failure_is_reported():
    """Every trimmed variant still 2 pages -> PDF is written but one_page_ok False."""
    res = _with_compile([(True, 2)], fit_one_page=True)
    assert res.ok is True
    assert res.pages == 2
    assert res.one_page_ok is False, "should report the one-page constraint was not met"
    print("test_one_page_failure_is_reported: PASS")


def test_returns_the_trimmed_variant_that_fit():
    """Full = 2 pages, first trimmed variant = 1 page -> result carries the
    TRIMMED content (fewer bullets, summary kept), which is what should be scored."""
    res = _with_compile([(True, 2), (True, 1)], fit_one_page=True)
    assert res.ok is True and res.pages == 1 and res.one_page_ok is True
    assert res.data is not RESUME, "should return the trimmed variant, not the input"
    assert res.data.get("summary") == RESUME["summary"], "the first trim keeps the summary"
    assert len(res.data["experience"][0]["bullets"]) == 3, "...and trims bullets instead"
    print("test_returns_the_trimmed_variant_that_fit: PASS")


def test_no_constraint_returns_input_unchanged():
    """fit_one_page=False: a 2-page render is a success with no constraint, and
    the returned data is the original input."""
    res = _with_compile([(True, 2)], fit_one_page=False)
    assert res.ok is True and res.one_page_ok is True
    assert res.data is RESUME
    print("test_no_constraint_returns_input_unchanged: PASS")


def test_compile_failure_is_falsy():
    res = _with_compile([(False, None)], fit_one_page=True)
    assert not res and res.ok is False
    print("test_compile_failure_is_falsy: PASS")


if __name__ == "__main__":
    test_one_page_failure_is_reported()
    test_returns_the_trimmed_variant_that_fit()
    test_no_constraint_returns_input_unchanged()
    test_compile_failure_is_falsy()
    print("\nAll render-result tests passed.")
