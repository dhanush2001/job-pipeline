"""
Tests for the scale-up features: cover letters, LinkedIn referral drafts, and
the prompt-cache key ordering that keeps cost sane at 100 jobs/run.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.render_resume import render_cover_letter
from src.tailor_resume import (
    LINKEDIN_NOTE_HARD_LIMIT,
    ApplicationPackage,
    _build_user_prompt,
    _enforce_note_limit,
)

MASTER = Path(__file__).parent.parent / "data" / "master_resume.example.yaml"


def test_note_under_limit_untouched():
    note = "Hi Priya, I just applied for the New Grad SWE role at Acme and would love to connect."
    out, truncated = _enforce_note_limit(note)
    assert out == note
    assert truncated is False
    print(f"test_note_under_limit_untouched: PASS ({len(out)} chars)")


def test_overlong_note_is_cut_to_limit():
    """LinkedIn BLOCKS the send past 300 chars, so an overlong note is useless."""
    note = ("Hello there, I am reaching out about the role. " * 12)
    assert len(note) > LINKEDIN_NOTE_HARD_LIMIT
    out, truncated = _enforce_note_limit(note)
    assert truncated is True
    assert len(out) <= LINKEDIN_NOTE_HARD_LIMIT, f"Still {len(out)} chars — would be blocked"
    print(f"test_overlong_note_is_cut_to_limit: PASS ({len(note)} -> {len(out)} chars)")


def test_truncation_prefers_sentence_boundary():
    note = "First sentence here. " + ("padding words to push past the cap " * 12)
    out, truncated = _enforce_note_limit(note)
    assert truncated is True
    assert len(out) <= LINKEDIN_NOTE_HARD_LIMIT
    assert out.endswith((".", "!", "?", "…")), f"Ended mid-word: {out[-30:]!r}"
    print("test_truncation_prefers_sentence_boundary: PASS")


def test_whitespace_collapsed_before_counting():
    """Newlines count toward LinkedIn's 300; collapse them so the count is honest."""
    note = "Hi   there,\n\n  I applied   for the role.\n"
    out, _ = _enforce_note_limit(note)
    assert "\n" not in out and "  " not in out
    print("test_whitespace_collapsed_before_counting: PASS")


def test_prompt_puts_variable_content_last_for_caching():
    """
    DeepSeek caches on shared prefixes. Stable content (resume, repos) must
    come BEFORE the per-job description, or every one of 100 calls is a full
    cache miss and the bill multiplies.
    """
    master = yaml.safe_load(MASTER.read_text())
    repos = [{"name": "repo-radar", "description": "monitor", "language": "Python"}]
    prompt = _build_user_prompt(master, "SWE", "Acme", "UNIQUE_JOB_DESCRIPTION", repos)

    i_master = prompt.index("master_resume")
    i_repos = prompt.index("github_repos")
    i_job = prompt.index("target_job")

    assert i_master < i_repos < i_job, (
        "Cache-killing key order: per-job content must come last "
        f"(master={i_master}, repos={i_repos}, job={i_job})"
    )
    # The stable prefix should be the bulk of the prompt.
    stable_share = i_job / len(prompt)
    assert stable_share > 0.3, f"Only {stable_share:.0%} of prompt is cacheable prefix"
    print(f"test_prompt_puts_variable_content_last_for_caching: PASS ({stable_share:.0%} cacheable)")


def test_bullet_bank_reaches_prompt_in_cached_prefix():
    """
    The bullet bank is identical every run, so it must sit in the cached prefix
    -- before target_job -- or it re-bills on every one of 100 calls.
    """
    master = yaml.safe_load(MASTER.read_text())
    bank = "## Matrix Engine\n- Reached 42 GFLOPS on matrix multiply with tiling."
    prompt = _build_user_prompt(master, "SWE", "Acme", "JOB_DESC", None, bank)

    assert "bullet_bank" in prompt, "Bank not sent to the model"
    assert "42 GFLOPS" in prompt
    assert prompt.index("bullet_bank") < prompt.index("target_job"), \
        "Bank must precede per-job content so it stays in the cached prefix"

    # Omitted entirely when empty -- no dangling key, no wasted tokens.
    without = _build_user_prompt(master, "SWE", "Acme", "JOB_DESC", None, "")
    assert "bullet_bank" not in without
    print("test_bullet_bank_reaches_prompt_in_cached_prefix: PASS")


def test_cover_letter_renders_pdf_and_text():
    pkg = ApplicationPackage(
        summary="", skills=[], experience_bullets={}, projects=[],
        cover_letter=(
            "I am writing to apply for the New Grad Software Engineer role.\n"
            "My graduate work focuses on applied machine learning systems.\n"
            "I would welcome the chance to discuss the role further."
        ),
        referral_note="short note", referral_dm="short dm",
    )
    candidate = {"first_name": "Jane", "last_name": "Doe",
                 "email": "jane@example.com", "phone": "555-555-5555"}

    out_pdf = Path("output/cover_letters/_test_letter.pdf")
    ok = render_cover_letter(pkg, candidate, "Acme & Co", "templates/cover_letter.tex.j2", str(out_pdf))
    assert ok, "Cover letter render failed"
    assert out_pdf.exists()

    txt = out_pdf.with_suffix(".txt")
    assert txt.exists(), "Paste-ready .txt not written (most ATS use a textarea, not upload)"
    assert "New Grad Software Engineer" in txt.read_text()

    text = subprocess.run(["pdftotext", str(out_pdf), "-"], capture_output=True, text=True).stdout
    assert "Jane Doe" in text
    # "&" is a LaTeX special char -- confirm escaping survived into the PDF.
    assert "Acme & Co" in text, "LaTeX escaping broke the company name"
    assert "graduate work" in text

    out_pdf.unlink(missing_ok=True)
    txt.unlink(missing_ok=True)
    print("test_cover_letter_renders_pdf_and_text: PASS")


def test_empty_cover_letter_returns_false():
    pkg = ApplicationPackage(
        summary="", skills=[], experience_bullets={}, projects=[],
        cover_letter="", referral_note="", referral_dm="",
    )
    ok = render_cover_letter(pkg, {"first_name": "A", "last_name": "B"}, "X",
                             "templates/cover_letter.tex.j2", "output/cover_letters/_none.pdf")
    assert ok is False
    print("test_empty_cover_letter_returns_false: PASS")


if __name__ == "__main__":
    test_note_under_limit_untouched()
    test_overlong_note_is_cut_to_limit()
    test_truncation_prefers_sentence_boundary()
    test_whitespace_collapsed_before_counting()
    test_prompt_puts_variable_content_last_for_caching()
    test_bullet_bank_reaches_prompt_in_cached_prefix()
    test_cover_letter_renders_pdf_and_text()
    test_empty_cover_letter_returns_false()
    print("\nAll cover-letter / referral tests passed.")
