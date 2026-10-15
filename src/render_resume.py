"""
render_resume.py
-----------------
Merges tailored content (from tailor_resume.py) into the master resume data,
renders it through a Jinja2-templated .tex file, and compiles that to a PDF
with latexmk. Jinja2's default delimiters ({{ }}, {% %}) collide with LaTeX
syntax, so this uses LaTeX-safe delimiters instead (see templates/resume.tex.j2
for the matching template config).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jinja2

from datetime import date

from src.tailor_resume import ApplicationPackage

logger = logging.getLogger(__name__)

LATEX_JINJA_ENV = jinja2.Environment(
    block_start_string=r"\BLOCK{",
    block_end_string="}",
    variable_start_string=r"\VAR{",
    variable_end_string="}",
    comment_start_string=r"\#{",
    comment_end_string="}",
    line_statement_prefix="%%",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(searchpath="."),
)

# LLM output routinely contains typographic Unicode: em/en dashes, smart
# quotes, arrows, ellipses. With a standard pdflatex preamble these do NOT
# fail loudly -- they compile and silently emit garbage characters (verified:
# an arrow became a control char, quotes came out mangled). A garbled glyph on
# a resume sent to 100 employers is worse than a crash, so all are normalized.
_UNICODE_REPLACEMENTS = {
    "\u2014": "---", "\u2013": "--", "\u2012": "--",
    "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u2018": "`", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": "``", "\u201d": "''", "\u201e": "``", "\u201f": "``",
    "\u2192": "$\\rightarrow$", "\u2190": "$\\leftarrow$",
    "\u2194": "$\\leftrightarrow$", "\u21d2": "$\\Rightarrow$",
    "\u2264": "$\\leq$", "\u2265": "$\\geq$", "\u2260": "$\\neq$",
    "\u00d7": "$\\times$", "\u00f7": "$\\div$", "\u00b1": "$\\pm$",
    "\u2248": "$\\approx$", "\u221e": "$\\infty$",
    "\u2026": "...", "\u2022": "$\\bullet$", "\u00b7": "$\\cdot$",
    "\u2032": "'", "\u2033": "''",
    "\u00a0": " ", "\u2009": " ", "\u200a": " ", "\u2007": " ", "\u202f": " ",
    "\u200b": "", "\ufeff": "",
    "\u00ae": "\\textregistered{}", "\u00a9": "\\copyright{}",
    "\u2122": "\\texttrademark{}", "\u20ac": "\\texteuro{}",
    "\u00a3": "\\pounds{}", "\u00b0": "$^{\\circ}$",
}


def normalize_unicode(value: str) -> str:
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        if char in value:
            value = value.replace(char, replacement)
    return value


_LATEX_SPECIAL_CHARS = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}


# Bold matters in this resume style, but letting the model emit raw \textbf{}
# would mean not escaping its output at all -- handing it arbitrary LaTeX. So
# the model writes markdown **bold**, which survives escaping untouched
# (asterisks are not special in LaTeX) and is converted afterwards.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def escape_latex(value: Any) -> Any:
    """Escape LaTeX specials, normalize Unicode, then re-enable **bold** only."""
    if not isinstance(value, str):
        return value
    # Escape FIRST: Unicode is not in the special-char table so it passes
    # through untouched, and normalizing afterwards means replacements that
    # themselves contain LaTeX (like the arrow) are not escaped to death.
    pattern = re.compile("|".join(re.escape(k) for k in _LATEX_SPECIAL_CHARS))
    escaped = pattern.sub(lambda m: _LATEX_SPECIAL_CHARS[m.group()], value)
    normalized = normalize_unicode(escaped)
    return _MD_BOLD.sub(r"\\textbf{\1}", normalized)


def _escape_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _escape_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_escape_tree(v) for v in obj]
    return escape_latex(obj)


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _project_metadata_lookup(
    master_resume: dict[str, Any],
    github_repos: list[dict[str, Any]] | None,
) -> dict[str, dict[str, str]]:
    """
    Map project name -> {dates, technologies} from the grounded sources. The
    model returns only name + bullets for projects; dates and tech stack are
    facts it must not invent, so they are carried over from here instead.
    Master-resume projects win over a same-named repo.
    """
    lookup: dict[str, dict[str, str]] = {}
    for proj in master_resume.get("projects", []):
        name = _norm_name(proj.get("name", ""))
        if name:
            lookup[name] = {
                "dates": proj.get("dates", ""),
                "technologies": proj.get("technologies", ""),
            }
    for repo in github_repos or []:
        name = _norm_name(repo.get("name", ""))
        if name and name not in lookup:
            lookup[name] = {
                "dates": repo.get("last_pushed", ""),
                "technologies": repo.get("language", ""),
            }
    return lookup


def _match_project_meta(name: str, lookup: dict[str, dict[str, str]]) -> dict[str, str] | None:
    """
    Exact match first, then a lenient substring match either direction -- the
    model sometimes trims or extends a project's name while tailoring (e.g.
    "LLaVA-Med" vs "LLaVA-Med: Medical Data Platform").
    """
    key = _norm_name(name)
    if not key:
        return None
    if key in lookup:
        return lookup[key]
    for source_name, meta in lookup.items():
        if source_name and (source_name in key or key in source_name):
            return meta
    return None


def _reconcile_skills(master_skills: Any, tailored_flat: list[str]) -> Any:
    """
    The model returns a FLAT, relevance-ordered skills list, but a grouped
    template expects [{category, entries}]. Overwriting grouped skills with a
    flat list makes the template print an empty line per skill -- blank section
    that overflows to a second page. So:

    - Grouped master ([{category, entries}]): keep the groups, but reorder the
      entries WITHIN each group so tailored-prioritized skills come first. All
      real skills are kept; nothing invented is added.
    - Flat master ([str]): the tailored flat list is the right shape already.
    """
    if not tailored_flat:
        return master_skills
    if master_skills and isinstance(master_skills[0], dict):
        priority = {s.strip().lower(): i for i, s in enumerate(tailored_flat)}
        regrouped = []
        for group in master_skills:
            entries = [e.strip() for e in str(group.get("entries", "")).split(",") if e.strip()]
            entries.sort(key=lambda e: priority.get(e.lower(), 10**6))  # stable: unknowns keep order
            regrouped.append({**group, "entries": ", ".join(entries)})
        return regrouped
    return tailored_flat


def select_certifications(
    certifications: list[dict[str, Any]] | None,
    job_text: str = "",
) -> list[dict[str, Any]]:
    """Pick which certifications belong on THIS resume.

    A certification may carry an `include_if` list of trigger terms. It is then
    rendered only when the job (title + description) actually mentions one of
    them -- so the GUVI Java course appears on a Java/Spring posting but not on
    every resume. A certification with no `include_if` is unconditional and
    always kept.

    Matching is word-boundary and case-insensitive, so "java" matches "Java"
    and "Spring/Java" but NOT "JavaScript". With no job_text (e.g. a stub/dry
    render) only the unconditional certs are kept -- a conditional one needs a
    real signal to show up, never by default.
    """
    if not certifications:
        return []
    text_lc = (job_text or "").lower()
    out: list[dict[str, Any]] = []
    for cert in certifications:
        triggers = cert.get("include_if") or []
        clean = {k: v for k, v in cert.items() if k != "include_if"}
        if not triggers:
            out.append(clean)
            continue
        if text_lc and any(
            re.search(rf"\b{re.escape(str(t).lower())}\b", text_lc) for t in triggers
        ):
            out.append(clean)
    return out


def merge_tailored_content(
    master_resume: dict[str, Any],
    tailored: ApplicationPackage | None,
    github_repos: list[dict[str, Any]] | None = None,
    include_summary: bool = False,
    job_text: str = "",
) -> dict[str, Any]:
    """Overlay tailored summary/skills/bullets onto the master resume structure.

    job_text (title + description of the target role) gates conditional
    certifications -- see select_certifications.
    """
    merged = dict(master_resume)  # shallow copy is enough; we rebuild nested pieces below

    # Certifications are factual and never rewritten by the model, so they're
    # resolved deterministically here (not via the LLM) and gated by the job.
    merged["certifications"] = select_certifications(
        master_resume.get("certifications", []), job_text
    )

    if tailored is None:
        return merged

    # Summary section. It adds ~2-3 lines, so it's opt-in (resume.include_summary)
    # to protect the one-page layout -- the generated summary also always feeds
    # the cover letter regardless. If the master already ships a summary, honor
    # that too. When on, keep it short (see the prompt) so it stays one page.
    if include_summary and tailored.summary:
        merged["summary"] = tailored.summary
    elif (master_resume.get("summary") or "").strip():
        merged["summary"] = tailored.summary or master_resume["summary"]

    if tailored.skills:
        merged["skills"] = _reconcile_skills(master_resume.get("skills", []), tailored.skills)

    new_experience = []
    for entry in master_resume.get("experience", []):
        entry = dict(entry)
        bullets = tailored.experience_bullets.get(entry.get("id"))
        if bullets:
            entry["bullets"] = bullets
        new_experience.append(entry)
    merged["experience"] = new_experience

    # Projects: the tailored list REPLACES the master list, because the point
    # is to show only the 2-4 repos relevant to this role rather than all of
    # them. Falls back to the master resume's projects if tailoring returned
    # none (e.g. GitHub fetch failed) so the section never silently vanishes.
    #
    # The model returns only name + bullets, so dates and the tech stack are
    # re-attached here from the grounded source (master projects or GitHub
    # repos) by matching on name -- otherwise the template's date/technologies
    # fields render empty and the tailored resume loses layout the master had.
    if tailored.projects:
        meta = _project_metadata_lookup(master_resume, github_repos)
        merged_projects: list[dict[str, Any]] = []
        for p in tailored.projects:
            entry: dict[str, Any] = {"name": p["name"], "bullets": p.get("bullets", [])}
            source = _match_project_meta(p["name"], meta)
            if source:
                if source.get("dates"):
                    entry["dates"] = source["dates"]
                if source.get("technologies"):
                    entry["technologies"] = source["technologies"]
            merged_projects.append(entry)
        merged["projects"] = merged_projects

    return merged


# pdflatex prints e.g. "Output written on resume.pdf (2 pages, 12345 bytes)".
_PAGES_RE = re.compile(r"Output written on \S+ \((\d+) pages?,")


def _pages_from_log(stdout: str) -> int | None:
    m = _PAGES_RE.search(stdout)
    return int(m.group(1)) if m else None


def _cap(data: dict[str, Any], drop_summary: bool, exp_bullets: int,
         proj_bullets: int, max_projects: int) -> dict[str, Any]:
    """Return a trimmed copy of the resume data to help it fit one page."""
    d = dict(data)
    if drop_summary:
        d["summary"] = ""
    if d.get("experience"):
        d["experience"] = [{**e, "bullets": (e.get("bullets") or [])[:exp_bullets]}
                           for e in d["experience"]]
    if d.get("projects"):
        d["projects"] = [{**p, "bullets": (p.get("bullets") or [])[:proj_bullets]}
                         for p in d["projects"][:max_projects]]
    return d


def _one_page_variants(data: dict[str, Any]):
    """Progressively tighter versions: as-is, then drop summary, then fewer
    bullets/projects. First one that compiles to a single page wins."""
    yield data
    yield _cap(data, True, 3, 3, 3)
    yield _cap(data, True, 3, 2, 3)
    yield _cap(data, True, 2, 2, 2)


def _compile_tex(tex_source: str, output_pdf_path: str) -> tuple[bool, int | None]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "resume.tex").write_text(tex_source, encoding="utf-8")
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "resume.tex"],
            cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )
        pdf_file = tmp_path / "resume.pdf"
        if result.returncode != 0 or not pdf_file.exists():
            logger.error("LaTeX compile failed:\n%s", result.stdout[-3000:])
            return False, None
        Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf_file, output_pdf_path)
        return True, _pages_from_log(result.stdout)


def render_resume(
    resume_data: dict[str, Any],
    template_path: str,
    output_pdf_path: str,
    fit_one_page: bool = False,
) -> bool:
    """
    Renders resume_data through the template and compiles it with latexmk.
    Returns True on success; only the final PDF is copied to output_pdf_path.

    With fit_one_page=True, if the resume compiles to more than one page it is
    re-rendered with progressively less content (drop summary, then fewer
    bullets/projects) until it fits -- a hard one-page guarantee.
    """
    template = LATEX_JINJA_ENV.get_template(str(Path(template_path)))
    variants = _one_page_variants(resume_data) if fit_one_page else iter([resume_data])

    last_ok = False
    variants = list(variants)
    for i, data in enumerate(variants):
        tex_source = template.render(resume=_escape_tree(data))
        ok, pages = _compile_tex(tex_source, output_pdf_path)
        if not ok:
            return False
        last_ok = True
        if pages is None or pages <= 1:
            logger.info("Rendered resume -> %s (%s page)", output_pdf_path, pages or "?")
            return True
        if i == len(variants) - 1:
            logger.warning("Resume still %d pages after trimming -> %s", pages, output_pdf_path)
            return True
        logger.info("Resume was %d pages; trimming to fit one page...", pages)
    return last_ok


def render_cover_letter(
    package: ApplicationPackage,
    candidate: dict[str, Any],
    company: str,
    template_path: str,
    output_pdf_path: str,
    also_write_text: bool = True,
) -> bool:
    """
    Renders the cover letter to PDF, and by default also writes a .txt beside
    it. The text version matters more than it sounds: most ATS forms have a
    paste-in cover letter textarea rather than an upload field, so the plain
    text is what you'll actually use most of the time.
    """
    if not package.cover_letter:
        return False

    paragraphs = [p.strip() for p in package.cover_letter.split("\n") if p.strip()]

    letter_data = {
        "name": f"{candidate.get('first_name','')} {candidate.get('last_name','')}".strip(),
        "email": candidate.get("email", ""),
        "phone": candidate.get("phone", ""),
        "company": company,
        "date": date.today().strftime("%B %d, %Y"),
        "paragraphs": paragraphs,
    }

    if also_write_text:
        txt_path = Path(output_pdf_path).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(package.cover_letter, encoding="utf-8")

    template = LATEX_JINJA_ENV.get_template(template_path)
    tex_source = template.render(letter=_escape_tree(letter_data))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "letter.tex").write_text(tex_source, encoding="utf-8")
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "letter.tex"],
            cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )
        pdf_file = tmp_path / "letter.pdf"
        if result.returncode != 0 or not pdf_file.exists():
            logger.error("Cover letter compile failed:\n%s", result.stdout[-2000:])
            return False
        Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf_file, output_pdf_path)
    return True
