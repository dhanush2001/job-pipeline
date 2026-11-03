"""
render_resume.py
-----------------
Merges tailored content (from tailor_resume.py) into the master resume data,
renders it through a Jinja2-templated .tex file, and compiles that to a PDF
with latexmk. Jinja2's default delimiters ({{ }}, {% %}) collide with LaTeX
syntax, so this uses LaTeX-safe delimiters instead (see resume_template.tex.j2
for the matching template config).
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jinja2

from datetime import date

from src.ats_score import _has_skill, skills_in
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
                # No date for GitHub-only projects: a repo's last-push date is when
                # you last touched the code, not when you did the work -- a 2022
                # project pushed in 2026 would show up as "2026-09-03". Leaving the
                # date off is honest; add the project to master_resume.yaml with
                # real dates if you want them shown.
                "dates": "",
                "technologies": repo.get("language", ""),
            }
    return lookup


def _match_project_meta(name: str, lookup: dict[str, dict[str, str]]) -> dict[str, str] | None:
    """
    Exact match first, then a lenient substring match either direction -- the
    model sometimes trims or extends a project's name while tailoring (e.g.
    "MedVQA" vs "MedVQA: Medical Data Platform").
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


# Never drop below this many skills: a near-empty section reads worse than an
# untrimmed one (and means the model returned an odd list).
MIN_TAILORED_SKILLS = 8
# A group with at least FIELD_MIN_HITS relevant skills is a field this role
# works in, so its neighbouring skills are kept too (in master order), up to
# FIELD_FILL entries for the group.
FIELD_MIN_HITS = 2
FIELD_FILL = 7
# A shown group never has fewer than this many skills: a group with only 1-2
# relevant ones is padded with the group's next skills (in master order).
MIN_PER_GROUP = 3


def _mentions(entry: str, term: str) -> bool:
    """Does a resume skill entry name this term? "Microsoft Azure" names
    "Azure"; "LLM APIs (OpenAI, Azure OpenAI)" names "OpenAI"."""
    e, t = entry.lower().strip(), term.lower().strip()
    if not t:
        return False
    if e == t:
        return True
    pat = (r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9+#])") if len(t) >= 2 else None
    return bool(pat and re.search(pat, e)) or (len(e) >= 3 and e in t)


# Other ways postings name a master entry ("Leverage AI tools in your workflow").
_ENTRY_ALIASES = {
    "ai coding tools": ["AI tools", "AI coding", "AI-assisted", "AI assistants",
                        "coding assistants", "AI-powered development"],
}


def _entry_terms(entry: str) -> list[str]:
    """The names a skill entry goes by: "LLM APIs (OpenAI, Azure OpenAI)" ->
    ["LLM APIs", "OpenAI", "Azure OpenAI"]; "C/C++" -> ["C/C++", "C++"]."""
    outer = re.sub(r"\([^)]*\)", "", entry).strip()
    inner = [x.strip() for m in re.findall(r"\(([^)]*)\)", entry) for x in re.split(r"[,;]", m)]
    terms = [outer] + [x for x in inner if x and not re.search(r"\b(daily|use|familiarity|tests?)\b", x)]
    terms += [p.strip() for p in outer.split("/") if "/" in outer and len(p.strip()) >= 2]
    terms += _ENTRY_ALIASES.get(outer.lower(), [])
    return [x for x in terms if len(x) >= 2]


def _named_in(job_text_lc: str, entry: str) -> bool:
    """Does the posting name this skill entry (or one of its parts) outright?"""
    for term in _entry_terms(entry):
        pat = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9+#])"
        if re.search(pat, job_text_lc):
            return True
    return False


def tailor_skills(master_skills: Any, tailored_flat: list[str], job_text: str = "",
                  core: list[str] | None = None) -> Any:
    """
    The skills section for ONE role: keep a master entry when the posting
    names it (exactly or by a known synonym -- deterministic, so a skill the
    posting asks for is never dropped), the model ranked it for this posting,
    or it is a core skill. Skills you don't have are never added. Groups keep their
    names; a group with nothing relevant is hidden; a group with several
    relevant skills keeps its closely related ones too (see FIELD_FILL); groups
    are ordered by their most relevant skill ("Languages" stays first). Falls
    back to the full list when tailoring would leave fewer than
    MIN_TAILORED_SKILLS.
    """
    if not (master_skills and isinstance(master_skills[0], dict)):
        return master_skills
    picks = [s for s in (tailored_flat or []) if isinstance(s, str) and s.strip()]
    jd = skills_in(job_text) if job_text else set()
    jd_lc = (job_text or "").lower()
    core = core or []
    if not picks and not jd:
        return master_skills

    def rank(entry: str) -> int | None:
        # Named in the posting -> always kept, ahead of everything else.
        if jd_lc and _named_in(jd_lc, entry):
            return -1
        for i, p in enumerate(picks):
            if _mentions(entry, p):
                return i
        if any(_has_skill(entry.lower(), s) for s in jd):
            return len(picks)
        if any(_mentions(entry, c) for c in core):
            return len(picks) + 1
        return None

    groups = []
    for gi, g in enumerate(master_skills):
        ranked = [(rank(e), i, e) for i, e in enumerate(_split_entries(g.get("entries", "")))]
        kept = sorted((r, i, e) for r, i, e in ranked if r is not None)
        if kept:
            # Same field as the role: add the group's other skills, in master
            # order (put niche ones last in master_resume.yaml) -- up to
            # FIELD_FILL when several are relevant, else up to MIN_PER_GROUP.
            target = FIELD_FILL if len(kept) >= FIELD_MIN_HITS else MIN_PER_GROUP
            extra = [(10**6, i, e) for r, i, e in ranked if r is None]
            kept += extra[:max(0, target - len(kept))]
        if kept:
            first = -1 if str(g.get("category", "")).lower().startswith("language") else kept[0][0]
            groups.append((first, gi, {**g, "entries": ", ".join(e for _, _, e in kept)}))
    total = sum(len(_split_entries(g["entries"])) for _, _, g in groups)
    if total < MIN_TAILORED_SKILLS:
        return master_skills
    return [g for _, _, g in sorted(groups, key=lambda x: (x[0], x[1]))]


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
        return _dedupe_skills(master_skills)
    if master_skills and isinstance(master_skills[0], dict):
        priority = {s.strip().lower(): i for i, s in enumerate(tailored_flat)}
        regrouped = []
        for group in master_skills:
            entries = _split_entries(group.get("entries", ""))
            entries.sort(key=lambda e: priority.get(e.lower(), 10**6))  # stable: unknowns keep order
            regrouped.append({**group, "entries": ", ".join(entries)})
        return _dedupe_skills(regrouped)
    return _dedupe_skills(tailored_flat)


def _split_entries(entries: Any) -> list[str]:
    """Split "A, B (x, y), C" on top-level commas only -- a comma inside
    parentheses belongs to that entry ("LLM APIs (OpenAI, Azure OpenAI)")."""
    out, buf, depth = [], "", 0
    for ch in str(entries or ""):
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth <= 0:
            out.append(buf.strip())
            buf = ""
        else:
            buf += ch
    out.append(buf.strip())
    return [e for e in out if e]


def _dedupe_skills(skills: Any) -> Any:
    """Each skill appears once on the page: a later repeat (case-insensitive,
    in any group) is dropped, and a group left empty is removed."""
    seen: set[str] = set()

    def first(e: str) -> bool:
        k = re.sub(r"\s+", " ", e.strip().lower())
        if k in seen:
            return False
        seen.add(k)
        return True

    if skills and isinstance(skills[0], dict):
        out = []
        for g in skills:
            kept = [e for e in _split_entries(g.get("entries", "")) if first(e)]
            if kept:
                out.append({**g, "entries": ", ".join(kept)})
        return out
    return [s for s in (skills or []) if isinstance(s, str) and first(s)]


def _gate_hits(item: dict[str, Any], text_lc: str, title_lc: str) -> bool:
    """Does a conditional entry (include_if / title_include_if) match this job?
    Unconditional entries always match."""
    def hits(text: str, terms: list) -> bool:
        return bool(text) and any(re.search(rf"\b{re.escape(str(t).lower())}\b", text) for t in terms)
    triggers, title_triggers = item.get("include_if") or [], item.get("title_include_if") or []
    if not triggers and not title_triggers:
        return True
    return hits(text_lc, triggers) or hits(title_lc, title_triggers)


def _ungated(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k not in ("include_if", "title_include_if")}


def scope_master(master: dict[str, Any], job_title: str = "", job_text: str = "") -> dict[str, Any]:
    """The master resume as THIS job should see it: experience entries gated by
    include_if / title_include_if are dropped when the job doesn't match (an IT
    support job shows only on IT/help-desk roles). Applied before tailoring, so
    the model can't mention a gated entry in bullets, summary or cover letter."""
    exp = master.get("experience") or []
    if not any(e.get("include_if") or e.get("title_include_if") for e in exp):
        return master
    scoped = dict(master)
    scoped["experience"] = [_ungated(e) for e in exp
                            if _gate_hits(e, (job_text or "").lower(), (job_title or "").lower())]
    return scoped


def select_certifications(
    certifications: list[dict[str, Any]] | None,
    job_text: str = "",
    job_title: str = "",
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

    `title_include_if` terms are matched against the job TITLE only. Use it
    for a word a description can mention in passing ("Python, Java or Go")
    that only marks a matching role when it names the role ("Java Developer").
    A cert with both lists shows when either one matches.
    """
    if not certifications:
        return []
    text_lc = (job_text or "").lower()
    title_lc = (job_title or "").lower()

    return [_ungated(c) for c in certifications if _gate_hits(c, text_lc, title_lc)]


def merge_tailored_content(
    master_resume: dict[str, Any],
    tailored: ApplicationPackage | None,
    github_repos: list[dict[str, Any]] | None = None,
    include_summary: bool | None = None,
    job_text: str = "",
    job_title: str = "",
) -> dict[str, Any]:
    """Overlay tailored summary/skills/bullets onto the master resume structure.

    include_summary: True always shows a summary, False never does, None (the
    default) shows one only when the master resume has one.

    job_text (title + description of the target role) gates conditional
    certifications -- see select_certifications.
    """
    merged = dict(master_resume)  # shallow copy is enough; we rebuild nested pieces below
    merged["skills"] = _dedupe_skills(master_resume.get("skills", []))

    # Certifications are factual and never rewritten by the model, so they're
    # resolved deterministically here (not via the LLM) and gated by the job.
    merged["certifications"] = select_certifications(
        master_resume.get("certifications", []), job_text, job_title
    )

    merged["experience"] = scope_master(master_resume, job_title, job_text).get("experience", [])

    if include_summary is False:
        merged["summary"] = ""
    if tailored is None:
        return merged

    # Summary section. It adds ~2-3 lines, so it's opt-in (resume.include_summary)
    # to protect the one-page layout -- the generated summary also always feeds
    # the cover letter regardless. If the master already ships a summary, honor
    # that too. When on, keep it short (see the prompt) so it stays one page.
    if include_summary and tailored.summary:
        merged["summary"] = tailored.summary
    elif include_summary is None and (master_resume.get("summary") or "").strip():
        merged["summary"] = tailored.summary or master_resume["summary"]

    if tailored.skills:
        merged["skills"] = _reconcile_skills(master_resume.get("skills", []), tailored.skills)
    if master_resume.get("tailor_skills", True):
        merged["skills"] = _dedupe_skills(tailor_skills(
            merged["skills"], tailored.skills, job_text, master_resume.get("core_skills")))

    new_experience = []
    for entry in merged["experience"]:
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
         proj_bullets: int, max_projects: int, skills_per_group: int = 0) -> dict[str, Any]:
    """Return a trimmed copy of the resume data to help it fit one page.
    skills_per_group keeps each group's first (most relevant) N entries."""
    d = dict(data)
    if drop_summary:
        d["summary"] = ""
    skills = d.get("skills")
    if skills_per_group and skills and isinstance(skills[0], dict):
        d["skills"] = [{**g, "entries": ", ".join(_split_entries(g.get("entries", ""))[:skills_per_group])}
                       for g in skills]
    if d.get("experience"):
        d["experience"] = [{**e, "bullets": (e.get("bullets") or [])[:exp_bullets]}
                           for e in d["experience"]]
    if d.get("projects"):
        d["projects"] = [{**p, "bullets": (p.get("bullets") or [])[:proj_bullets]}
                         for p in d["projects"][:max_projects]]
    return d


def _one_page_variants(data: dict[str, Any]):
    """Progressively tighter versions: as-is, then fewer bullets/projects, and
    only then without the summary. First one that compiles to one page wins."""
    yield data
    # The summary is the most role-specific part of the resume, so trim bullets
    # and projects first and drop the summary only as a later resort.
    yield _cap(data, False, 3, 3, 3)
    yield _cap(data, False, 3, 2, 3)
    # Shorter skill lists before giving up the summary.
    yield _cap(data, False, 3, 2, 3, skills_per_group=6)
    yield _cap(data, False, 3, 2, 3, skills_per_group=MIN_PER_GROUP + 1)
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
        # Keep the exact source of this PDF beside it, for hand edits.
        Path(output_pdf_path).with_suffix(".tex").write_text(tex_source, encoding="utf-8")
        return True, _pages_from_log(result.stdout)


@dataclasses.dataclass
class RenderResult:
    """Outcome of a render. Truthy when a PDF was produced, so existing callers
    can keep writing `if not render_resume(...)`. `data` is the content variant
    ACTUALLY rendered (possibly trimmed to fit one page) -- score that, not the
    pre-trim input, or the reported ATS will overstate what's in the PDF.
    `one_page_ok` is False when fit_one_page was requested but couldn't be met."""
    ok: bool
    data: dict[str, Any] | None = None
    pages: int | None = None
    one_page_ok: bool = True

    def __bool__(self) -> bool:
        return self.ok


def render_resume(
    resume_data: dict[str, Any],
    template_path: str,
    output_pdf_path: str,
    fit_one_page: bool = False,
) -> RenderResult:
    """
    Renders resume_data through the template and compiles it with latexmk.
    Returns a RenderResult (truthy on success); the final PDF is copied to
    output_pdf_path and its LaTeX source to the same path with a .tex suffix.

    With fit_one_page=True, if the resume compiles to more than one page it is
    re-rendered with progressively less content (drop summary, then fewer
    bullets/projects). If even the most-trimmed variant is >1 page, the PDF is
    still written but RenderResult.one_page_ok is False -- the caller is told the
    constraint could not be met rather than being misled by a success return.
    """
    template = LATEX_JINJA_ENV.get_template(str(Path(template_path)))
    variants = list(_one_page_variants(resume_data) if fit_one_page else iter([resume_data]))

    for i, data in enumerate(variants):
        tex_source = template.render(resume=_escape_tree(data))
        ok, pages = _compile_tex(tex_source, output_pdf_path)
        if not ok:
            return RenderResult(False)
        if pages is None or pages <= 1:
            logger.info("Rendered resume -> %s (%s page)", output_pdf_path, pages or "?")
            return RenderResult(True, data=data, pages=pages, one_page_ok=True)
        if i == len(variants) - 1:
            if fit_one_page:
                logger.warning(
                    "Resume still %d pages after trimming -> %s (one-page constraint NOT met)",
                    pages, output_pdf_path,
                )
            return RenderResult(True, data=data, pages=pages, one_page_ok=not fit_one_page)
        logger.info("Resume was %d pages; trimming to fit one page...", pages)
    return RenderResult(False)


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
        Path(output_pdf_path).with_suffix(".tex").write_text(tex_source, encoding="utf-8")
    return True
