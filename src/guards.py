"""
guards.py
---------
Deterministic checks on AI-rewritten resume content, run after each model pass.
The prompts ask for these rules too, but prompts are requests; these are checks.

1. Skills stay under the job where they were used.
   The grounding check accepts any skill found anywhere in your evidence, so a
   skill from a project (e.g. "security" from a security benchmark) could be
   written into your internship bullets -- true skill, false placement. Every
   experience bullet may only mention skills that the SAME entry in the master
   resume supports. A bullet that adds others is replaced with the previous
   version (pre-judge bullet, or your original master bullet).

2. The judge must fix what it flags.
   Red flags often quote a weak phrase ('fast-moving, loosely-defined
   environments'). If the rewritten summary still contains a quoted phrase,
   the sentence carrying it is dropped.

3. In-progress work isn't described as finished.
   For projects whose dates end in "Present", a summary sentence that pairs the
   project's distinctive skills with a completed-result verb ("built", "cut",
   "achieved", "benchmarks"...) is flagged for review.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from src.ats_score import skills_in

_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _plain(text: str) -> str:
    return _BOLD.sub(r"\1", text or "")


def _singular(text: str) -> str:
    # "REST APIs" should also allow "REST API"; plural -> singular per word.
    return re.sub(r"\b([A-Za-z]{3,})s\b", r"\1", text)


# ---- 1. per-entry skill scoping ---------------------------------------------------
# A skill that necessarily comes with another: TypeScript/React work is
# JavaScript work, a Postgres database means SQL, FastAPI means Python.
_IMPLIES: dict[str, set[str]] = {
    "typescript": {"javascript"}, "react": {"javascript"}, "next.js": {"javascript", "react"},
    "node.js": {"javascript"}, "postgresql": {"sql"}, "mysql": {"sql"}, "sqlite": {"sql"},
    "fastapi": {"python", "api"}, "flask": {"python"}, "django": {"python"}, "pytorch": {"python"},
    "pandas": {"python"}, "spring boot": {"java", "spring"}, "rest": {"api"}, "rest apis": {"api"},
}


def entry_skills(entry: dict[str, Any]) -> set[str]:
    """Skills the master resume supports for ONE experience entry: its bullets,
    technologies line, title, and company -- plus skills they imply."""
    text = " ".join([entry.get("title", ""), entry.get("company", ""), entry.get("technologies", "")]
                    + [_plain(b) for b in entry.get("bullets") or []])
    found = skills_in(text) | skills_in(_singular(text))
    for s in list(found):
        found |= _IMPLIES.get(s, set())
    return found


def allowed_skills_by_entry(master: dict[str, Any]) -> dict[str, list[str]]:
    return {e["id"]: sorted(entry_skills(e)) for e in master.get("experience", []) or [] if e.get("id")}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _plain(text).lower())


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _words(a), _words(b)).ratio()


def _closest(bullet: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: _similar(bullet, c))


# Above this word-level similarity, two bullets say the same thing.
_DUPLICATE = 0.6


def guard_experience(bullets_by_id: dict[str, list[str]], master: dict[str, Any],
                     fallback_by_id: dict[str, list[str]] | None = None
                     ) -> tuple[dict[str, list[str]], list[str]]:
    """Return (cleaned bullets, notes). A bullet that names a skill its own
    entry doesn't support is replaced by the fallback bullet at the same index
    (if that one is clean), else by the closest original master bullet -- but
    never by one that repeats a bullet already kept (the job would read the
    same thing twice). If every candidate is a repeat, the bullet is dropped."""
    entries = {e.get("id"): e for e in master.get("experience", []) or []}
    cleaned: dict[str, list[str]] = {}
    notes: list[str] = []
    for eid, bullets in (bullets_by_id or {}).items():
        entry = entries.get(eid)
        if entry is None:
            cleaned[eid] = bullets
            continue
        allowed = entry_skills(entry)
        originals = list(entry.get("bullets") or [])
        fallback = (fallback_by_id or {}).get(eid) or []
        bullets = list(bullets or [])
        clean_ones = [b for b in bullets if not (skills_in(_plain(b)) - allowed)]
        out: list[str] = []
        for i, b in enumerate(bullets):
            extra = sorted(skills_in(_plain(b)) - allowed)
            if not extra:
                out.append(b)
                continue
            # Candidates in preference order: same-position previous version,
            # then your originals, closest first. Must be clean and must not
            # repeat a bullet this job already shows.
            cands = ([(fallback[i], "previous version")] if i < len(fallback) else []) + \
                    [(o, "your original") for o in sorted(originals, key=lambda o: -_similar(b, o))]
            others = [x for x in clean_ones + out if x is not b]
            pick = next(((c, src) for c, src in cands
                         if not (skills_in(_plain(c)) - allowed)
                         and all(_similar(c, x) < _DUPLICATE for x in others)), None)
            if pick:
                out.append(pick[0])
                notes.append(f"{eid}: replaced a bullet that named {', '.join(extra)} (not used in this job) "
                             f"with {pick[1]}: \"{_plain(pick[0])[:80]}\"")
            else:
                notes.append(f"{eid}: dropped a bullet that named {', '.join(extra)} (not used in this job); "
                             "every replacement would repeat another bullet")
        cleaned[eid] = out
    return cleaned, notes


# ---- sentences ------------------------------------------------------------------
# "M.S.", "Ph.D.", "e.g." end in a period but don't end a sentence.
_ABBREV = re.compile(r"\b(?:[A-Za-z]{1,2}\.){2,}$|\b(?:e\.g|i\.e|etc|vs|approx|Dr|Mr|Ms|Mrs|Jr|Sr|Inc|Co)\.$")


def _sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, but never after an abbreviation or
    inside parentheses -- "(M.S. CS, December 2026) who shipped ..." is one sentence."""
    text = (text or "").strip()
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        head = text[start:m.start()]
        if _ABBREV.search(head) or head.count("(") > head.count(")"):
            continue
        out.append(head)
        start = m.end()
    if text[start:]:
        out.append(text[start:])
    return out


# ---- 2. judge must fix what it flags ----------------------------------------------
_QUOTED = re.compile(r"['‘“\"]([^'’”\"]{12,120})['’”\"]")


def flagged_phrases(red_flags: list[str]) -> list[str]:
    """Phrases quoted inside red flags -- the exact wording the judge objected to."""
    out = []
    for flag in red_flags or []:
        for m in _QUOTED.finditer(flag):
            phrase = m.group(1).strip().rstrip(".…").strip().lower()
            if len(phrase.split()) >= 2 and phrase not in out:
                out.append(phrase)
    return out


def repair_summary(summary: str, phrases: list[str]) -> tuple[str, list[str]]:
    """Drop summary sentences that still contain a flagged phrase. If that
    would empty the summary, keep it and report instead."""
    if not summary or not phrases:
        return summary, []
    sentences = _sentences(summary)
    keep, removed = [], []
    for s in sentences:
        hit = next((p for p in phrases if p in s.lower()), None)
        (removed if hit else keep).append(hit or s)
    if not removed:
        return summary, []
    remainder = " ".join(keep)
    # Dropping a sentence must leave a real summary (>= 15 words naming a skill),
    # never filler -- e.g. a strong sentence that merely STARTS with a flagged
    # "MS CS candidate" must not be thrown away for it.
    if not keep or len(remainder.split()) < 15 or not skills_in(_plain(remainder)):
        return summary, [f"summary still contains flagged phrase(s): {', '.join(removed)} (left as-is)"]
    return remainder, [f"dropped summary sentence with flagged phrase: '{p}'" for p in removed]


# ---- pipeline entry points -------------------------------------------------------
def guard_tailored(pkg: Any, master: dict[str, Any]) -> list[str]:
    """Run right after the tailoring pass (before any judge)."""
    if pkg is None or not getattr(pkg, "experience_bullets", None):
        return []
    pkg.experience_bullets, notes = guard_experience(pkg.experience_bullets, master)
    return notes


def apply_judge(pkg: Any, review: dict | None, master: dict[str, Any],
                repair: Any = None) -> list[str]:
    """Merge a judge review into the package with all three guards applied:
    rewritten bullets are skill-scoped per job (falling back to the pre-judge
    bullet), the summary loses sentences that repeat a flagged phrase, and an
    in-progress project presented as finished is reported."""
    notes: list[str] = []
    if pkg is None:
        return notes
    pre_judge_summary = pkg.summary
    if review and review.get("experience"):
        pre_judge = {k: list(v) for k, v in pkg.experience_bullets.items()}
        judged, n = guard_experience(review["experience"], master, pre_judge)
        pkg.experience_bullets.update(judged)
        notes += n
    if review and review.get("summary"):
        phrases_ = flagged_phrases(review.get("red_flags") or [])
        pkg.summary, n = repair_summary(review["summary"], phrases_)
        still = [p for p in phrases_ if p in review["summary"].lower()]
        if still and repair is not None:
            # The judge repeated a phrase it flagged. Dropping that sentence can
            # lose the strongest claim (or gut the summary), so first ask for a
            # targeted rewrite of just those phrases in the full summary.
            names = [p.get("name", "").split(":")[0].strip() for p in in_progress_projects(master)]
            try:
                fixed = repair(review["summary"], names, phrases_)
            except Exception:  # noqa: BLE001
                fixed = None
            if fixed and _summary_ok(fixed, master, phrases_):
                pkg.summary = fixed
                n = [f"rewrote summary phrases the judge itself flagged: {', '.join(still)}"]
        notes += n
    phrases = flagged_phrases((review or {}).get("red_flags") or [])
    pkg.summary, n = fix_overclaim(pkg.summary, pre_judge_summary, master, phrases, repair)
    notes += n
    return notes


# A summary shorter than this, or naming no skill at all, is filler -- not an
# acceptable result of a repair (e.g. "Comfortable owning systems end to end").
_MIN_SUMMARY_WORDS = 15


def _summary_ok(text: str, master: dict[str, Any], flagged: list[str]) -> bool:
    return bool(text) and len(text.split()) >= _MIN_SUMMARY_WORDS and bool(skills_in(_plain(text))) \
        and not overclaimed_in_progress(text, master) and not any(p in text.lower() for p in flagged)


def fix_overclaim(summary: str, previous: str, master: dict[str, Any],
                  flagged: list[str] | None = None, repair: Any = None) -> tuple[str, list[str]]:
    """If the summary presents an in-progress project as finished, try in order:
    1. the previous (pre-judge) summary, if it passes every check;
    2. `repair(summary, project_names, flagged)` -- a targeted model rewrite
       that only fixes the tense -- if its result passes every check;
    3. dropping the offending sentence, if what's left is still a real summary
       (>= 15 words and names a skill), not filler;
    4. otherwise keep it and report it for review."""
    flagged = flagged or []
    hits = overclaimed_in_progress(summary, master)
    if not hits:
        return summary, []
    if previous and previous != summary and _summary_ok(previous, master, flagged):
        return previous, ["summary presented in-progress work as finished -> kept the pre-judge summary"]
    if repair is not None:
        names = [p.get("name", "").split(":")[0].strip() for p in in_progress_projects(master)]
        try:
            fixed = repair(summary, names, flagged)
        except Exception:  # noqa: BLE001 -- a failed repair falls through to the next option
            fixed = None
        if fixed and _summary_ok(fixed, master, flagged):
            return fixed, ["summary presented in-progress work as finished -> rewritten in progressive tense"]
    sentences = _sentences(summary)
    keep = " ".join(s for s in sentences if not overclaimed_in_progress(s, master))
    if keep and keep != summary.strip() and len(keep.split()) >= _MIN_SUMMARY_WORDS and skills_in(_plain(keep)):
        return keep, ["dropped a summary sentence that presented in-progress work as finished"]
    return summary, hits


# ---- 3. in-progress work --------------------------------------------------------
_DONE_VERBS = re.compile(
    r"\b(built|builds|shipped|ships|deployed|deploys|delivered|achieved|reduced|reduces|improved|"
    r"improves|cut|cuts|raised|raises|optimized|optimizes|benchmarked|implemented|"
    r"launched|scaled|accelerated|architected|engineered|designed)\b", re.I)


def in_progress_projects(master: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in master.get("projects", []) or [] if "present" in str(p.get("dates", "")).lower()]


def _distinctive(project: dict[str, Any], master: dict[str, Any]) -> set[str]:
    """Skills this project names that no other master entry -- and not the
    skills section -- does. (C/C++ in your skills list isn't tiny-serve-only.)"""
    own = skills_in(" ".join([project.get("technologies", "")] + [_plain(b) for b in project.get("bullets") or []]))
    others = []
    for e in master.get("experience", []) or []:
        others += [e.get("technologies", "")] + list(e.get("bullets") or [])
    for p in master.get("projects", []) or []:
        if p is not project:
            others += [p.get("technologies", "")] + list(p.get("bullets") or [])
    for g in master.get("skills", []) or []:
        others.append(g.get("entries", "") if isinstance(g, dict) else str(g))
    return own - skills_in(" ".join(_plain(o) for o in others))


# "implementing" is left out on purpose: "built X, implementing Y" is past work.
_PROGRESSIVE = re.compile(r"\b(building|developing|prototyping|currently|working on|in progress)\b", re.I)


def _governed_by_done_verb(sentence: str, start: int, end: int) -> bool:
    """Is the project mention at [start, end) claimed as finished? True when a
    completed-result verb precedes it with no progressive word in between, or
    follows it within the same clause ("tiny-serve, which cut latency")."""
    before = sentence[:start]
    dones = list(_DONE_VERBS.finditer(before))
    if dones:
        last_done = dones[-1].end()
        if not _PROGRESSIVE.search(before[last_done:]) and not re.search(r"[;.]", before[last_done:]):
            # "Building X and shipped Y" -- a progressive before the done verb in
            # the same clause still governs X only if the done verb isn't closer.
            return True
    clause_after = re.split(r"[,;.]", sentence[end:], maxsplit=1)[0]
    return bool(_DONE_VERBS.search(" ".join(clause_after.split()[:8])))


def overclaimed_in_progress(text: str, master: dict[str, Any]) -> list[str]:
    """Sentences that present an in-progress project as finished: they use the
    project's distinctive skills alongside a completed-result verb. Returned
    for review (not auto-edited -- the wording is a judgment call)."""
    hits = []
    for p in in_progress_projects(master):
        # Only projects whose own bullets are still progressive ("Building...").
        if not any(re.match(r"\s*\**\s*\w+ing\b", _plain(b)) for b in p.get("bullets") or []):
            continue
        marks = _distinctive(p, master) | ({p.get("name", "").split(":")[0].strip().lower()} - {""})
        for s in _sentences(_plain(text or "")):
            low = s.lower()
            spans = [(m.start(), m.end()) for mark in marks if mark
                     for m in re.finditer(re.escape(mark), low)]
            if any(_governed_by_done_verb(s, a, b) for a, b in spans):
                hits.append(f"{p.get('name', '?').split(':')[0]} is in progress but reads as done: \"{s.strip()[:140]}\"")
    return hits


# ---- project selection -----------------------------------------------------------
def enforce_projects(pkg: Any, selected: list[str] | None) -> list[str]:
    """Make the tailored projects match the picker's choice: keep the selected
    ones in the selected order and drop anything else. If the tailoring call
    omitted a selected project, keep what it returned and report the gap
    (bullets can't be written without the model)."""
    if pkg is None or not selected or not getattr(pkg, "projects", None):
        return []
    norm = lambda n: re.sub(r"[^a-z0-9]", "", (n or "").lower())
    def match(name: str) -> int | None:
        k = norm(name)
        for i, sel in enumerate(selected):
            s_ = norm(sel)
            if k and (k == s_ or k in s_ or s_ in k):
                return i
        return None
    indexed = [(match(p.get("name", "")), p) for p in pkg.projects]
    kept = sorted(((i, p) for i, p in indexed if i is not None), key=lambda x: x[0])
    notes = []
    extra = [p.get("name", "") for i, p in indexed if i is None]
    missing = [s_ for j, s_ in enumerate(selected) if j not in {i for i, _ in kept}]
    if not kept:
        return [f"tailoring ignored the selected projects {selected}; kept its own picks"]
    pkg.projects = [p for _, p in kept]
    if extra:
        notes.append(f"removed projects not selected for this role: {', '.join(extra)}")
    if missing:
        notes.append(f"selected project(s) missing from tailoring output: {', '.join(missing)}")
    return notes
