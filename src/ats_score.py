"""
ats_score.py
------------
A transparent, offline ATS-style keyword-match score: of the skills and
technologies a job posting asks for, how many actually appear in the tailored
resume. This is exactly what tools like Jobscan compute (a match rate against
the posting) -- done here with a curated skills vocabulary so stopwords don't
inflate the number. The actionable part is the list of *missing* keywords: the
posting mentions them, your resume doesn't.

Deterministic and free -- keyword overlap, no LLM, no network. Tune SKILLS
rather than the code.
"""
from __future__ import annotations

import re
from typing import Any

# Curated skills / technologies to look for. Kept lowercase; multi-word and
# symbol-bearing entries (c++, ci/cd, node.js) are matched as substrings, plain
# alphanumeric ones on word boundaries so "go" doesn't match "google".
SKILLS: set[str] = {
    # languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "scala", "kotlin", "swift", "php", "matlab", "sql", "bash",
    # web / backend
    "react", "angular", "vue", "next.js", "node.js", "node", "express", "html",
    "css", "tailwind", "redux", "graphql", "rest", "rest api", "rest apis",
    "django", "flask", "fastapi", "spring", "spring boot", ".net", "asp.net",
    # data / ml
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "llm", "llms", "pytorch", "tensorflow", "keras",
    "scikit-learn", "sklearn", "pandas", "numpy", "scipy", "hugging face",
    "transformers", "cuda", "reinforcement learning", "data analysis",
    "data engineering", "etl", "spark", "hadoop", "kafka", "airflow", "dbt",
    "tableau", "power bi", "statistics", "a/b testing", "xgboost", "lora",
    "peft", "fine-tuning", "rag", "quantization",
    # cloud / devops / infra
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
    "ci/cd", "jenkins", "github actions", "linux", "unix", "microservices",
    "distributed systems", "nginx", "serverless", "lambda", "grpc",
    # ml inference / performance (tiny-serve territory)
    "inference", "latency", "throughput", "vllm", "tensorrt", "triton", "kernel",
    "kernels", "batching", "kv cache", "paged attention", "cuda", "gpu",
    "high-performance", "high performance", "concurrency", "profiling",
    # databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "sqlite", "dynamodb",
    "snowflake", "bigquery", "firebase", "elasticsearch",
    # practices / cs fundamentals
    "git", "agile", "scrum", "unit testing", "testing", "tdd", "api",
    "data structures", "algorithms", "system design", "object-oriented", "oop",
    # roles / analytics
    "data analyst", "data scientist", "data engineer", "business intelligence",
    "data pipeline", "dashboards", "reporting",
    # IT support / systems administration
    "technical support", "help desk", "service desk", "desktop support", "troubleshooting",
    "hardware", "active directory", "office 365", "microsoft 365", "google workspace", "jamf",
    "intune", "okta", "mdm", "servicenow", "zendesk", "itil", "macos", "windows", "powershell",
    "vpn", "dns", "dhcp", "tcp/ip", "endpoint management", "ticketing",
    # systems / other
    "cybersecurity", "security", "networking", "operating systems",
    "multithreading", "concurrency", "optimization", "gpu",
}

_ALNUM = re.compile(r"^[a-z0-9]+$")

# Spellings of the SAME skill. Without this, a posting asking for "OOP" scores as
# missing against a resume that says "Object-Oriented Programming". Each alias is
# folded onto its canonical name, and a skill counts as present if ANY spelling
# of it appears -- and is counted once, not once per spelling.
_SYNONYMS: dict[str, str] = {
    "oop": "object-oriented",
    "postgres": "postgresql",
    "sklearn": "scikit-learn",
    "llms": "llm",
    "natural language processing": "nlp",
    "golang": "go",
    "high performance": "high-performance",
}


def _canon(term: str) -> str:
    return _SYNONYMS.get(term, term)


def _spellings(canonical: str) -> list[str]:
    return [canonical] + [a for a, c in _SYNONYMS.items() if c == canonical]


# Words that are also everyday English and need context to count as the skill.
# "Spring" is the Java framework -- but in "graduating fall 2027 or spring 2028"
# it's a season. Don't count it when a year / academic-term word follows.
_NOT_SKILL_AFTER: dict[str, str] = {
    "spring": r"(?!\s*(?:'?\d{2}\b|\d{4}\b|semester|term|quarter|session|break|season|cohort|intern))",
}


def _contains(text: str, term: str) -> bool:
    if _ALNUM.match(term):
        guard = _NOT_SKILL_AFTER.get(term, "")
        return re.search(r"\b" + re.escape(term) + r"\b" + guard, text) is not None
    return term in text  # multi-word or symbol-bearing: substring is safe


def _has_skill(text: str, canonical: str) -> bool:
    return any(_contains(text, s) for s in _spellings(canonical))


def _spans(text: str, term: str) -> list[tuple[int, int]]:
    if _ALNUM.match(term):
        pat = r"\b" + re.escape(term) + r"\b" + _NOT_SKILL_AFTER.get(term, "")
    else:
        pat = re.escape(term)
    return [(m.start(), m.end()) for m in re.finditer(pat, text)]


# Text allowed between two skills in one list: a comma, "or", ", or", "and/or".
# A list is a set of ALTERNATIVES only when an explicit "or" joins it ("Go,
# Ruby, or Java" -> one requirement, met by any). Plain commas / "and" lists stay
# separate requirements, and so do slashes ("Python/SQL" usually means both).
_LIST_GAP = re.compile(r"\s*(?:,\s*(?:or\b|and/or\b)?|\bor\b|\band/or\b)\s*")
_OR = re.compile(r"\bor\b|and/or")
# A posting block (list item / line) that says its tools can be learned on the
# job: "We work mostly in Java, Ruby, Scala, and Go. We believe new programming
# languages can be learned." Its skills are alternatives, not each required.
_LEARNABLE = re.compile(
    r"can be (?:learned|learnt|picked up)|learn (?:it|them|new [a-z ]+) on the job|"
    r"no prior experience (?:in|with)|(?:don'?t|do not|doesn'?t|does not) need (?:to know|experience)")
_BLOCK = re.compile(r"&lt;/li&gt;|</li>|\n")


def _learnable_blocks(jd: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in list(_BLOCK.finditer(jd)) + [None]:
        end = m.start() if m else len(jd)
        if _LEARNABLE.search(jd, start, end):
            spans.append((start, end))
        start = m.end() if m else end
    return spans


def requirements(job_description: str) -> list[frozenset[str]]:
    """The posting's skill requirements as units. A unit with several members is
    a set of alternatives (any one satisfies it); singletons are individual
    requirements."""
    jd = (job_description or "").lower()
    mentions = [(a, b, _canon(s)) for s in SKILLS for a, b in _spans(jd, s)]
    mentions.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    kept: list[tuple[int, int, str]] = []
    for m in mentions:                       # longest match wins ("spring boot" over "spring")
        if kept and m[0] < kept[-1][1]:
            continue
        kept.append(m)

    singles: set[str] = set()
    groups: list[frozenset[str]] = []
    for a, b in _learnable_blocks(jd):
        inside = frozenset(c for start, _, c in kept if a <= start < b)
        kept = [m for m in kept if not a <= m[0] < b]
        if len(inside) > 1:          # any one of them meets it; a lone one isn't required
            groups.append(inside)
    i = 0
    while i < len(kept):
        chain, gaps = [kept[i]], []
        while i + 1 < len(kept):
            gap = jd[chain[-1][1]:kept[i + 1][0]]
            if len(gap) > 12 or not _LIST_GAP.fullmatch(gap):
                break
            chain.append(kept[i + 1])
            gaps.append(gap)
            i += 1
        last_or = max((k for k, g in enumerate(gaps) if _OR.search(g)), default=None)
        if last_or is None:
            singles.update(c for _, _, c in chain)
        else:
            alt = frozenset(c for _, _, c in chain[:last_or + 2])
            if len(alt) > 1:
                groups.append(alt)
            else:
                singles.update(alt)
            singles.update(c for _, _, c in chain[last_or + 2:])
        i += 1

    units = [frozenset([s]) for s in sorted(singles)]
    for g in groups:
        # A skill that's required on its own elsewhere dominates the group.
        if not (g & singles) and g not in units:
            units.append(g)
    return units


def claimable_missing(missing: list[str], evidence_skills: set) -> list[str]:
    """Missing keywords the candidate can honestly add: singletons present in
    their evidence, and -- for an unmet "a or b" group -- whichever members
    they have."""
    out: list[str] = []
    for entry in missing:
        for k in entry.split(" or "):
            if k in evidence_skills and k not in out:
                out.append(k)
    return out


def _resume_text(resume_data: Any) -> str:
    parts: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
        elif isinstance(obj, str):
            parts.append(obj)

    walk(resume_data)
    return " ".join(parts).lower()


def missing_skills_md(title: str, company: str, apply_url: str, ats: dict,
                      evidence_skills: set, review: dict | None = None) -> str:
    """Per-job missing-skills sheet: ATS match, what the resume already covers,
    and the gaps split into 'you have it -- surface it' vs 'you don't have it yet
    -- consider learning'. Plus the judge's red flags / woven-in keywords."""
    matched = ats.get("matched") or []
    missing = ats.get("missing") or []
    # An "a or b" entry is a gap only if the candidate has none of its members.
    have_but_off = sorted(claimable_missing(missing, evidence_skills))   # rare after the judge pass
    gaps = sorted(k for k in missing if not any(m in evidence_skills for m in k.split(" or ")))
    score = ats.get("score")
    lines = [
        f"# {title} — {company}",
        "",
        f"**Apply:** {apply_url}",
        f"**ATS match:** {score}%" if score is not None else "**ATS match:** n/a",
        "",
        "## Keywords this posting wants that your resume already covers",
        ("- " + ", ".join(matched)) if matched else "- (none)",
        "",
        "## Missing skills — the posting wants these, you don't have them yet",
        "_Consider whether any are worth learning; do NOT add them to the resume unless true._",
    ]
    lines += [f"- {g}" for g in gaps] if gaps else ["- (none — you cover everything named 🎉)"]
    if have_but_off:
        lines += ["", "## In your evidence but not on this resume — surface these",
                  "- " + ", ".join(have_but_off)]
    if review and review.get("keywords_added"):
        lines += ["", f"**Keywords the judge wove in:** {', '.join(review['keywords_added'])}"]
    if review and review.get("red_flags"):
        lines += ["", "## Red flags the recruiter pass caught (and fixed)"]
        lines += [f"- {f}" for f in review["red_flags"]]
    return "\n".join(lines) + "\n"


def skills_in(*texts: str) -> set[str]:
    """The recognized skills present across the given text(s) -- e.g. everything
    the candidate can legitimately claim, from master resume + bank + repos."""
    blob = " ".join(t for t in texts if t).lower()
    return {_canon(s) for s in SKILLS if _contains(blob, s)}


def score_resume(resume_data: Any, job_description: str) -> dict:
    """
    Returns {score: int|None, matched: [str], missing: [str]}.
    score is the % of posting-mentioned skills that also appear in the resume;
    None when the posting names no recognizable skills (can't score fairly).
    """
    resume = _resume_text(resume_data)
    # Requirement units: synonyms folded ("OOP" == "object-oriented"), and an
    # explicit either/or list counted as ONE requirement met by any member.
    units = requirements(job_description)
    if not units:
        return {"score": None, "matched": [], "missing": [], "alternatives": []}
    matched: list[str] = []
    missing: list[str] = []
    met = 0
    for unit in units:
        present = sorted(s for s in unit if _has_skill(resume, s))
        if present:
            met += 1
            matched.extend(p for p in present if p not in matched)
        else:
            missing.append(" or ".join(sorted(unit)))
    return {
        "score": round(100 * met / len(units)),
        "matched": sorted(matched),
        "missing": sorted(missing),
        "alternatives": [sorted(u) for u in units if len(u) > 1],
    }
