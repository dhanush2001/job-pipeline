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
    # ml inference / performance (nano-serve territory)
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
    # systems / other
    "cybersecurity", "security", "networking", "operating systems",
    "multithreading", "concurrency", "optimization", "gpu",
}

_ALNUM = re.compile(r"^[a-z0-9]+$")


def _contains(text: str, term: str) -> bool:
    if _ALNUM.match(term):
        return re.search(r"\b" + re.escape(term) + r"\b", text) is not None
    return term in text  # multi-word or symbol-bearing: substring is safe


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
    have_but_off = sorted(k for k in missing if k in evidence_skills)   # rare after the judge pass
    gaps = sorted(k for k in missing if k not in evidence_skills)
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
    return {s for s in SKILLS if _contains(blob, s)}


def score_resume(resume_data: Any, job_description: str) -> dict:
    """
    Returns {score: int|None, matched: [str], missing: [str]}.
    score is the % of posting-mentioned skills that also appear in the resume;
    None when the posting names no recognizable skills (can't score fairly).
    """
    jd = (job_description or "").lower()
    resume = _resume_text(resume_data)
    jd_skills = {s for s in SKILLS if _contains(jd, s)}
    if not jd_skills:
        return {"score": None, "matched": [], "missing": []}
    matched = sorted(s for s in jd_skills if _contains(resume, s))
    missing = sorted(s for s in jd_skills if not _contains(resume, s))
    return {
        "score": round(100 * len(matched) / len(jd_skills)),
        "matched": matched,
        "missing": missing,
    }
