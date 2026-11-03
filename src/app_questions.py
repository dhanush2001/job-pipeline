"""
app_questions.py -- pull a posting's application-form questions and draft answers.

Greenhouse and Ashby expose the full application form publicly (the same data
their apply pages render). Each question is sorted into one of five kinds:

  contact  -- name, email, phone, links: filled from config `candidate`
  file     -- resume / cover letter uploads: points at the generated PDFs
  profile  -- work authorization, sponsorship, start date, location ...: filled
              ONLY from `application_profile` in config.yaml, never by a model
  manual   -- legal acknowledgements, EEO self-identification, unmatched
              yes/no and pick-list questions: left for you to answer
  written  -- open-ended questions ("What engineering work excites you?"):
              drafted by the model from your resume evidence

The result is a Markdown sheet saved next to each job's resume.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from src import llm_client
from src.tailor_resume import _strip_fences, _thinking_params

logger = logging.getLogger(__name__)

TIMEOUT = 20
_ASHBY_GQL = "https://jobs.ashbyhq.com/api/non-user-graphql?op=ApiJobPosting"
_ASHBY_QUERY = """query ApiJobPosting($organizationHostedJobsPageName: String!, $jobPostingId: String!) {
  jobPosting(organizationHostedJobsPageName: $organizationHostedJobsPageName, jobPostingId: $jobPostingId) {
    applicationForm { sections { fieldEntries { ... on FormFieldEntry { isRequired descriptionHtml field } } } }
  }
}"""


@dataclass
class Question:
    label: str
    kind_hint: str  # board field type: String, LongText, Boolean, textarea, input_text, ...
    required: bool = False
    options: list[str] = field(default_factory=list)
    description: str = ""
    kind: str = ""  # contact | file | profile | manual | written
    answer: str = ""
    note: str = ""


# ---- fetching ---------------------------------------------------------------------
def _plain(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(text or "")).split())


def fetch_greenhouse(board: str, job_id: str) -> list[Question]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}?questions=true"
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    out = []
    for q in resp.json().get("questions") or []:
        fields = q.get("fields") or []
        types = [f.get("type", "") for f in fields]
        options = [v.get("label", "") for f in fields for v in (f.get("values") or [])]
        # "Resume/CV" offers input_file + textarea; it's still an upload.
        hint = "input_file" if "input_file" in types else (types[0] if types else "")
        out.append(Question(label=_plain(q.get("label", "")), kind_hint=hint,
                            required=bool(q.get("required")), options=options,
                            description=_plain(q.get("description") or "")))
    return out


def fetch_ashby(org: str, job_id: str) -> list[Question]:
    resp = requests.post(_ASHBY_GQL, timeout=TIMEOUT, json={
        "operationName": "ApiJobPosting", "query": _ASHBY_QUERY,
        "variables": {"organizationHostedJobsPageName": org, "jobPostingId": job_id},
    })
    resp.raise_for_status()
    posting = ((resp.json().get("data") or {}).get("jobPosting")) or {}
    out = []
    for section in (posting.get("applicationForm") or {}).get("sections") or []:
        for entry in section.get("fieldEntries") or []:
            f = entry.get("field") or {}
            if not f.get("title"):
                continue
            out.append(Question(label=_plain(f["title"]), kind_hint=f.get("type", ""),
                                required=bool(entry.get("isRequired")),
                                options=[v.get("label", "") for v in (f.get("selectableValues") or [])],
                                description=_plain(entry.get("descriptionHtml") or "")))
    return out


_GH_URL = re.compile(r"greenhouse\.io/(?:embed/job_app\?for=)?([^/?#&]+)/jobs/(\d+)", re.I)
_ASHBY_URL = re.compile(r"ashbyhq\.com/([^/?#]+)/([0-9a-f-]{36})", re.I)


def board_ref(job: Any) -> tuple[str, str, str] | None:
    """(board, org, posting id) for a job whose form we can read, else None.
    Uses the scanner's id first, then any Greenhouse/Ashby link (LinkedIn and
    Adzuna results often point at one)."""
    parts = (job.id or "").split(":")
    if len(parts) == 3 and parts[0] in ("greenhouse", "ashby"):
        return parts[0], parts[1], parts[2]
    for link in (job.apply_url, job.url):
        if m := _GH_URL.search(link or ""):
            return "greenhouse", m.group(1), m.group(2)
        if m := _ASHBY_URL.search(link or ""):
            return "ashby", m.group(1), m.group(2)
    return None


def fetch_questions(job: Any) -> list[Question] | None:
    """The posting's application questions; None when the board isn't supported
    or the fetch failed (the sheet then says to open the form)."""
    ref = board_ref(job)
    if not ref:
        return None
    board, org, job_id = ref
    try:
        return fetch_greenhouse(org, job_id) if board == "greenhouse" else fetch_ashby(org, job_id)
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Application questions unavailable for %s @ %s: %s", job.title, job.company, exc)
        return None


# ---- sorting ----------------------------------------------------------------------
_FILE_TYPES = {"file", "input_file"}
_TEXT_TYPES = {"string", "longtext", "textarea", "input_text"}
_CONTACT = [
    (re.compile(r"^(legal )?first name", re.I), "first_name"),
    (re.compile(r"^(legal )?last name", re.I), "last_name"),
    (re.compile(r"^(full |legal )?name$|^legal name", re.I), "full_name"),
    (re.compile(r"^preferred (first )?name", re.I), "preferred_name"),
    (re.compile(r"^e-?mail", re.I), "email"),
    (re.compile(r"^phone", re.I), "phone"),
    (re.compile(r"linkedin", re.I), "linkedin_url"),
    (re.compile(r"github", re.I), "github_url"),
    (re.compile(r"^(website|portfolio|personal (site|website))", re.I), "website"),
]
_EEO = re.compile(r"\b(gender|race|racial|ethnic|hispanic|latino|veteran|disabilit|pronoun|sexual orientation|"
                  r"transgender|lgbt)", re.I)
_LEGAL = re.compile(r"\b(acknowledg|arbitration|privacy (notice|policy)|i (hereby )?certify|consent|"
                    r"i agree|terms (and|&) conditions|i confirm|attest)", re.I)
# Optional free-text boxes that aren't a question ("Additional Information").
_FILLER = re.compile(r"^(additional information|anything else|other|comments?)\b", re.I)


def classify(q: Question, profile: list[dict]) -> None:
    hint = (q.kind_hint or "").lower()
    if hint in _FILE_TYPES:
        q.kind = "file"
        return
    for pat, key in _CONTACT:
        if pat.search(q.label):
            q.kind, q.note = "contact", key
            return
    if _EEO.search(q.label):
        q.kind, q.note = "manual", "Voluntary self-identification -- your choice."
        return
    if _LEGAL.search(q.label):
        q.kind, q.note = "manual", "Legal acknowledgement -- read it and tick it yourself."
        return
    for entry in profile or []:
        if entry.get("match") and re.search(entry["match"], q.label, re.I):
            q.kind = "profile"
            q.answer = str(entry.get("answer") or "").strip()
            if not q.answer:
                q.note = f"Add your answer under application_profile (match: {entry['match']!r}) in config.yaml."
            return
    # A factual yes/no or pick-list about you is never the model's to guess.
    if hint not in _TEXT_TYPES or q.options:
        q.kind, q.note = "manual", "Personal fact or choice -- answer it yourself."
        return
    if not q.required and (_FILLER.search(q.label) or "?" not in q.label):
        q.kind, q.note = "manual", "Optional -- leave blank unless you have something to add."
        return
    q.kind = "written"


def _contact_value(key: str, candidate: dict, master: dict) -> str:
    first, last = candidate.get("first_name", ""), candidate.get("last_name", "")
    values = {
        "first_name": first, "last_name": last, "full_name": f"{first} {last}".strip(),
        "preferred_name": candidate.get("preferred_name", ""),
        "email": candidate.get("email", ""), "phone": candidate.get("phone", ""),
        "linkedin_url": candidate.get("linkedin_url", ""),
        "github_url": candidate.get("github_url") or (f"github.com/{master['github_handle']}"
                                                      if master.get("github_handle") else ""),
        "website": candidate.get("website", ""),
    }
    return values.get(key, "")


# ---- drafting ---------------------------------------------------------------------
ANSWER_SYSTEM_PROMPT = """You draft a candidate's answers to the written questions
on one job application, in the candidate's own first-person voice.

Rules:
- Use ONLY facts in `candidate_evidence` (resume, GitHub repos, bullet bank). Name
  real projects, tools and results from it. Never invent a project, employer,
  metric, preference, or experience; copy any number exactly as the evidence has it.
- Projects in `in_progress_projects` are not finished: describe them as ongoing.
- Tie each answer to this company and role using the posting's own language.
- Open-ended questions: 90-170 words unless the question sets a length. Short
  fields (a language, a single preference) get a short, direct answer.
- Plain, specific, confident prose. No clichés ("passionate", "fast-paced",
  "I believe I would be a great fit"), no bullet lists, no markdown.
- For "what excites you / what doesn't" questions, the "doesn't" half must be
  honest but constructive (e.g. work with no feedback loop), never a dig at a
  team or a task this role clearly requires.
- If a question needs a fact the evidence doesn't hold (visa status, salary,
  availability, relocation, location, personal circumstances), return "" for it.

Respond with ONLY this JSON:
{"answers": [{"id": <question id>, "answer": "..."}]}"""


def draft_written_answers(questions: list[Question], master: dict, github_repos: list | None,
                          bullet_bank: str, title: str, company: str, description: str,
                          api_key: str, model: str, reasoning: str = "disabled",
                          temperature: float = 0.4, max_tokens: int | None = None) -> None:
    """Fill `answer` on every `written` question with one model call per job."""
    todo = [(i, q) for i, q in enumerate(questions) if q.kind == "written"]
    if not todo or not api_key:
        return
    from src.guards import in_progress_projects

    user = {
        "candidate_evidence": {"master_resume": master, "github_repos": github_repos or [],
                               "bullet_bank": bullet_bank or ""},
        "in_progress_projects": [p.get("name", "") for p in in_progress_projects(master)],
        "job": {"title": title, "company": company, "description": _plain(description)[:6000]},
        "questions": [{"id": i, "question": q.label, "help_text": q.description[:500],
                       "required": q.required} for i, q in todo],
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(user)}],
        "response_format": {"type": "json_object"},
    }
    payload.update(_thinking_params(reasoning, temperature))
    if max_tokens:
        payload["max_tokens"] = max_tokens
    result = llm_client.chat_json(payload, api_key, label=f"questions {company} / {title}")
    if not result.ok:
        for _, q in todo:
            q.note = "Draft failed -- write this one yourself."
        return
    try:
        parsed = json.loads(_strip_fences(result.content))
    except json.JSONDecodeError:
        parsed = {}
    by_id = {a.get("id"): str(a.get("answer") or "").strip()
             for a in (parsed.get("answers") or []) if isinstance(a, dict)}
    for i, q in todo:
        q.answer = by_id.get(i, "")
        if not q.answer:
            q.note = "Needs a fact your resume doesn't hold -- write this one yourself."


def answer_application(job: Any, master: dict, candidate: dict, profile: list[dict],
                       github_repos: list | None = None, bullet_bank: str = "",
                       api_key: str = "", model: str = "", **llm_kwargs) -> list[Question] | None:
    """Fetch, sort and answer one job's application form. None = form unavailable."""
    questions = fetch_questions(job)
    if questions is None:
        return None
    for q in questions:
        classify(q, profile)
        if q.kind == "contact":
            q.answer = _contact_value(q.note, candidate, master)
            q.note = "" if q.answer else "Not in config `candidate` -- fill it in."
    if api_key and model:
        draft_written_answers(questions, master, github_repos, bullet_bank, job.title, job.company,
                              job.description, api_key, model, **llm_kwargs)
    return questions


# ---- output -----------------------------------------------------------------------
_SECTIONS = [
    ("written", "Written answers (drafted from your resume -- read before pasting)"),
    ("profile", "About you (from application_profile in config.yaml)"),
    ("manual", "Answer these yourself"),
    ("contact", "Contact details"),
    ("file", "Uploads"),
]


def render_markdown(job: Any, questions: list[Question] | None, unsupported_numbers: list[str] = ()) -> str:
    head = [f"# Application questions -- {job.company} · {job.title}", "", f"Apply: <{job.apply_url}>", ""]
    if questions is None:
        return "\n".join(head + ["This board's form can't be read automatically (only Greenhouse and "
                                 "Ashby are supported) -- open the link to see its questions.", ""])
    if not questions:
        return "\n".join(head + ["The form has no questions beyond the standard upload.", ""])
    if unsupported_numbers:
        head += [f"> **Check before sending:** these numbers aren't in your sources: "
                 f"{', '.join(unsupported_numbers)}", ""]
    lines = head
    for kind, title in _SECTIONS:
        group = [q for q in questions if q.kind == kind]
        if not group:
            continue
        lines += [f"## {title}", ""]
        for q in group:
            req = " *(required)*" if q.required else ""
            if kind == "file":
                upload = "cover_letter.pdf" if "cover" in q.label.lower() else "resume.pdf"
                lines += [f"- **{q.label}**{req} -- upload `{upload}` from this folder."]
                continue
            lines += [f"**{q.label}**{req}", ""]
            if q.options and kind != "written":
                lines += ["Options: " + " / ".join(o[:80] for o in q.options), ""]
            if q.answer:
                lines += [q.answer, ""]
            if q.note:
                lines += [f"_{q.note}_", ""]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
