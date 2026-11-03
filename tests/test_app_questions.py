"""Application-question sorting, drafting and the answer sheet -- no network."""

import json
from types import SimpleNamespace

import yaml

from src import app_questions as aq
from src import llm_client

PROFILE = yaml.safe_load(open("config.yaml"))["application_profile"]
CANDIDATE = {"first_name": "Jane", "last_name": "Doe", "email": "d@x.edu", "phone": "555"}
JOB = SimpleNamespace(id="ashby:openai:55150071-fce8-48f5-aea4-14ed78b83511", title="SWE, Applied Emerging Talent",
                      company="openai", apply_url="https://jobs.ashbyhq.com/openai/x", url="", description="Build products.")


def _q(label, hint="String", required=True, options=()):
    q = aq.Question(label=label, kind_hint=hint, required=required, options=list(options))
    aq.classify(q, PROFILE)
    return q


def test_questions_are_sorted_by_who_should_answer():
    assert _q("What type of engineering work excites you? What does not excite you?", "LongText").kind == "written"
    assert _q("Resume", "File").kind == "file"
    assert _q("Legal Name").kind == "contact"
    # Facts about you never go to the model: profile if configured, else manual.
    assert _q("Will you now or in the future require sponsorship?", "Boolean").kind == "profile"
    assert _q("Are you a former intern here?", "Boolean").kind == "manual"
    assert _q("Pick your favourite team", "ValueSelect", options=["A", "B"]).kind == "manual"
    assert _q("Applicant Arbitration Agreement Acknowledgement", "MultiValueSelect", options=["I acknowledge"]).kind == "manual"
    assert _q("What is your gender identity?", "ValueSelect").kind == "manual"
    assert _q("Additional Information", "LongText", required=False).kind == "manual"


def test_location_answer_never_lands_on_a_yes_no_relocation_question():
    roblox = ("This role requires you to work a hybrid schedule (3+ days in office) at our San Mateo, CA HQ. Will "
              "this schedule work for you and are you currently located or willing to relocate to San Francisco?")
    q = _q(roblox, "multi_value_single_select", options=["Yes", "No"])
    assert q.kind == "profile" and q.answer != "Anytown, USA"
    assert _q("Where are you currently located?", "Location").answer == "Anytown, USA"


def test_board_ref_reads_scanner_ids_and_links():
    assert aq.board_ref(JOB) == ("ashby", "openai", "55150071-fce8-48f5-aea4-14ed78b83511")
    li = SimpleNamespace(id="apify:123", apply_url="https://job-boards.greenhouse.io/roblox/jobs/8171283", url="")
    assert aq.board_ref(li) == ("greenhouse", "roblox", "8171283")
    assert aq.board_ref(SimpleNamespace(id="lever:x:y", apply_url="https://jobs.lever.co/x/y", url="")) is None


def test_written_answers_drafted_and_sheet_rendered(monkeypatch):
    form = [aq.Question("Legal Name", "String", True), aq.Question("Resume", "File", True),
            aq.Question("What type of engineering work excites you? What does not excite you?", "LongText", True),
            aq.Question("Are you authorized to work in the country where the job is located?", "Boolean", True),
            aq.Question("Applicant Arbitration Agreement Acknowledgement", "MultiValueSelect", True, ["I acknowledge"])]
    monkeypatch.setattr(aq, "fetch_questions", lambda job: [aq.Question(**vars(q)) for q in form])
    sent = {}

    def fake_chat(payload, key, label=""):
        sent["user"] = json.loads(payload["messages"][1]["content"])
        return llm_client.CallResult(json.dumps({"answers": [{"id": 2, "answer": "Shipping tools people use."}]}), "ok")

    monkeypatch.setattr(llm_client, "chat_json", fake_chat)
    qs = aq.answer_application(JOB, {"name": "D"}, CANDIDATE, PROFILE, api_key="k", model="m")
    # Only the written question is sent to the model.
    assert [q["question"] for q in sent["user"]["questions"]] == [form[2].label]
    md = aq.render_markdown(JOB, qs, ["42%"])
    assert "Shipping tools people use." in md and "Jane Doe" in md
    assert "upload `resume.pdf`" in md and "42%" in md
    assert md.index("Written answers") < md.index("About you") < md.index("Answer these yourself")
    assert "application_profile" in md  # blank work-authorization answer is flagged, not guessed


def test_unsupported_board_says_open_the_form():
    assert "can't be read automatically" in aq.render_markdown(JOB, None)
