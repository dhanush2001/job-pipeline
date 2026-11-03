"""
Tests for the separate project-selection step (select_projects) and its
enforcement on the tailored package (guards.enforce_projects). The model call
is stubbed -- offline, no spend.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.tailor_resume as tr
from src.guards import enforce_projects

MASTER = {"projects": [
    {"name": "tiny-serve: LLM Inference Server", "technologies": "Python, CUDA", "bullets": ["Building a KV cache"]},
    {"name": "MedVQA: Medical Data Processing & Serving Platform", "technologies": "PyTorch, Transformers",
     "bullets": ["Served a vision-language model"]},
    {"name": "SynthEval: Data Validation & Generation Pipeline", "bullets": ["DDPM data pipeline"]},
]}
REPOS = [{"name": "matrix_engine", "language": "C", "description": "pthreads linear algebra"}]


def _stub(content, ok=True):
    def fake(payload, api_key, label=""):
        fake.payload = payload
        return SimpleNamespace(ok=ok, content=json.dumps(content) if ok else None)
    return fake


def _select(content, ok=True):
    orig = tr.llm_client.chat_json
    tr.llm_client.chat_json = fake = _stub(content, ok)
    try:
        return tr.select_projects(MASTER, REPOS, "Performance Engineer", "Acme", "CUDA, C++, transformers",
                                  api_key="k"), fake
    finally:
        tr.llm_client.chat_json = orig


def test_candidates_include_master_and_github():
    names = [c["name"] for c in tr.project_candidates(MASTER, REPOS)]
    assert names[:3] == [p["name"] for p in MASTER["projects"]] and names[3] == "matrix_engine"
    print("test_candidates_include_master_and_github: PASS")


def test_picks_are_matched_to_real_names_and_capped():
    out, fake = _select({"projects": [
        {"name": "tiny-serve", "why": "inference serving"},                 # shortened -> matched
        {"name": "matrix_engine", "why": "HPC kernels"},
        {"name": "Totally Made Up Project", "why": "?"},                    # not a candidate -> ignored
        {"name": "tiny-serve: LLM Inference Server", "why": "dup"},         # duplicate -> ignored
        {"name": "MedVQA", "why": "transformers"},
        {"name": "SynthEval", "why": "over the cap"},
    ]})
    names, reasons = out
    assert names == ["tiny-serve: LLM Inference Server", "matrix_engine",
                     "MedVQA: Medical Data Processing & Serving Platform"], names
    assert len(reasons) == 3
    # Reasoning is ON for the picker, and the role is in the request.
    assert fake.payload.get("thinking") == {"type": "enabled"} and fake.payload.get("reasoning_effort") == "high"
    assert "Performance Engineer" in fake.payload["messages"][1]["content"]
    print("test_picks_are_matched_to_real_names_and_capped: PASS")


def test_failure_or_garbage_returns_none():
    assert _select({}, ok=False)[0] is None
    assert _select({"projects": [{"name": "Nope"}]})[0] is None
    assert _select({"projects": "not a list"})[0] is None
    print("test_failure_or_garbage_returns_none: PASS")


def test_enforce_orders_and_filters():
    pkg = SimpleNamespace(projects=[{"name": "SynthEval", "bullets": ["x"]},
                                    {"name": "matrix_engine", "bullets": ["y"]},
                                    {"name": "tiny-serve: LLM Inference Server", "bullets": ["z"]}])
    notes = enforce_projects(pkg, ["tiny-serve: LLM Inference Server", "matrix_engine",
                                   "MedVQA: Medical Data Processing & Serving Platform"])
    assert [p["name"] for p in pkg.projects] == ["tiny-serve: LLM Inference Server", "matrix_engine"]
    assert any("removed" in n and "SynthEval" in n for n in notes)
    assert any("missing" in n and "MedVQA" in n for n in notes)
    # Tailoring ignored the selection entirely -> keep its picks, report it.
    pkg2 = SimpleNamespace(projects=[{"name": "SynthEval", "bullets": ["x"]}])
    assert "ignored" in enforce_projects(pkg2, ["tiny-serve: LLM Inference Server"])[0]
    assert [p["name"] for p in pkg2.projects] == ["SynthEval"]
    print("test_enforce_orders_and_filters: PASS")


if __name__ == "__main__":
    test_candidates_include_master_and_github()
    test_picks_are_matched_to_real_names_and_capped()
    test_failure_or_garbage_returns_none()
    test_enforce_orders_and_filters()
    print("\nAll project-selection tests passed.")
