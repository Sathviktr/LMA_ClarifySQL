"""Regression checks for resuming, fair pairing, held-out references, and real-dialogue decisions."""

import json
from pathlib import Path
import pytest
from clarifysql.eval.artifacts import RunStore, read_jsonl
from clarifysql.eval.bootstrap import paired_bootstrap
from clarifysql.eval.cosql_slice import run_slice
from clarifysql.eval.metrics import reference_slots
from clarifysql.episodes.schema import load_episodes
from clarifysql.llm.backend import MockBackend


def test_resume_refuses_changed_inputs_duplicates_and_broken_tail(tmp_path):
    store = RunStore(tmp_path / "run", {"data": "v1"})
    store.append([{"episode": "one", "correct": True}])
    assert RunStore(tmp_path / "run", {"data": "v1"}).completed == {"one"}
    with pytest.raises(ValueError, match="changed"):
        RunStore(tmp_path / "run", {"data": "v2"})
    with pytest.raises(ValueError, match="once"):
        store.append([{"episode": "one"}])
    with (tmp_path / "run/traces.jsonl").open("a") as f:
        f.write('{"episode":')
    with pytest.raises(ValueError, match="corrupt"):
        read_jsonl(tmp_path / "run/traces.jsonl")


def test_slot_corrections_change_only_scoring_terms():
    episodes = {ep.id: ep for ep in load_episodes("test")}
    ep = episodes["test-ambiguous-669"]
    before = ep.to_dict()
    assert reference_slots(ep)[0]["term"] == "tall poker players"
    assert ep.to_dict() == before


def test_database_bootstrap_pairs_ids_and_averages_seeds():
    base = [{"episode": str(i), "db_id": f"db{i // 2}", "kind": "clear", "gold_decision": "execute",
             "decision": "execute", "correct": False} for i in range(4)]
    first = [{**r, "correct": r["db_id"] == "db0"} for r in reversed(base)]
    second = [{**r, "correct": True} for r in base]
    result = paired_bootstrap(base, [first, second], "execution_accuracy", repeats=200)
    assert result["system_seeds"] == [0.5, 1.0]
    assert result["difference"] == 0.75 and result["ci95"] == [0.5, 1.0]
    assert result["databases"] == 2
    with pytest.raises(ValueError, match="same episode"):
        paired_bootstrap(base, [first[:-1]], "execution_accuracy")


def test_cosql_context_is_visible_but_review_label_is_hidden(tiny_db, monkeypatch):
    from clarifysql.eval import cosql_slice
    monkeypatch.setattr(cosql_slice.spider, "db_path", lambda _: tiny_db)
    def reply(chats, adapter):
        text = chats[-1]["content"]
        assert "Earlier dialogue" in text and "the French singers" in text
        assert "gold_decision" not in text
        return '{"action":"clarify","question":"Which year?"}'
    backend = MockBackend(reply)
    rows = run_slice([{"episode": "one", "db_id": "toy", "request": "What about 2014?",
                       "context": "user: Show the French singers.", "gold_decision": "clarify"}], backend, "b3")
    assert rows[0]["correct"] and rows[0]["calls"] == 1
    assert len(backend.calls) == 1  # no writer or simulated reply after the first decision


def test_real_slice_sizes_and_mock_run_resume(tmp_path, monkeypatch):
    from clarifysql.config import data_dir
    from clarifysql.eval.cosql_slice import load_slice
    from clarifysql.eval.final_run import run_condition
    assert len(load_slice()) == 59
    if not (data_dir() / "spider").exists():
        pytest.skip("Spider data not available")
    result = run_condition("sft", "validation", tmp_path, backend_name="mock", limit=3, chunk_size=2)
    assert result["complete"] and result["episodes"] == 3
    again = run_condition("sft", "validation", tmp_path, backend_name="mock", limit=3, chunk_size=2)
    assert again == result
    with pytest.raises(ValueError, match="changed"):
        run_condition("sft", "validation", tmp_path, backend_name="mock", limit=4)


def test_simulator_cost_counts_invalid_generated_choice():
    from clarifysql.simuser.simulated_user import SimulatedUser
    backend = MockBackend(lambda chat, adapter: "?")
    reply = SimulatedUser(backend).answer_batch([{"episode": "one", "request": "older singers",
        "question": "What age?", "slots": [{"kind": "threshold", "term": "older singers", "hidden": "above 20",
                                               "value": ">20", "answer": "above 20"}]}])[0]
    assert reply.choice is None and reply.model_calls == 1 and reply.tokens_in > 0
