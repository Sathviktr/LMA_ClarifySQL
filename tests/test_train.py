"""Training data and validation scoring for the fine-tuning (no PyTorch needed)."""

import json

import pytest

from clarifysql.train.qlora import lr_factor
from clarifysql.train.sft_data import IGNORE, SFT_DIR
from clarifysql.train.state_eval import score_states, selection_key, slot_match


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    from clarifysql.config import MODEL_ID

    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer is not in the local Hugging Face cache")


@pytest.mark.skipif(not (SFT_DIR / "train.jsonl").exists(), reason="fine-tuning data not built")
@pytest.mark.parametrize("name", ["train", "validation", "train_noslot", "validation_noslot"])
def test_every_state_trains_on_its_target_only_in_the_run_time_context(tok, name):
    from clarifysql.llm.tokens import chat_text
    from clarifysql.train.sft_data import encode_all, load_states, prompt_text

    states = load_states(name)
    examples, over = encode_all(tok, states, max_len=1024)  # raises if a prompt's tokens change
    assert not over and len(examples) == len(states)
    for s, ex in zip(states, examples):
        assert tok.decode([t for t in ex.labels if t != IGNORE]) == s["target"] + "<|im_end|>"
    s = states[0]  # the template adds nothing (no "<think>" block): the same prompt as the agent at run time
    assert prompt_text(tok, s["messages"]) == chat_text(s["messages"]) + "<|im_start|>assistant\n"


def test_slot_match_needs_the_same_kind_and_a_term_about_the_same_thing():
    ref = [{"name": "ranking", "term": "city ranks highest"}]
    assert slot_match([{"name": "ranking", "term": "top city", "meanings": []}], ref)
    assert not slot_match([{"name": "threshold", "term": "top city"}], ref)        # wrong kind of detail
    assert not slot_match([{"name": "ranking", "term": "best airport"}], ref)      # about something else
    assert not slot_match([], ref) and not slot_match(None, ref)
    two = ref + [{"name": "threshold", "term": "many airports"}]
    assert not slot_match([{"name": "ranking", "term": "city ranks highest"}], two)  # one of two missing


def _state(target: dict) -> dict:
    return {"id": "x", "target": json.dumps(target)}


def test_state_scores_by_hand():
    clarify = {"action": "clarify", "slots": [{"name": "threshold", "term": "many customers", "meanings": []}],
               "question": "How many?"}
    states = [_state(clarify), _state(clarify), _state({"action": "execute", "request": "r"}),
              _state({"action": "inspect", "tool": "search_definitions", "args": {"query": "q"}})]
    answers = [json.dumps(clarify),                                  # right, with the right slot
               '{"action":"execute","request":"Show it."}',          # wrong decision
               "I would run the query.",                             # not an action
               '{"action":"inspect","tool":"describe_table","args":{"table":"t"}}']  # right action, wrong tool
    s = score_states(states, answers)
    assert s["valid_rate"] == 0.75 and s["action_accuracy"] == 0.5
    assert s["slot_accuracy"] == 0.5 and s["tool_accuracy"] == 0.0
    # decisions: clarify->clarify, clarify->execute, execute->invalid; F1(clarify)=2/3, F1(execute)=0
    assert s["decision_macro_f1"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["confusion"] == {"clarify->clarify": 1, "clarify->execute": 1, "execute->invalid": 1,
                              "inspect->inspect": 1}
    assert score_states(states, answers, with_slots=False)["slot_accuracy"] is None


def test_selection_prefers_decisions_and_lets_slots_break_near_ties():
    a = {"decision_macro_f1": 0.801, "slot_accuracy": 0.5, "valid_rate": 1.0}
    b = {"decision_macro_f1": 0.799, "slot_accuracy": 0.7, "valid_rate": 1.0}
    c = {"decision_macro_f1": 0.75, "slot_accuracy": 0.9, "valid_rate": 1.0}
    assert max([a, b, c], key=selection_key) is b


def test_learning_rate_warms_up_then_decays_to_zero():
    f = [lr_factor(s, total=20, warmup=4) for s in range(20)]
    assert f[0] == 0.25 and f[3] == 1.0 and f[4] == 1.0
    assert all(x >= y for x, y in zip(f[4:], f[5:])) and f[-1] < 0.01
