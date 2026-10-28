import json
import re

import pytest

from clarifysql.agent.actions import format_action, parse_action
from clarifysql.agent.prompts import render_state, system_prompt
from clarifysql.config import REPO_ROOT, data_dir

SFT = REPO_ROOT / "data" / "sft"


def test_format_and_parse_roundtrip():
    a = {"action": "clarify", "question": "How many customers count as \"many\"?",
         "slots": [{"name": "threshold", "term": "many customers", "meanings": ["number of customers above some value"],
                    "extra": "dropped"}]}
    text = format_action(a)
    assert text.startswith('{"action":"clarify","slots":[') and " " not in text.split('"question"')[0].replace(
        "many customers", "").replace("number of customers above some value", "")
    parsed, err = parse_action("Sure! " + text)
    assert err is None and parsed["slots"][0]["term"] == "many customers"
    assert "slots" not in json.loads(format_action(a, with_slots=False))


@pytest.mark.parametrize("text,error", [
    ("no json here", "no JSON"),
    ('{"action": "drop"}', "must be one of"),
    ('{"action": "inspect", "tool": "rm", "args": {}}', "'tool' must be"),
    ('{"action": "clarify", "question": ""}', "non-empty 'question'"),
    ('{"action": "execute"}', "non-empty 'request'"),
    ('{"action": "abstain", "missing": 3}', "non-empty 'missing'"),
    ('{"action": "clarify", "question": "Which?"}', "non-empty 'slots'"),
])
def test_parse_rejects_invalid_actions(text, error):
    action, err = parse_action(text, require_slots=True)
    assert action is None and error in err


def test_prompt_versions_differ_only_in_the_clarify_format():
    a, b = system_prompt(True), system_prompt(False)
    assert "slots" in a and "slots" not in b
    assert a.splitlines()[0] == b.splitlines()[0] and a.splitlines()[-1] == b.splitlines()[-1]


def test_render_state_shows_budget_and_memory():
    s = render_state("db", "t(a, b)", "Show big ones.", [], ["I mean above 5. (answer to: \"How big?\")"], 1, 2)
    assert "Budget left: 1 inspection, 2 questions." in s and "Details the user has given:" in s


built = pytest.mark.skipif(not (SFT / "train.jsonl").exists(), reason="SFT data not built")


def _rows(name):
    return [json.loads(line) for line in (SFT / name).read_text(encoding="utf-8").splitlines() if line]


@built
def test_sft_sizes_and_action_mix():
    rows = _rows("train.jsonl")
    assert 600 <= len(rows) <= 800  # proposal Section 8.3
    mix = {a: sum(r["target_action"] == a for r in rows) for a in ("inspect", "clarify", "execute", "abstain")}
    assert mix["execute"] > mix["clarify"]  # many clear examples stop the model from asking every time
    assert all(r["tokens"] <= 1024 for r in rows)


@built
def test_targets_parse_and_noslot_targets_have_no_slots():
    for r in _rows("train.jsonl"):
        action, err = parse_action(r["target"], require_slots=r["target_action"] == "clarify")
        assert err is None, (r["id"], err)
    assert all('"slots"' not in r["target"] for r in _rows("train_noslot.jsonl"))


@built
def test_no_leakage_of_the_hidden_intent():
    """The policy's input never contains the gold SQL or the original question, and a slot's hidden value
    appears only after the user has given it (proposal Section 9)."""
    from clarifysql.episodes.schema import load_episodes

    episodes = {ep.id: ep for split in ("train", "validation") for ep in load_episodes(split)}
    for name in ("train.jsonl", "validation.jsonl", "train_noslot.jsonl"):
        for r in _rows(name):
            ep = episodes[r["episode"]]
            visible = " ".join(m["content"] for m in r["messages"])
            if ep.gold_sql:
                assert ep.gold_sql not in visible
            if ep.kind in ("ambiguous", "two_slot", "definition"):
                assert ep.source_question not in visible, r["id"]
            answered = visible.count("   user: ")
            for i, slot in enumerate(ep.slots):
                if ep.kind == "definition" or i < answered:
                    continue  # given by the user (or by the glossary, in a definition episode)
                assert not slot["hidden"] or slot["hidden"] not in visible, (r["id"], slot["hidden"])
                after_catalog = r["messages"][1]["content"].split("Request:")[1].split("Budget left:")[0]
                for num in re.findall(r"-?\d+(?:\.\d+)?", slot["value"]):
                    assert not re.search(rf"(?<![\w.]){re.escape(num)}(?![\w.])", after_catalog), (r["id"], num)


@built
@pytest.mark.skipif(not (data_dir() / "spider").exists(), reason="benchmark data not downloaded")
def test_states_follow_the_reference_path():
    from clarifysql.episodes.schema import load_episodes
    from clarifysql.episodes.states import episode_states

    ep = next(e for e in load_episodes("train") if e.kind == "two_slot")
    states = episode_states(ep)
    assert [s["target_action"] for s in states] == ["inspect", "clarify", "clarify", "execute"]
    assert "user:" in states[2]["messages"][1]["content"] and "Budget left: 2 inspections, 1 question." in \
        states[2]["messages"][1]["content"]
