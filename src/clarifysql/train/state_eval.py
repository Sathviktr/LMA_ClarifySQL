"""Judging a policy on the validation states, without playing whole episodes (checkpoint selection).

For every validation state the policy writes its next action, which is compared with the reference action:
  valid_rate          the answer parses as an action
  action_accuracy     the same action type (inspect / clarify / execute / abstain)
  decision_macro_f1   macro-F1 over clarify / execute / abstain, on the states whose reference is one of them
  slot_accuracy       on the states whose reference is clarify: the policy clarifies and, for every missing detail,
                      names one of the same kind (threshold / ranking) with a term about the same thing
  tool_accuracy       on the states whose reference is inspect: the policy inspects with the same tool
The proposal (Section 9): "A lower training loss alone is not success; we judge by validation decisions and slot
accuracy." The whole agent is run on the validation episodes only for the chosen checkpoints.
"""

from __future__ import annotations

import json
from collections import Counter

from clarifysql.agent.actions import parse_action
from clarifysql.episodes.checks import term_words
from clarifysql.eval.metrics import DECISIONS, macro_f1


def slot_match(predicted: list, reference: list[dict]) -> bool:
    """Every reference slot has a predicted slot of the same kind whose term is about the same thing."""
    def same(p, r) -> bool:
        return (isinstance(p, dict) and p.get("name") == r["name"]
                and bool(term_words(str(p.get("term", ""))) & term_words(r["term"])))
    return bool(predicted) and all(any(same(p, r) for p in predicted) for r in reference)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score_states(states: list[dict], answers: list[str], with_slots: bool = True) -> dict:
    refs = [json.loads(s["target"]) for s in states]
    preds = [parse_action(a)[0] for a in answers]
    kinds = [p["action"] if p else "invalid" for p in preds]
    decision = [(r["action"], k) for r, k in zip(refs, kinds) if r["action"] in DECISIONS]
    f1, per_class = macro_f1([g for g, _ in decision], [k for _, k in decision])
    clarify = [(r, p) for r, p in zip(refs, preds) if r["action"] == "clarify"]
    inspect = [(r, p) for r, p in zip(refs, preds) if r["action"] == "inspect"]
    return {
        "states": len(states),
        "valid_rate": _rate(sum(p is not None for p in preds), len(preds)),
        "action_accuracy": _rate(sum(r["action"] == k for r, k in zip(refs, kinds)), len(refs)),
        "decision_macro_f1": f1,
        "decision_per_class": per_class,
        "slot_accuracy": _rate(sum(p is not None and p["action"] == "clarify" and slot_match(p.get("slots"), r["slots"])
                                   for r, p in clarify), len(clarify)) if with_slots else None,
        "tool_accuracy": _rate(sum(p is not None and p["action"] == "inspect" and p.get("tool") == r["tool"]
                                   for r, p in inspect), len(inspect)),
        "confusion": dict(Counter(f"{r['action']}->{k}" for r, k in zip(refs, kinds))),
    }


def selection_key(scores: dict) -> tuple:
    """Fixed before training: decision macro-F1 first; slot accuracy decides between near-ties (same 0.01)."""
    return round(scores["decision_macro_f1"], 2), scores.get("slot_accuracy") or 0.0, scores["valid_rate"] or 0.0
