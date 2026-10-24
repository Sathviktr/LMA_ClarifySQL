"""The policy's four actions as small JSON records (proposal, Section 7).

    {"action": "inspect", "tool": "search_definitions", "args": {"query": "many customers"}}
    {"action": "clarify", "slots": [{"name": "threshold", "term": "many customers", "meanings": [...]}],
     "question": "How many customers count as \"many\"?"}
    {"action": "execute", "request": "Find the policy type used by more than 4 customers."}
    {"action": "abstain", "missing": "The database has no information about ..."}

In the slot version (our main adapter) a clarify action lists the missing slots and their possible meanings
*before* the question (RQ2). The no-slot version (ablation A2) writes only the question. Outputs are compact
(no spaces after separators) because generated tokens are the expensive part of a call (pilot, Phase 1).
"""

from __future__ import annotations

import json
import re

ACTIONS = ("inspect", "clarify", "execute", "abstain")
TOOLS = ("describe_table", "sample_values", "search_definitions")
_KEY_ORDER = {"inspect": ("action", "tool", "args"), "clarify": ("action", "slots", "question"),
              "execute": ("action", "request"), "abstain": ("action", "missing")}


def format_action(action: dict, with_slots: bool = True) -> str:
    """Canonical compact JSON for an action (keys in a fixed order)."""
    kind = action["action"]
    keys = [k for k in _KEY_ORDER[kind] if k != "slots" or with_slots]
    out = {}
    for k in keys:
        if k == "slots":
            out[k] = [{"name": s["name"], "term": s["term"], "meanings": s["meanings"]} for s in action.get("slots", [])]
        else:
            out[k] = action[k]
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def parse_action(text: str, require_slots: bool = False) -> tuple[dict | None, str | None]:
    """Read the first JSON object in a model answer and check it is a valid action.

    Returns (action, None) or (None, error message). Tool arguments are checked later, against the schema.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no JSON object found"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(obj, dict) or obj.get("action") not in ACTIONS:
        return None, f"'action' must be one of {', '.join(ACTIONS)}"
    kind = obj["action"]
    if kind == "inspect":
        if obj.get("tool") not in TOOLS:
            return None, f"'tool' must be one of {', '.join(TOOLS)}"
        if not isinstance(obj.get("args"), dict):
            return None, "'args' must be an object"
    elif kind == "clarify":
        if not isinstance(obj.get("question"), str) or not obj["question"].strip():
            return None, "a clarify action needs a non-empty 'question'"
        slots = obj.get("slots")
        if require_slots and not (isinstance(slots, list) and slots):
            return None, "a clarify action needs a non-empty 'slots' list"
        if slots is not None and not (isinstance(slots, list) and all(isinstance(s, dict) for s in slots)):
            return None, "'slots' must be a list of objects"
    elif kind == "execute":
        if not isinstance(obj.get("request"), str) or not obj["request"].strip():
            return None, "an execute action needs a non-empty 'request'"
    elif kind == "abstain":
        if not isinstance(obj.get("missing"), str) or not obj["missing"].strip():
            return None, "an abstain action needs a non-empty 'missing'"
    return obj, None
