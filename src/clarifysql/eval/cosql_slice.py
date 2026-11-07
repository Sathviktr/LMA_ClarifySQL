"""First non-inspect decision on reviewed real dialogues; never generate a user reply or SQL."""

from __future__ import annotations

from clarifysql.agent.actions import parse_action
from clarifysql.agent.prompts import b3_system_prompt, policy_messages, render_state
from clarifysql.config import DEFAULT_BUDGETS
from clarifysql.data import spider
from clarifysql.db.catalog import compact_catalog
from clarifysql.episodes.review import read_sheet, cell
from clarifysql.eval.metrics import macro_f1
from clarifysql.tools import ToolContext, run_tool


def load_slice() -> list[dict]:
    out = []
    for row in read_sheet("cosql_dev_slice.csv"):
        label = cell(row, "cleaned_decision").lower()
        if label == "drop":
            continue
        if label not in ("execute", "clarify", "abstain") or not cell(row, "annotator"):
            raise ValueError("CoSQL slice has an unsigned or incomplete reviewed label")
        out.append({"episode": row["turn_id"], "db_id": row["db_id"], "request": row["user_turn"],
                    "context": row["earlier_turns"], "gold_decision": label})
    return out


def run_slice(items: list[dict], backend, system: str, batch_size: int = 16) -> list[dict]:
    if system not in ("b3", "sft"):
        raise ValueError("CoSQL protocol compares b3 and sft only")
    runs = [{**item, "decision": None, "steps": [], "calls": 0, "invalid_outputs": 0,
             "format_retries": 0, "format_error": None, "tokens_in": 0, "tokens_out": 0,
             "seconds": 0.0, "outcome": None} for item in items]
    catalogs = {r["db_id"]: compact_catalog(spider.db_path(r["db_id"])) for r in runs}
    for _ in range(DEFAULT_BUDGETS.max_model_calls):
        active = [r for r in runs if r["outcome"] is None]
        if not active:
            break
        chats = []
        for r in active:
            state = render_state(r["db_id"], catalogs[r["db_id"]], r["request"], r["steps"],
                                 ["Earlier dialogue (context only):\n" + r["context"]],
                                 DEFAULT_BUDGETS.max_inspections - len(r["steps"]), 2)
            if r["format_error"]:
                state += "\nPrevious output invalid: " + r["format_error"] + ". Return one JSON action."
            chat = policy_messages(state)
            if system == "b3":
                chat[0]["content"] = b3_system_prompt()
            chats.append(chat)
        gens = backend.generate(chats, max_new_tokens=192, adapter=system == "sft", batch_size=batch_size)
        for r, g in zip(active, gens):
            r["calls"] += 1
            r["tokens_in"] += g.prompt_tokens
            r["tokens_out"] += g.new_tokens
            r["seconds"] += g.seconds
            action, error = parse_action(g.text, require_slots=False)
            r.setdefault("events", []).append({"raw": g.text, "error": error})
            if error:
                r["invalid_outputs"] += 1
                if r["format_retries"]:
                    r["outcome"] = "invalid"
                else:
                    r["format_retries"] += 1
                    r["format_error"] = error
                continue
            r["format_error"] = None
            if action["action"] != "inspect":
                r["decision"], r["outcome"] = action["action"], "decided"
            elif len(r["steps"]) >= DEFAULT_BUDGETS.max_inspections:
                r["outcome"] = "limit"
            else:
                result = run_tool(ToolContext(spider.db_path(r["db_id"]), []), action["tool"], action["args"])
                r["steps"].append({"action": action, "result": result.text})
    for r in runs:
        r["decision"] = r["decision"] or "none"
        r["outcome"] = r["outcome"] or "limit"
        r["correct"] = r["decision"] == r["gold_decision"]
    return runs


def summarize_slice(rows: list[dict]) -> dict:
    f1, per = macro_f1([r["gold_decision"] for r in rows], [r["decision"] for r in rows])
    return {"turns": len(rows), "decision_macro_f1": f1, "decision_per_class": per,
            "decision_accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
            "note": "Only one reviewed abstention; abstention estimates are not reliable on this slice.",
            "calls": sum(r["calls"] for r in rows), "tokens_in": sum(r["tokens_in"] for r in rows),
            "tokens_out": sum(r["tokens_out"] for r in rows), "seconds": sum(r["seconds"] for r in rows)}
