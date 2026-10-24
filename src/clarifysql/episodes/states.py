"""Cut episodes into conversation states: the policy's training examples (proposal, Section 9).

Each step of an episode's reference path becomes one example:
    input  = the policy prompt for the state *before* that step (request, catalog, earlier steps with their
             real tool results and the user's answers, remaining budget)
    target = the reference action, as compact JSON
Tool results are produced by running the real tools on the real database and the episode's glossary, so
the policy trains on exactly the text it will see at test time.
"""

from __future__ import annotations

from clarifysql.agent.actions import format_action
from clarifysql.agent.prompts import budgets_left, policy_messages, render_state
from clarifysql.data import spider
from clarifysql.db.catalog import compact_catalog
from clarifysql.episodes.schema import Episode
from clarifysql.tools import ToolContext, run_tool
from clarifysql.tools.search_definitions import Definition


def memory_line(question: str, answer: str) -> str:
    """One entry of the session memory: the user's answer, linked to the question it answers."""
    return f'{answer} (answer to: "{question}")'


def episode_states(ep: Episode, with_slots: bool = True) -> list[dict]:
    db = spider.db_path(ep.db_id)
    glossary = [Definition(g["term"], g["definition"], tuple(g.get("aliases", []))) for g in ep.glossary]
    ctx = ToolContext(db, glossary)
    catalog = compact_catalog(db)
    steps: list[dict] = []
    memory: list[str] = []
    answers = iter(ep.slots)
    out = []
    for k, action in enumerate(ep.reference):
        inspections_left, questions_left = budgets_left(steps)
        state = render_state(ep.db_id, catalog, ep.request, steps, memory, inspections_left, questions_left)
        out.append({
            "id": f"{ep.id}#{k}", "episode": ep.id, "kind": ep.kind, "db_id": ep.db_id, "step": k,
            "messages": policy_messages(state, with_slots),
            "target": format_action(action, with_slots), "target_action": action["action"],
        })
        if action["action"] == "inspect":
            steps.append({"action": action, "result": run_tool(ctx, action["tool"], action["args"]).text})
        elif action["action"] == "clarify":
            answer = next(answers)["answer"]
            steps.append({"action": action, "result": answer})
            memory.append(memory_line(action["question"], answer))
        else:
            break
    return out
