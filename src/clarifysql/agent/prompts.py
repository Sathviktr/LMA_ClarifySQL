"""The policy's prompt: fixed instructions plus a rendering of the current conversation state.

The state contains only what the policy may see (proposal, Section 9): the request, the compact catalog of
table and column names, the earlier steps with their real results, the details the user has given so far
(the session memory) and the remaining budget. The hidden intent, future replies and gold SQL never appear.
"""

from __future__ import annotations

import json

from clarifysql.config import DEFAULT_BUDGETS

POLICY_SYSTEM_SLOTS = """You are the clarification policy of a careful database assistant.
Each turn, choose exactly ONE next action and answer with ONE JSON object, nothing else.
Actions:
{"action":"inspect","tool":"describe_table","args":{"table":T}}  columns, keys and row count of a table
{"action":"inspect","tool":"sample_values","args":{"table":T,"column":C}}  frequent values of a column
{"action":"inspect","tool":"search_definitions","args":{"query":Q}}  look up what a vague phrase means
{"action":"clarify","slots":[{"name":K,"term":P,"meanings":[M,...]}],"question":Q}  first list the missing detail (K: threshold or ranking), the vague phrase P and its possible meanings, then ask ONE short question
{"action":"execute","request":R}  the request is fully clear: restate it with every detail resolved, for the SQL writer
{"action":"abstain","missing":M}  the database does not store what is needed: say what is missing
Rules: ask only when a needed detail is missing and no stored definition settles it; never ask about a detail the user already gave; if the data does not exist, abstain instead of asking; keep questions short."""

POLICY_SYSTEM_NOSLOTS = POLICY_SYSTEM_SLOTS.replace(
    '{"action":"clarify","slots":[{"name":K,"term":P,"meanings":[M,...]}],"question":Q}  first list the missing '
    'detail (K: threshold or ranking), the vague phrase P and its possible meanings, then ask ONE short question',
    '{"action":"clarify","question":Q}  ask the user ONE short question about the missing detail')


B3_EXAMPLES = """

Examples (on a different, made-up database: student(id, name, age, dept_id), dept(id, name, budget)):
1. Request: List the names of students older than 20.
   -> {"action":"execute","request":"List the names of students older than 20."}
2. Request: Show the older students.
   -> {"action":"inspect","tool":"search_definitions","args":{"query":"older students"}}
   result: no stored definition matches "older students"
   -> {"action":"clarify","slots":[{"name":"threshold","term":"older students","meanings":["age above some value (which?)"]}],"question":"What age counts as \\"older\\"?"}
   user: I mean older than 25.
   -> {"action":"execute","request":"Show the students older than 25."}
3. Request: Which department ranks highest?
   -> {"action":"inspect","tool":"search_definitions","args":{"query":"department ranks highest"}}
   result: department ranks highest: ranked by budget
   -> {"action":"execute","request":"Which department has the highest budget?"}
4. Request: What is the email address of each student?
   -> {"action":"inspect","tool":"describe_table","args":{"table":"student"}}
   result: table student (40 rows): id int PK, name text, age int, dept_id int
   -> {"action":"abstain","missing":"The database has no email addresses of students."}"""


def b3_system_prompt(with_slots: bool = True) -> str:
    """The prompted baseline B3: the same instructions plus worked examples (tuned on validation only)."""
    examples = B3_EXAMPLES
    if not with_slots:
        examples = examples.replace(
            '{"action":"clarify","slots":[{"name":"threshold","term":"older students","meanings":["age above some '
            'value (which?)"]}],"question":', '{"action":"clarify","question":')
    return system_prompt(with_slots) + examples


def system_prompt(with_slots: bool = True) -> str:
    return POLICY_SYSTEM_SLOTS if with_slots else POLICY_SYSTEM_NOSLOTS


def render_step(step: dict) -> str:
    """One earlier step: the action the policy took and what came back."""
    a = step["action"]
    if a["action"] == "inspect":
        head = f"inspect {a['tool']} {json.dumps(a['args'], ensure_ascii=False)}"
        return f"{head}\n   result: {step['result']}"
    if a["action"] == "clarify":
        return f"clarify: {a['question']}\n   user: {step['result']}"
    return f"{a['action']}: {step.get('result', '')}"


def render_state(db_id: str, catalog: str, request: str, steps: list[dict], memory: list[str],
                 inspections_left: int, questions_left: int) -> str:
    lines = [f"Database: {db_id}", "Tables:", catalog, "", f"Request: {request}"]
    if steps:
        lines += ["", "Steps so far:"] + [f"{i}. {render_step(s)}" for i, s in enumerate(steps, 1)]
    if memory:
        lines += ["", "Details the user has given:"] + [f"- {m}" for m in memory]
    ins = f"{inspections_left} inspection" + ("" if inspections_left == 1 else "s")
    que = f"{questions_left} question" + ("" if questions_left == 1 else "s")
    lines += ["", f"Budget left: {ins}, {que}.", "Next action:"]
    return "\n".join(lines)


def policy_messages(state_text: str, with_slots: bool = True) -> list[dict]:
    return [{"role": "system", "content": system_prompt(with_slots)}, {"role": "user", "content": state_text}]


def budgets_left(steps: list[dict]) -> tuple[int, int]:
    used_inspect = sum(s["action"]["action"] == "inspect" for s in steps)
    used_clarify = sum(s["action"]["action"] == "clarify" for s in steps)
    return (DEFAULT_BUDGETS.max_inspections - used_inspect, DEFAULT_BUDGETS.max_questions - used_clarify)
