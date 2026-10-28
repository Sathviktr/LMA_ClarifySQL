"""The controller: plain Python that runs the agent loop and enforces every limit (proposal, Section 7).

Episodes are run in lockstep, one turn at a time, so each round needs only one batched model call per role
(policy, simulated user, SQL writer): batching multiplies throughput (pilot, Phase 1).

Per episode:
  1. the policy (adapter on for the tuned systems, off for B3) proposes ONE action as JSON;
     an invalid answer gets one format retry with the error message, for every system alike;
  2. inspect -> the controller checks the arguments and runs the tool; clarify -> the simulated user answers
     and the answer is stored in the session memory; abstain -> stop;
  3. execute -> the SQL writer (adapter off) turns the rewritten request into SQL; the controller runs it
     read-only and checks that values the user gave appear in it; an SQL error or a missing value triggers a
     repair (at most 2);
  4. limits: at most 3 inspections, 2 questions, 2 repairs and 8 agent model calls (policy + SQL writer).
     When a limit would be exceeded the episode ends as "limit" instead of guessing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from clarifysql.agent.actions import format_action, parse_action
from clarifysql.agent.prompts import policy_messages, render_state
from clarifysql.config import DEFAULT_BUDGETS, Budgets
from clarifysql.data import spider
from clarifysql.db.catalog import compact_catalog, create_table_text
from clarifysql.db.safe_exec import execute
from clarifysql.episodes.schema import Episode
from clarifysql.episodes.states import memory_line
from clarifysql.llm.prompts import extract_sql, repair_messages, sql_writer_messages
from clarifysql.tools import ToolContext, run_tool
from clarifysql.tools.search_definitions import Definition

POLICY_MAX_TOKENS = 192
WRITER_MAX_TOKENS = 256


@dataclass
class Run:
    ep: Episode
    steps: list[dict] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    pending: str = "policy"  # policy | user | writer | done
    calls: int = 0
    inspections: int = 0
    questions: int = 0
    repairs: int = 0
    format_error: str | None = None
    format_retries: int = 0
    invalid_outputs: int = 0
    decision: str | None = None      # first non-inspect action (clarify / execute / abstain)
    outcome: str | None = None       # executed | abstained | limit | invalid
    question: str = ""
    request: str = ""                # the rewritten request handed to the SQL writer
    sql: str | None = None
    sql_status: str | None = None
    sql_error: str | None = None
    repair_reason: str | None = None
    abstain_text: str | None = None
    answered: set = field(default_factory=set)
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    blocked_attempts: int = 0
    events: list[dict] = field(default_factory=list)


def user_values(memory_answers: list[str]) -> set[str]:
    """Numbers the user stated in answers; the final SQL must use them (intent check)."""
    return {n for a in memory_answers for n in re.findall(r"-?\d+(?:\.\d+)?", a)}


class Controller:
    def __init__(self, policy_backend, writer_backend, user, *, with_slots: bool = True, policy_adapter: bool = True,
                 policy_system: str | None = None, budgets: Budgets = DEFAULT_BUDGETS, full_context: bool = False,
                 batch_size: int = 16):
        self.policy_backend = policy_backend
        self.writer_backend = writer_backend
        self.user = user
        self.with_slots = with_slots
        self.policy_adapter = policy_adapter
        self.policy_system = policy_system
        self.budgets = budgets
        self.full_context = full_context
        self.batch_size = batch_size
        self._catalog: dict[str, str] = {}
        self._schema: dict[str, str] = {}

    # ---------------------------------------------------------------------------------------- helpers
    def catalog(self, db_id: str) -> str:
        if db_id not in self._catalog:
            path = spider.db_path(db_id)
            self._catalog[db_id] = create_table_text(path) if self.full_context else compact_catalog(path)
        return self._catalog[db_id]

    def schema(self, db_id: str) -> str:
        if db_id not in self._schema:
            self._schema[db_id] = create_table_text(spider.db_path(db_id))
        return self._schema[db_id]

    def policy_prompt(self, r: Run) -> list[dict]:
        ins_left = self.budgets.max_inspections - r.inspections
        q_left = self.budgets.max_questions - r.questions
        catalog = self.catalog(r.ep.db_id)
        if self.full_context and r.ep.glossary:  # ablation A3: every definition in the prompt
            catalog += "\n\nGlossary:\n" + "\n".join(f"{g['term']}: {g['definition']}" for g in r.ep.glossary)
        state = render_state(r.ep.db_id, catalog, r.ep.request, r.steps, r.memory, ins_left, q_left)
        if r.format_error:
            state += (f"\n(Your previous answer was not a valid action: {r.format_error}. "
                      "Answer with ONE JSON object.)")
        msgs = policy_messages(state, self.with_slots)
        if self.policy_system:
            msgs[0] = {"role": "system", "content": self.policy_system}
        return msgs

    def _finish(self, r: Run, outcome: str) -> None:
        r.outcome, r.pending = outcome, "done"

    # ---------------------------------------------------------------------------------------- rounds
    def _policy_round(self, runs: list[Run]) -> None:
        need = [r for r in runs if r.pending == "policy"]
        if not need:
            return
        for r in need:
            if r.calls >= self.budgets.max_model_calls:
                self._finish(r, "limit")
        need = [r for r in need if r.pending == "policy"]
        gens = self.policy_backend.generate([self.policy_prompt(r) for r in need], max_new_tokens=POLICY_MAX_TOKENS,
                                            adapter=self.policy_adapter, batch_size=self.batch_size)
        for r, g in zip(need, gens):
            r.calls += 1
            r.tokens_in += g.prompt_tokens
            r.tokens_out += g.new_tokens
            r.seconds += g.seconds
            action, err = parse_action(g.text, require_slots=False)
            r.events.append({"role": "policy", "raw": g.text, "error": err})
            if err:
                r.invalid_outputs += 1
                if r.format_retries < 1:
                    r.format_retries += 1
                    r.format_error = err
                else:
                    self._finish(r, "invalid")
                continue
            r.format_error = None
            self._apply_action(r, action)

    def _apply_action(self, r: Run, action: dict) -> None:
        kind = action["action"]
        if kind != "inspect" and r.decision is None:
            r.decision = kind
        if kind == "inspect":
            if r.inspections >= self.budgets.max_inspections:
                self._finish(r, "limit")
                return
            glossary = [Definition(g["term"], g["definition"], tuple(g.get("aliases", []))) for g in r.ep.glossary]
            res = run_tool(ToolContext(spider.db_path(r.ep.db_id), glossary), action["tool"], action["args"])
            r.inspections += 1
            r.steps.append({"action": action, "result": res.text})
        elif kind == "clarify":
            if r.questions >= self.budgets.max_questions:
                self._finish(r, "limit")
                return
            r.question = action["question"].strip()
            r.steps.append({"action": {"action": "clarify", "question": r.question,
                                       "slots": action.get("slots", [])}, "result": None})
            r.pending = "user"
        elif kind == "execute":
            r.request = action["request"].strip()
            r.pending = "writer"
        else:  # abstain
            r.abstain_text = action["missing"].strip()
            self._finish(r, "abstained")

    def _user_round(self, runs: list[Run]) -> None:
        need = [r for r in runs if r.pending == "user"]
        if not need:
            return
        # definition episodes: the meaning is in the glossary, so the user has nothing to add
        items = [{"episode": r.ep.id, "request": r.ep.request, "question": r.question,
                  "slots": r.ep.slots if r.ep.kind != "definition" else []} for r in need]
        replies = self.user.answer_batch(items, batch_size=self.batch_size)
        for r, rep in zip(need, replies):
            r.answered.update(rep.slot_indices)
            r.steps[-1]["result"] = rep.text
            r.memory.append(memory_line(r.question, rep.text))
            r.questions += 1
            r.events.append({"role": "user", "question": r.question, "choice": rep.choice, "reply": rep.text})
            r.pending = "policy"

    def _writer_round(self, runs: list[Run]) -> None:
        need = [r for r in runs if r.pending == "writer"]
        for r in need:
            if r.calls >= self.budgets.max_model_calls:
                self._finish(r, "limit")
        need = [r for r in need if r.pending == "writer"]
        if not need:
            return
        chats = []
        for r in need:
            if r.repair_reason and r.sql is not None:
                chats.append(repair_messages(self.schema(r.ep.db_id), r.request, r.sql, r.repair_reason))
            else:
                chats.append(sql_writer_messages(self.schema(r.ep.db_id), r.request))
        gens = self.writer_backend.generate(chats, max_new_tokens=WRITER_MAX_TOKENS, adapter=False,
                                            batch_size=self.batch_size)
        for r, g in zip(need, gens):
            r.calls += 1
            r.tokens_in += g.prompt_tokens
            r.tokens_out += g.new_tokens
            r.seconds += g.seconds
            r.sql = extract_sql(g.text)
            res = execute(spider.db_path(r.ep.db_id), r.sql)
            r.sql_status, r.sql_error = res.status, res.error
            if res.status == "blocked":
                r.blocked_attempts += 1
            missing = sorted(v for v in user_values([s["result"] or "" for s in r.steps
                                                       if s["action"]["action"] == "clarify"])
                             if not re.search(rf"(?<![\w.]){re.escape(v)}(?![\w.])", r.sql))
            reason = None
            if res.status == "sql_error":
                reason = res.error
            elif res.ok and missing:
                reason = f"the query must use the value(s) the user gave: {', '.join(missing)}"
            r.events.append({"role": "writer", "sql": r.sql, "status": res.status, "error": res.error,
                             "intent_missing": missing})
            if reason and r.repairs < self.budgets.max_repairs:
                r.repairs += 1
                r.repair_reason = reason
                continue  # stays pending on the writer
            self._finish(r, "executed")

    # -------------------------------------------------------------------------------------------- run
    def run(self, episodes: list[Episode], max_rounds: int = 20) -> list[Run]:
        runs = [Run(ep) for ep in episodes]
        t0 = time.perf_counter()
        for _ in range(max_rounds):
            if all(r.pending == "done" for r in runs):
                break
            self._policy_round(runs)
            self._user_round(runs)
            self._writer_round(runs)
        for r in runs:
            if r.pending != "done":
                self._finish(r, "limit")
        self.wall_seconds = time.perf_counter() - t0
        return runs


def trace(r: Run) -> dict:
    """Everything about one episode run, for scoring and error analysis (no hidden fields are used here)."""
    return {
        "episode": r.ep.id, "kind": r.ep.kind, "db_id": r.ep.db_id, "request": r.ep.request,
        "decision": r.decision or "none", "gold_decision": r.ep.first_decision, "outcome": r.outcome,
        "steps": [{"action": format_action(s["action"], with_slots=True) if s["action"]["action"] != "clarify"
                   else s["action"].get("question"), "kind": s["action"]["action"], "result": s["result"]}
                  for s in r.steps],
        "rewritten_request": r.request or None, "sql": r.sql, "sql_status": r.sql_status, "sql_error": r.sql_error,
        "abstain_text": r.abstain_text, "calls": r.calls, "inspections": r.inspections, "questions": r.questions,
        "repairs": r.repairs, "invalid_outputs": r.invalid_outputs, "format_retries": r.format_retries,
        "blocked_attempts": r.blocked_attempts, "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
        "seconds": round(r.seconds, 3), "events": r.events,
    }
