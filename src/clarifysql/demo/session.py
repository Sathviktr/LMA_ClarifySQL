"""Advance the existing controller until a real person needs to reply; no simulated user."""

from clarifysql.agent.controller import Controller, Run
from clarifysql.agent.prompts import system_prompt
from clarifysql.db.safe_exec import execute
from clarifysql.episodes.schema import Episode
from clarifysql.episodes.states import memory_line
from clarifysql.demo.university import GLOSSARY


class LiveSession:
    def __init__(self, backend, db_path, request, adapter=True):
        prompt = "\n".join(line for line in system_prompt().splitlines() if '"sample_values"' not in line)
        self.controller = Controller(backend, backend, None, policy_adapter=adapter, policy_system=prompt,
            db_path_of=lambda _: db_path, allowed_tools=frozenset({"describe_table", "search_definitions"}))
        self.run = Run(Episode("live", "demo", "university_demo", "clear", request, glossary=GLOSSARY))
        self.db_path = db_path

    def advance(self, answer=None):
        r = self.run
        if r.pending == "user":
            if not answer or not answer.strip():
                return self.view()
            r.steps[-1]["result"] = answer.strip()
            r.memory.append(memory_line(r.question, answer.strip()))
            r.questions += 1
            r.events.append({"role": "user", "question": r.question, "reply": answer.strip()})
            r.pending = "policy"
        for _ in range(20):
            if r.pending in ("done", "user"):
                break
            self.controller._policy_round([r])
            self.controller._writer_round([r])
        return self.view()

    def view(self):
        r = self.run
        if r.pending == "user":
            message = r.question
        elif r.outcome == "abstained":
            message = r.abstain_text
        elif r.outcome == "executed":
            message = "Query completed." if r.sql_status == "ok" else f"Query was not successful: {r.sql_error or r.sql_status}"
        else:
            message = "The request is still unclear or a session limit was reached. Please start a more specific request."
        result = execute(self.db_path, r.sql) if r.sql and r.sql_status == "ok" else None
        return {"message": message, "sql": r.sql or "", "columns": result.columns if result else [],
                "rows": result.rows if result else [], "pending": r.pending,
                "history": "\n".join(f"Question: {s['action']['question']}\nReply: {s['result'] or '(waiting)'}"
                                     for s in r.steps if s['action']['action'] == 'clarify')}
