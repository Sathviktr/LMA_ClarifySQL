"""Prompts for the frozen SQL writer, and extraction of the SQL from its answer.

The SQL writer is the base model with the adapter switched off. It receives the (rewritten, fully specified)
request and the database schema, and must answer with one SQLite query. The same prompt is used by every
system (B1, B2, B3, ours, oracle), so differences between systems never come from the SQL writer.
Prompts may only be tuned on validation data (proposal, Section 11).
"""

from __future__ import annotations

import re

SQL_WRITER_SYSTEM = (
    "You translate requests about a SQLite database into one SQLite query.\n"
    "Rules: use only tables and columns that appear in the schema; write exactly one SELECT query "
    "(WITH is allowed); do not explain anything; put the query in a ```sql code block."
)


def sql_writer_messages(schema: str, request: str) -> list[dict]:
    user = f"Database schema:\n{schema}\n\nRequest: {request}"
    return [{"role": "system", "content": SQL_WRITER_SYSTEM}, {"role": "user", "content": user}]


def repair_messages(schema: str, request: str, previous_sql: str, error: str) -> list[dict]:
    """Second chance after SQLite rejected the query (used by B2 and the agents, at most twice)."""
    msgs = sql_writer_messages(schema, request)
    msgs.append({"role": "assistant", "content": f"```sql\n{previous_sql}\n```"})
    msgs.append({"role": "user", "content": f"SQLite returned an error: {error}\nWrite a corrected query."})
    return msgs


_FENCE = re.compile(r"```(?:sqlite|sql)?\s*\n?(.*?)```", re.I | re.S)
_START = re.compile(r"\b(SELECT|WITH)\b", re.I)


def extract_sql(text: str) -> str:
    """The first SQL statement in a model answer: inside a ```sql block if present, else from SELECT/WITH."""
    m = _FENCE.search(text)
    body = m.group(1) if m else text
    if not m:
        start = _START.search(body)
        body = body[start.start():] if start else body
    return _first_statement(body).strip().rstrip(";").strip()


def _first_statement(sql: str) -> str:
    quote = None
    for i, ch in enumerate(sql):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == ";":
            return sql[:i]
    return sql
