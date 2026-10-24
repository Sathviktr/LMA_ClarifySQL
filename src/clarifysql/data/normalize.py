"""Canonical SQL text, used to find questions that reword the same query.

normalize_sql lower-cases keywords and names, collapses whitespace, removes spaces around punctuation,
writes every string literal with single quotes and drops a trailing semicolon. String literal contents
keep their case, because 'Paris' and 'paris' are different values. On the Spider dev set this gives
551 distinct (database, query) pairs among 1,034 questions, as stated in the proposal (Section 8.1).
"""

from __future__ import annotations

import re

_LITERAL = re.compile(r"('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")")
_PUNCT = re.compile(r"\s*([(),=<>!+\-/])\s*")


def _code(part: str) -> str:
    part = re.sub(r"\s+", " ", part.lower())
    return _PUNCT.sub(r"\1", part)


def _literal(part: str) -> str:
    if part.startswith('"'):
        inner = part[1:-1].replace('""', '"').replace("'", "''")
        return f"'{inner}'"
    return part


def normalize_sql(sql: str) -> str:
    pieces = _LITERAL.split(sql.strip().rstrip(";").strip())
    out = [(_literal(p) if i % 2 else _code(p)) for i, p in enumerate(pieces)]
    return "".join(out).strip()
