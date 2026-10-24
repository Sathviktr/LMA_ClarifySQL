"""Execution match: is a predicted query correct? Compare *results*, not SQL text (proposal, Section 10).

Rules:
  * row order matters only when the gold query sorts at its outermost level (ORDER BY outside brackets),
  * otherwise rows are compared as a multiset: same rows, same number of copies, any order,
  * columns may come in a different order (SELECT a, b vs SELECT b, a), as in the test-suite evaluator,
  * NULL equals NULL; numbers are equal if they agree after rounding to 4 decimals (5 == 5.0),
  * text is compared exactly (case matters).
A prediction that fails to run (error, blocked, timeout) is incorrect.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from clarifysql.db.safe_exec import ExecResult, execute, strip_comments_and_strings

MAX_PERMUTED_COLUMNS = 6


def _cell(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        r = round(float(v), 4)
        return int(r) if r.is_integer() else r
    return v


def _rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_cell(v) for v in row) for row in rows]


def order_matters(gold_sql: str) -> bool:
    """True if the gold SQL has ORDER BY at bracket depth 0 (sorting inside a subquery does not count)."""
    skeleton = strip_comments_and_strings(gold_sql).lower()
    depth = 0
    for i, ch in enumerate(skeleton):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and skeleton.startswith("order by", i):
            return True
    return False


def results_match(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    pred, gold = _rows(pred), _rows(gold)
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    width = len(gold[0])
    if any(len(r) != width for r in pred):
        return False

    def same(p_rows):
        return p_rows == gold if ordered else Counter(p_rows) == Counter(gold)

    if same(pred):
        return True
    if width > MAX_PERMUTED_COLUMNS:
        return False
    for perm in itertools.permutations(range(width)):
        if same([tuple(r[j] for j in perm) for r in pred]):
            return True
    return False


@dataclass
class MatchResult:
    correct: bool
    reason: str  # "match" | "different_result" | "pred_<status>" | "gold_<status>"
    pred: ExecResult
    gold: ExecResult


def exec_match(db: str | Path, pred_sql: str, gold_sql: str) -> MatchResult:
    gold = execute(db, gold_sql)
    pred = execute(db, pred_sql)
    if not gold.ok:
        return MatchResult(False, f"gold_{gold.status}", pred, gold)
    if not pred.ok:
        return MatchResult(False, f"pred_{pred.status}", pred, gold)
    ok = results_match(pred.rows, gold.rows, order_matters(gold_sql))
    return MatchResult(ok, "match" if ok else "different_result", pred, gold)
