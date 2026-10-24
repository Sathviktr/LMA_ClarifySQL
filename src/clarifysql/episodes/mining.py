"""Find Spider questions that can be turned into ambiguous episodes (proposal, Section 8.2).

* threshold: a number appears in the question and is used as a cut-off in the gold SQL (with >, <, >=,
  <= or BETWEEN), e.g. "students older than 20" / `age > 20`; removing it ("older students") hides the
  cut-off. Equalities such as `student_id = 171` pick out one record and are not cut-offs.
* ranking: the gold SQL sorts and keeps the top rows (ORDER BY ... LIMIT), e.g. "the largest district"
  could be ranked by population or by area.

Candidates are de-duplicated: at most one source question per (database, normalised gold SQL).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from clarifysql.data.normalize import normalize_sql
from clarifysql.data.spider import Example

_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_CUTOFF = re.compile(r"(?:>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)"
                     r"|\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", re.I)
_RANKING = re.compile(r"\border\s+by\b.+\blimit\s+\d+", re.I | re.S)


def _canon(num: str) -> str:
    value = float(num)
    return str(int(value)) if value.is_integer() else str(value)


def numbers(text: str) -> set[str]:
    return {_canon(n) for n in _NUMBER.findall(text)}


def cutoff_values(sql: str) -> set[str]:
    """Numbers the SQL compares against with >, <, >=, <= or BETWEEN."""
    return {_canon(g) for m in _CUTOFF.finditer(sql) for g in m.groups() if g}


def threshold_values(question: str, sql: str) -> set[str]:
    """Cut-off values of the SQL that the question states explicitly."""
    return numbers(question) & cutoff_values(sql)


def is_threshold(ex: Example) -> bool:
    return bool(threshold_values(ex.question, ex.query))


def is_ranking(ex: Example) -> bool:
    return bool(_RANKING.search(ex.query))


def dedupe(examples: Iterable[Example]) -> list[Example]:
    """Keep the first question for every (database, normalised SQL) pair."""
    seen, out = set(), []
    for ex in examples:
        key = (ex.db_id, normalize_sql(ex.query))
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def candidates(examples: Iterable[Example]) -> dict[str, list[Example]]:
    unique = dedupe(examples)
    return {"threshold": [e for e in unique if is_threshold(e)],
            "ranking": [e for e in unique if is_ranking(e)]}
