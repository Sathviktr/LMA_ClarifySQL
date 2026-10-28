"""Make a clear Spider question vague by removing one detail, and describe the removed detail as a slot.

The gold SQL is kept unchanged as the *hidden intent*: it is what the user really means (proposal, 8.2).

* threshold: "heads older than 56" -> "older heads"; the slot is the cut-off (age > 56)
* ranking:   "the user who has the largest number of followers" -> "the user who ranks highest";
             the slot is the measure used for ranking (number of followers)

Rule drafts are deliberately simple. Training drafts are polished by the model (notebook 03) and checked
automatically; validation and test drafts are edited by the team (review sheets), so the tuned model is
never tested on its own writing style.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from clarifysql.episodes.mining import _canon, cutoff_values

NUMERIC_TYPES = ("int", "real", "num", "float", "double", "decimal", "bigint", "smallint")
UNIT = r"(?:\s+(?:years?(?:\s+old)?|dollars?|%|percent|points?|km|kg|cm|meters?|metres?|hours?|minutes?|days?|times?))?"
NOT_NOUNS = {"and", "or", "but", "if", "then", "not", "per", "yet", "so", "in", "of", "on", "at", "by", "for", "with",
             "from", "to", "the", "a", "an", "who", "which", "that", "whose", "is", "are", "was", "were", "times", "than"}
ADJECTIVES = {"high", "low", "large", "big", "small", "long", "short", "tall", "heavy"}
LINKING = {"is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "with", "than", "of", "by",
           "the", "a", "an", "and", "or", "but", "not", "who", "which", "that", "whose", "if", "where", "when"}


@dataclass
class Slot:
    kind: str                # "threshold" | "ranking"
    target: str              # the SQL expression the slot controls, e.g. "age" or "count(*)"
    hidden: str              # the words removed from the question, e.g. "older than 56"
    value: str               # what the user means, e.g. "> 56" or "number of followers"
    vague: str               # what replaced the words in the request, e.g. "older"
    meanings: list[str] = field(default_factory=list)  # competing interpretations the policy may list
    answer: str = ""         # what the simulated user replies when asked about this slot

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ threshold
# (comparator words, SQL operator, replacement if a noun follows the number, replacement otherwise)
_COMPARATORS = [
    (r"older than", ">", "older", "older"),
    (r"younger than", "<", "younger", "younger"),
    (r"longer than", ">", "long", "long"),
    (r"shorter than", "<", "short", "short"),
    (r"taller than", ">", "tall", "tall"),
    (r"heavier than", ">", "heavy", "heavy"),
    (r"(?:more|greater|bigger|larger|higher) than or equal to", ">=", "many", "high"),
    (r"(?:less|fewer|smaller|lower) than or equal to", "<=", "few", "low"),
    (r"at least|no less than|not less than", ">=", "many", "high"),
    (r"at most|no more than|not more than", "<=", "few", "low"),
    (r"more than|over|exceeding|exceeds", ">", "many", "high"),
    (r"greater than|larger than", ">", "many", "large"),
    (r"bigger than", ">", "many", "big"),
    (r"higher than|above", ">", "many", "high"),
    (r"fewer than|less than", "<", "few", "low"),
    (r"smaller than", "<", "few", "small"),
    (r"lower than|below|under", "<", "few", "low"),
]
_YEAR_COMPARATORS = [
    (r"after|later than|since", ">", "recently"),
    (r"before|earlier than|prior to", "<", "long ago"),
]


def _is_year(num: str) -> bool:
    try:
        return 1800 <= float(num) <= 2100 and float(num).is_integer()
    except ValueError:
        return False


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?.,])", r"\1", text)
    text = re.sub(r"\b(a|an) (many|few)\b", r"\2", text)
    return text[0].upper() + text[1:] if text else text


def threshold_draft(question: str, sql: str, column_of: dict[str, str] | None = None) -> tuple[str, Slot] | None:
    """Replace the comparison phrase around one stated cut-off by a vague word.

    Returns (vague question, slot) or None if no rule matches. `column_of` maps cut-off values to the SQL
    column they are compared with (for the slot's target); it is filled automatically when omitted.
    """
    values = sorted(v for v in cutoff_values(sql) if v in {_canon(n) for n in re.findall(r"-?\d+(?:\.\d+)?", question)})
    column_of = column_of or _columns_for_values(sql)
    for v in values:
        num = rf"(?P<num>{re.escape(v)}(?:\.0+)?)"
        if _is_year(v):
            for words, op, vague in _YEAR_COMPARATORS:
                m = re.search(rf"\b(?P<cmp>{words})\s+(?:the\s+year\s+)?{num}\b", question, re.I)
                if m:
                    return _make_threshold(question, m, v, op, vague, column_of.get(v, "?"))
        between = re.search(rf"\bbetween\s+{num}\s+and\s+(?P<num2>-?\d+(?:\.\d+)?){UNIT}", question, re.I)
        if between:
            hidden = between.group(0)
            vague_q = _tidy(question[:between.start()] + "within a certain range" + question[between.end():])
            col = column_of.get(v, "?")
            slot = Slot("threshold", col, hidden, f"between {v} and {_canon(between.group('num2'))}",
                        "within a certain range", [f"{col} between two values (not given)"],
                        f"I mean {hidden}.")
            return vague_q, slot
        for words, op, with_noun, without_noun in _COMPARATORS:
            m = re.search(rf"\b(?P<cmp>{words})\s+(?:the\s+)?{num}\b(?P<unit>{UNIT})", question, re.I)
            if not m:
                continue
            nxt = re.match(r"\s+([A-Za-z]+)", question[m.end():])
            noun_follows = bool(nxt) and nxt.group(1).lower() not in NOT_NOUNS and not m.group("unit")
            vague = with_noun if noun_follows else without_noun
            return _make_threshold(question, m, v, op, vague, column_of.get(v, "?"))
    return None


def _make_threshold(question, m, value, op, vague, column) -> tuple[str, Slot]:
    hidden = m.group(0).strip()
    before, after = question[:m.start()], question[m.end():]
    prev = re.search(r"([A-Za-z_]+)\s*$", before)
    shown = vague
    if vague in ADJECTIVES and prev and prev.group(1).lower() not in LINKING:
        before = before[:prev.start()] + f"{vague} {prev.group(1)} "  # "population bigger than 1500" -> "big population"
        shown = ""
    vague_q = _tidy(before + shown + after)
    slot = Slot("threshold", column, hidden, f"{op} {value}", vague,
                [f"{column} {op} some value (not given)"], f"I mean {hidden}.")
    return vague_q, slot


def _columns_for_values(sql: str) -> dict[str, str]:
    out = {}
    for m in re.finditer(r"([\w.\"`\[\]()*]+)\s*(?:>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", sql):
        out.setdefault(_canon(m.group(2)), m.group(1).strip('"`[]'))
    for m in re.finditer(r"([\w.\"`\[\]]+)\s+between\s+(-?\d+(?:\.\d+)?)", sql, re.I):
        out.setdefault(_canon(m.group(2)), m.group(1).strip('"`[]'))
    return out


# -------------------------------------------------------------------------------------------------- ranking
_HIGH = r"most|largest|greatest|biggest|highest|maximum|max"
_LOW = r"least|fewest|lowest|smallest|minimum|min"
_MEASURE = r"(?:(?:total\s+)?(?:number|amount|count)\s+of\s+|average\s+|total\s+)?(?:[A-Za-z_]+\s+){0,2}?[A-Za-z_]+"
_STOP = r"(?=\s+(?:in|among|for|from|of\s+all|whose|who|that|which|and|with|during|at|on|per|ever)\b|[?.,;]|$)"


def order_target(sql: str) -> tuple[str, str] | None:
    """(expression, direction) of the outermost ORDER BY ... LIMIT."""
    m = re.search(r"order\s+by\s+(.+?)(?:\s+(asc|desc))?\s+limit\s+\d+", sql, re.I | re.S)
    if not m:
        return None
    return m.group(1).strip(), (m.group(2) or "asc").lower()


def ranking_draft(question: str, sql: str, measures: list[str]) -> tuple[str | None, Slot] | None:
    """Hide the ranking measure. Returns (draft or None if no rule matched, slot), or None if not ranking."""
    tgt = order_target(sql)
    if not tgt:
        return None
    expr, direction = tgt
    word = "highest" if direction == "desc" else "lowest"
    pat = rf"\b(?:has|have|had|got|with|owns?|contains?)\s+(?:the\s+)?(?:{_HIGH}|{_LOW})\s+{_MEASURE}{_STOP}"
    m = re.search(pat, question, re.I)
    draft, hidden = None, ""
    if m:
        hidden = m.group(0).strip()
        lead = "that ranks" if re.match(r"\s*with\b", m.group(0), re.I) else "ranks"
        draft = _tidy(question[:m.start()] + f" {lead} {word}" + question[m.end():])
    value = _measure_words(hidden) or _readable(expr)
    slot = Slot("ranking", expr, hidden, value, f"ranks {word}", sorted(set(measures)), f"Rank them by {value}.")
    return draft, slot


def _readable(expr: str) -> str:
    """'count(*)' -> 'number of matching rows'; 'T2.horsepower' -> 'horsepower'."""
    if re.search(r"count\s*\(", expr, re.I):
        return "number of records"
    inner = re.sub(r"^(?:avg|sum|max|min)\s*\((.*)\)$", r"\1", expr.strip(), flags=re.I)
    return re.sub(r"[_\s]+", " ", inner.split(".")[-1]).strip().lower()


def _measure_words(hidden: str) -> str:
    return re.sub(rf"^(?:has|have|had|got|with|owns?|contains?)\s+(?:the\s+)?(?:{_HIGH}|{_LOW})\s+",
                  "", hidden, flags=re.I).strip()


def candidate_measures(sql: str, schema) -> list[str]:
    """Numeric, non-identifier columns of the tables in the query, plus 'number of records' for counts.

    These are the competing meanings of "ranks highest": by which of them? The column the query orders by is
    always one of them, even when it is stored as text (car_1 keeps horsepower and mpg as text).
    """
    used = {t.name for t in schema.tables.values() if re.search(rf"\b{re.escape(t.name)}\b", sql, re.I)}
    ordered = " ".join(re.findall(r"order\s+by\s+([^;]*)", sql, re.I)).lower()
    out = []
    for name in used:
        for c in schema.tables[name].columns:
            low = c.name.lower()
            measure = any(k in c.type for k in NUMERIC_TYPES) or re.search(rf"\b{re.escape(low)}\b", ordered)
            if measure and not (low == "id" or low.endswith("_id") or low.endswith("id") or low.endswith("_code")):
                out.append(f"{name}.{c.name}")
    if re.search(r"count\s*\(", sql, re.I):
        out.append("number of records")
    return sorted(set(out))
