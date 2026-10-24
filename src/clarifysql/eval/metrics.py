"""Scoring agent runs (proposal, Section 12).

Primary metrics
  (1) execution accuracy on answerable episodes (clear, ambiguous, two-slot, definition); a failed run or a
      wrong abstention counts as incorrect
  (2) decision macro-F1 over clarify / execute / abstain (the first non-inspect action)
  (3) unnecessary-question rate on clear requests (and, separately, on definition episodes)
  (4) correct-abstention rate on unanswerable requests: abstained AND named the missing information
  (5) average number of clarification turns
plus the invalid-output rate, budget overruns, blocked (unsafe) attempts and cost.
"""

from __future__ import annotations

import re
from collections import Counter

from clarifysql.data import spider
from clarifysql.episodes.schema import Episode
from clarifysql.eval.exec_match import exec_match

ANSWERABLE = ("clear", "ambiguous", "two_slot", "definition")
DECISIONS = ("clarify", "execute", "abstain")


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "").lower().replace("_", " ")):
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out - {"the", "of", "a", "an", "and", "or", "is", "are", "no", "not", "information", "about", "store",
                  "database", "has", "does", "what", "this", "question", "needs"}


def names_missing(abstain_text: str | None, ep: Episode) -> bool:
    """Does the abstention name what the database lacks? (word overlap with the missing attribute/column)."""
    if not abstain_text or not ep.missing:
        return False
    target = ep.missing.get("attribute") or ""
    if ep.draft.get("method") == "trustsql":  # templates like "Highschooler.Capacity" -> the column part
        cols = re.findall(r"\.(\w+)", target)
        target = " ".join(cols) if cols else target
    return bool(_words(target) & _words(abstain_text))


def score_trace(t: dict, ep: Episode) -> dict:
    """Add `correct` (and the reason) to one trace."""
    if ep.kind == "unanswerable":
        ok = t["outcome"] == "abstained" and names_missing(t["abstain_text"], ep)
        reason = "abstained, named" if ok else ("abstained, not named" if t["outcome"] == "abstained" else t["outcome"])
        return {**t, "correct": ok, "reason": reason}
    if t["outcome"] != "executed" or not t["sql"]:
        return {**t, "correct": False, "reason": t["outcome"] or "no outcome"}
    m = exec_match(spider.db_path(ep.db_id), t["sql"], ep.gold_sql)
    return {**t, "correct": m.correct, "reason": m.reason}


def macro_f1(gold: list[str], pred: list[str], labels=DECISIONS) -> tuple[float, dict]:
    per = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0, "support": tp + fn}
    present = [c for c in labels if per[c]["support"]]
    return (round(sum(per[c]["f1"] for c in present) / len(present), 4) if present else 0.0), per


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def summarize(scored: list[dict]) -> dict:
    by_kind = Counter(t["kind"] for t in scored)
    answerable = [t for t in scored if t["kind"] in ANSWERABLE]
    unans = [t for t in scored if t["kind"] == "unanswerable"]
    clear = [t for t in scored if t["kind"] == "clear"]
    definition = [t for t in scored if t["kind"] == "definition"]
    vague = [t for t in scored if t["kind"] in ("ambiguous", "two_slot")]
    f1, per = macro_f1([t["gold_decision"] for t in scored], [t["decision"] for t in scored])
    policy_calls = sum(sum(e["role"] == "policy" for e in t["events"]) for t in scored)
    return {
        "episodes": len(scored), "by_kind": dict(by_kind),
        "execution_accuracy": _rate(sum(t["correct"] for t in answerable), len(answerable)),
        "execution_accuracy_by_kind": {k: _rate(sum(t["correct"] for t in scored if t["kind"] == k), n)
                                       for k, n in by_kind.items() if k in ANSWERABLE},
        "decision_macro_f1": f1, "decision_per_class": per,
        "decision_accuracy": _rate(sum(t["decision"] == t["gold_decision"] for t in scored), len(scored)),
        "unnecessary_question_rate_clear": _rate(sum(t["questions"] > 0 for t in clear), len(clear)),
        "unnecessary_question_rate_definition": _rate(sum(t["questions"] > 0 for t in definition), len(definition)),
        "correct_abstention_rate": _rate(sum(t["correct"] for t in unans), len(unans)),
        "wrong_abstention_rate_answerable": _rate(sum(t["outcome"] == "abstained" for t in answerable), len(answerable)),
        "avg_clarification_turns": _rate(sum(t["questions"] for t in scored), len(scored)),
        "avg_clarification_turns_vague": _rate(sum(t["questions"] for t in vague), len(vague)),
        "invalid_output_rate": _rate(sum(t["invalid_outputs"] for t in scored), policy_calls),
        "limit_rate": _rate(sum(t["outcome"] == "limit" for t in scored), len(scored)),
        "invalid_episode_rate": _rate(sum(t["outcome"] == "invalid" for t in scored), len(scored)),
        "blocked_attempts": sum(t["blocked_attempts"] for t in scored),
        "reasons": dict(Counter(t["reason"] for t in scored)),
        "cost": {"model_calls": sum(t["calls"] for t in scored), "policy_calls": policy_calls,
                 "tokens_in": sum(t["tokens_in"] for t in scored), "tokens_out": sum(t["tokens_out"] for t in scored),
                 "gpu_seconds": round(sum(t["seconds"] for t in scored), 1)},
    }
