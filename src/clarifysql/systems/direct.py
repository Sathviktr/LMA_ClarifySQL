"""Direct SQL systems: the request goes straight to the frozen SQL writer, without any questions.

* B1      the request + the schema -> one SQL query; no repair        (proposal, Section 11)
* B2      B1 plus up to 2 repairs when SQLite returns an error
* oracle  the original, fully specified question -> the same SQL writer: an upper bound, not a real system
All of them use the same prompt (llm/prompts.py) and greedy decoding, and run SQL only through the safe executor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from clarifysql.db.catalog import create_table_text
from clarifysql.db.safe_exec import execute
from clarifysql.eval.exec_match import exec_match
from clarifysql.llm.prompts import extract_sql, repair_messages, sql_writer_messages


@dataclass
class Item:
    id: str
    db_id: str
    request: str
    gold_sql: str | None = None


@lru_cache(maxsize=None)
def _schema(db_path: str) -> str:
    return create_table_text(db_path)


def run_direct(items: list[Item], backend, db_path_of: Callable[[str], Path], max_repairs: int = 0,
               batch_size: int = 8, max_new_tokens: int = 256) -> list[dict]:
    paths = {it.db_id: str(db_path_of(it.db_id)) for it in items}
    chats = [sql_writer_messages(_schema(paths[it.db_id]), it.request) for it in items]
    gens = backend.generate(chats, max_new_tokens=max_new_tokens, adapter=False, batch_size=batch_size)
    records = []
    for it, g in zip(items, gens):
        records.append({**asdict(it), "raw": g.text, "pred_sql": extract_sql(g.text), "repairs": 0,
                        "prompt_tokens": g.prompt_tokens, "new_tokens": g.new_tokens, "seconds": g.seconds,
                        "model_calls": 1})
    _score(records, paths)
    for rec in records:  # keep the first answer, so one run gives both "no repair" (B1) and "repair" (B2)
        rec.update(first_pred_sql=rec["pred_sql"], first_status=rec["status"], first_correct=rec["correct"])
    for _ in range(max_repairs):
        todo = []
        for rec in records:
            if rec["status"] == "sql_error":
                todo.append((rec, rec["error"]))
        if not todo:
            break
        fix = [repair_messages(_schema(paths[r["db_id"]]), r["request"], r["pred_sql"], err) for r, err in todo]
        for (rec, _), g in zip(todo, backend.generate(fix, max_new_tokens=max_new_tokens, adapter=False,
                                                       batch_size=batch_size)):
            rec.update(raw=g.text, pred_sql=extract_sql(g.text), repairs=rec["repairs"] + 1,
                       prompt_tokens=rec["prompt_tokens"] + g.prompt_tokens,
                       new_tokens=rec["new_tokens"] + g.new_tokens, seconds=rec["seconds"] + g.seconds,
                       model_calls=rec["model_calls"] + 1)
        _score([r for r, _ in todo], paths)
    return records


def _score(records: list[dict], paths: dict[str, str]) -> None:
    for rec in records:
        path = paths[rec["db_id"]]
        if rec["gold_sql"]:
            m = exec_match(path, rec["pred_sql"], rec["gold_sql"])
            rec.update(status=m.pred.status, correct=m.correct, reason=m.reason, error=m.pred.error)
        else:
            res = execute(path, rec["pred_sql"])
            rec.update(status=res.status, correct=None, reason=None, error=res.error)


def summarize(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("correct") is not None]
    n = len(records)
    return {
        "items": n,
        "execution_accuracy": round(sum(r["correct"] for r in scored) / len(scored), 4) if scored else None,
        "execution_accuracy_before_repair":
            round(sum(r["first_correct"] for r in scored) / len(scored), 4) if scored else None,
        "status_counts": dict(Counter(r["status"] for r in records)),
        "reason_counts": dict(Counter(r["reason"] for r in scored)),
        "repaired_items": sum(r["repairs"] > 0 for r in records),
        "mean_seconds_per_item": round(sum(r["seconds"] for r in records) / n, 3) if n else None,
        "mean_new_tokens": round(sum(r["new_tokens"] for r in records) / n, 1) if n else None,
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in records) / n, 1) if n else None,
    }
