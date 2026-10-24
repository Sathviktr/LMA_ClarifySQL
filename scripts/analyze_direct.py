"""Error analysis of a direct-SQL run (checkpoint 2): error types, judge agreement, per-database accuracy.

Usage:  python -I scripts/analyze_direct.py results/direct/validation_oracle_repair2.jsonl
                                            results/direct/test_oracle_repair2.jsonl
Writes results/direct/analysis.json.

Error types (for wrong predictions that ran): extra / missing columns, wrong row order, duplicates differ
(same distinct rows, different copies), different number of rows, or same shape with different values.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from clarifysql.config import results_dir
from clarifysql.data import spider
from clarifysql.db.safe_exec import execute
from clarifysql.eval.exec_match import _rows, order_matters, results_match


def error_type(r: dict) -> str:
    if r["status"] != "ok":
        return "did not run"
    db = spider.db_path(r["db_id"])
    gold, pred = execute(db, r["gold_sql"]), execute(db, r["pred_sql"])
    if len(pred.columns) != len(gold.columns):
        return "extra columns" if len(pred.columns) > len(gold.columns) else "missing columns"
    if order_matters(r["gold_sql"]) and results_match(pred.rows, gold.rows, ordered=False):
        return "wrong row order"
    if set(_rows(pred.rows)) == set(_rows(gold.rows)):
        return "duplicates differ"
    if len(pred.rows) != len(gold.rows):
        return "different number of rows"
    return "same shape, different values"


def analyze(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    wrong = [r for r in rows if not r["correct"]]
    by_db = defaultdict(list)
    for r in rows:
        by_db[r["db_id"]].append(r["correct"])
    seconds = sorted(r["seconds"] for r in rows)
    out = {
        "items": len(rows),
        "accuracy_before_repair": round(sum(r["first_correct"] for r in rows) / len(rows), 4),
        "accuracy_after_repair": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "first_pass_crashes": sum(r["first_status"] == "sql_error" for r in rows),
        "fixed_by_repair": sum(r["correct"] and not r["first_correct"] for r in rows),
        "error_types": dict(Counter(error_type(r) for r in wrong).most_common()),
        "per_database_accuracy": {db: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)}
                                  for db, v in sorted(by_db.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))},
        "seconds_per_item": {"median": round(seconds[len(seconds) // 2], 3),
                             "p90": round(seconds[int(0.9 * (len(seconds) - 1))], 3), "max": round(seconds[-1], 3)},
    }
    if "testsuite_correct" in rows[0]:
        agree = Counter((r["correct"], r["testsuite_correct"]) for r in rows)
        out["testsuite_accuracy"] = round(sum(r["testsuite_correct"] for r in rows) / len(rows), 4)
        out["judge_agreement"] = {"both_correct": agree[(True, True)], "both_wrong": agree[(False, False)],
                                  "exec_only_correct": agree[(True, False)], "suite_only_correct": agree[(False, True)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    result = {f.stem: analyze(f) for f in args.files}
    path = results_dir() / "direct" / "analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, a in result.items():
        print(name, {k: v for k, v in a.items() if k != "per_database_accuracy"})
    print(f"written to {path}")


if __name__ == "__main__":
    main()
