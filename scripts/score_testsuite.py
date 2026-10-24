"""Add test-suite correctness to a predictions file of the test split (Spider dev databases).

Usage:  python -I scripts/score_testsuite.py results/direct/test_oracle_repair2.jsonl
Needs the evaluator (cloned automatically) and the database variants:
    python -I scripts/download_data.py --only testsuite
Adds `testsuite_correct` (final answer) and `first_testsuite_correct` (before repair) to every line and
writes the accuracies into the matching *_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clarifysql.data import spider
from clarifysql.eval.testsuite import suite_root, testsuite_match


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    args = ap.parse_args()
    root = suite_root()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    for i, r in enumerate(rows, 1):
        db = spider.db_path(r["db_id"])
        r["testsuite_correct"] = testsuite_match(r["db_id"], r["pred_sql"], r["gold_sql"], db, root)
        first = r.get("first_pred_sql", r["pred_sql"])
        r["first_testsuite_correct"] = (r["testsuite_correct"] if first == r["pred_sql"]
                                        else testsuite_match(r["db_id"], first, r["gold_sql"], db, root))
        if i % 100 == 0:
            print(f"{i}/{len(rows)}")
    args.predictions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n = len(rows)
    acc = {"testsuite_accuracy": round(sum(r["testsuite_correct"] for r in rows) / n, 4),
           "testsuite_accuracy_before_repair": round(sum(r["first_testsuite_correct"] for r in rows) / n, 4)}
    summary_path = args.predictions.with_name(args.predictions.stem + "_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(acc)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acc, indent=2))


if __name__ == "__main__":
    main()
