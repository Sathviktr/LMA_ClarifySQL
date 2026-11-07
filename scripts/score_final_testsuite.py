"""Add official test-suite correctness to durable checkpoint-7 agent traces (CPU)."""

import argparse
import json
from pathlib import Path
from clarifysql.data import spider
from clarifysql.episodes.schema import load_episodes
from clarifysql.eval.artifacts import read_jsonl, write_json
from clarifysql.eval.testsuite import testsuite_match, suite_root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/final"))
    a = ap.parse_args()
    episodes = {e.id: e for e in load_episodes("test")}
    root = suite_root()
    for folder in sorted(a.root.glob("test_*")):
        if not (folder / "manifest.json").exists():
            continue
        rows = read_jsonl(folder / "traces.jsonl")
        for i, r in enumerate(rows):
            ep = episodes[r["episode"]]
            if "testsuite_correct" not in r:
                r["testsuite_correct"] = (bool(r.get("sql")) and r["outcome"] == "executed" and
                    testsuite_match(ep.db_id, r["sql"], ep.gold_sql, spider.db_path(ep.db_id), root)) if ep.gold_sql else None
            if (i + 1) % 25 == 0:
                print(folder.name, i + 1, "/", len(rows), flush=True)
        tmp = folder / "traces.scored.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(folder / "traces.jsonl")
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ans = [r for r in rows if r["kind"] != "unanswerable"]
        summary["testsuite_accuracy"] = sum(r["testsuite_correct"] for r in ans) / len(ans) if ans else None
        write_json(summary_path, summary)


if __name__ == "__main__":
    main()
