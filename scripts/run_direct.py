"""Run a direct SQL system (no questions) on the original Spider questions of a split.

Usage (GPU, e.g. on Kaggle):
    python scripts/run_direct.py --split validation --system oracle
    python scripts/run_direct.py --split test --system oracle --max-repairs 2     # oracle with repair
Laptop check of the whole pipeline without a model (the "model" just answers with the gold SQL):
    python -I scripts/run_direct.py --split validation --backend mock-gold --limit 20

On the original questions B1 and the oracle receive the same input, so this measures the SQL writer itself.
B1 on *vague* requests needs the episodes of Phase 3.
Writes results/direct/<split>_<system>.jsonl (one line per question) and ..._summary.json.
"""

from __future__ import annotations

import argparse
import json
import time

from clarifysql.config import results_dir
from clarifysql.data import spider
from clarifysql.data.splits import load_splits
from clarifysql.llm.backend import MockBackend
from clarifysql.systems.direct import Item, run_direct, summarize


def original_questions(split: str) -> list[Item]:
    source = "dev" if split == "test" else "train"
    dbs = set(load_splits()[split])
    return [Item(f"{source}-{e.idx}", e.db_id, e.question, e.query)
            for e in spider.load_split(source) if e.db_id in dbs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--system", choices=["oracle"], default="oracle")
    ap.add_argument("--backend", choices=["hf", "mock-gold"], default="hf")
    ap.add_argument("--max-repairs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = original_questions(args.split)[: args.limit]
    if args.backend == "hf":
        from clarifysql.llm.backend import HFBackend

        backend = HFBackend()
    else:
        gold = {it.request: it.gold_sql for it in items}
        backend = MockBackend(lambda chat, adapter: f"```sql\n{gold[chat[-1]['content'].split('Request: ')[-1]]}\n```")
    t0 = time.perf_counter()
    records = run_direct(items, backend, spider.db_path, max_repairs=args.max_repairs, batch_size=args.batch_size)
    wall = time.perf_counter() - t0
    name = f"{args.split}_{args.system}" + (f"_repair{args.max_repairs}" if args.max_repairs else "")
    if args.backend != "hf":
        name += f"_{args.backend}"
    out_dir = results_dir() / "direct"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"split": args.split, "system": args.system, "backend": args.backend,
               "max_repairs": args.max_repairs, "batch_size": args.batch_size, "wall_seconds": round(wall, 1),
               **summarize(records)}
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
