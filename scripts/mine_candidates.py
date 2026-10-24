"""Count ambiguity candidates per split (proposal, Section 8.2) and write results/episodes/candidates.json.

Usage:  python -I scripts/mine_candidates.py
threshold and ranking candidates are the main source of ambiguous episodes; date and entity candidates
are counted as a reserve (the proposal's fallback if there are too few good ambiguous examples).
"""

from __future__ import annotations

import json

from clarifysql.config import results_dir
from clarifysql.data import spider
from clarifysql.data.splits import load_splits
from clarifysql.db.catalog import schema_info
from clarifysql.db.safe_exec import open_readonly
from clarifysql.episodes.mining import dedupe, entity_columns, is_date_ambiguous, is_ranking, is_threshold


def main() -> None:
    splits = load_splits()
    out = {}
    for split, source in (("train", "train"), ("validation", "train"), ("test", "dev")):
        dbs = set(splits[split])
        unique = dedupe(e for e in spider.load_split(source) if e.db_id in dbs)
        counts = {"unique_questions": len(unique), "threshold": 0, "ranking": 0, "both": 0, "date": 0, "entity": 0}
        cons = {}
        for e in unique:
            schema = schema_info(str(spider.db_path(e.db_id)))
            t, r = is_threshold(e), is_ranking(e)
            counts["threshold"] += t
            counts["ranking"] += r
            counts["both"] += t and r
            counts["date"] += is_date_ambiguous(e, schema)
            con = cons.setdefault(e.db_id, open_readonly(spider.db_path(e.db_id), guarded=False))
            counts["entity"] += any(len(v) >= 2 for v in entity_columns(e, schema, con).values())
        for con in cons.values():
            con.close()
        out[split] = counts
        print(split, counts)
    path = results_dir() / "episodes" / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
