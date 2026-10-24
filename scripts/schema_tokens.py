"""Measure schema sizes in Qwen3 tokens: full CREATE TABLE text vs. the compact catalog of names.

Usage:  python -I scripts/schema_tokens.py
Writes results/data/schema_tokens.json. Only the tokenizer is downloaded (a few MB), not the model.
"""

from __future__ import annotations

import json
import statistics

from transformers import AutoTokenizer

from clarifysql.config import MODEL_ID, results_dir
from clarifysql.data import spider
from clarifysql.data.splits import load_splits
from clarifysql.db.catalog import compact_catalog, create_table_text


def summary(values: dict[str, int]) -> dict:
    v = sorted(values.values())
    return {"databases": len(v), "median": statistics.median(v), "p90": v[int(0.9 * (len(v) - 1))],
            "max": v[-1], "largest": max(values, key=values.get)}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    splits = load_splits()
    out: dict = {"tokenizer": MODEL_ID, "per_database": {}}
    for name in ("train", "validation", "test"):
        full, cat = {}, {}
        for db in splits[name]:
            path = spider.db_path(db)
            full[db] = count(create_table_text(path))
            cat[db] = count(compact_catalog(path))
            out["per_database"][db] = {"split": name, "create_table_tokens": full[db], "catalog_tokens": cat[db]}
        out[name] = {"create_table": summary(full), "catalog": summary(cat)}
        print(f"{name:<10} CREATE TABLE: {out[name]['create_table']}")
        print(f"{'':<10} catalog:      {out[name]['catalog']}")
    core = {db: v for db, v in out["per_database"].items() if v["split"] in ("train", "validation")}
    out["core_train"] = {
        "create_table": summary({db: v["create_table_tokens"] for db, v in core.items()}),
        "catalog": summary({db: v["catalog_tokens"] for db, v in core.items()}),
    }
    print(f"core train (140): {out['core_train']}")
    path = results_dir() / "data" / "schema_tokens.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
