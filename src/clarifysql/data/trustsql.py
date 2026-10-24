"""TrustSQL (Spider part): questions that cannot be answered from the database.

spider_test_infeasible.json has 527 infeasible questions on 10 Spider dev databases (`query` is "null").
The `note` field says why: "column-related", "column-surface" and "column-unrelated" (100 each: the
hand-written missing-column questions), "ext-know" (needs outside knowledge) and "non-sql" (not a
database task at all, e.g. text-to-speech).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from clarifysql.config import data_dir

MISSING_COLUMN_NOTES = ("column-related", "column-surface", "column-unrelated")


def trustsql_root() -> Path:
    root = data_dir() / "trustsql" / "dataset" / "spider"
    if not root.exists():
        raise FileNotFoundError(f"TrustSQL not found at {root}; run: python -I scripts/download_data.py --only trustsql")
    return root


@lru_cache(maxsize=None)
def load_infeasible() -> tuple[dict, ...]:
    rows = json.loads((trustsql_root() / "spider_test_infeasible.json").read_text(encoding="utf-8"))
    return tuple(rows)


def missing_column_questions() -> list[dict]:
    return [r for r in load_infeasible() if r["note"] in MISSING_COLUMN_NOTES]
