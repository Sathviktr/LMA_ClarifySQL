"""Spider 1.0: questions, gold SQL, schemas and SQLite database paths.

Layout after scripts/download_data.py:
    <data>/spider/spider_data/{train_spider,train_others,dev,test}.json, tables.json, test_tables.json,
    database/<db_id>/<db_id>.sqlite (train + dev), test_database/<db_id>/<db_id>.sqlite (all, incl. test)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from clarifysql.config import data_dir

SPLIT_FILES = {
    "train": "train_spider.json",      # 7,000 core training questions (140 databases)
    "train_others": "train_others.json",  # 1,659 legacy questions on 6 older databases (we skip these)
    "dev": "dev.json",                 # 1,034 questions on 20 databases (our main test set)
    "test": "test.json",               # 2,147 questions on 40 databases (not used for results)
}


@dataclass(frozen=True)
class Example:
    split: str
    idx: int
    db_id: str
    question: str
    query: str


def spider_root() -> Path:
    root = data_dir() / "spider" / "spider_data"
    if not root.exists():
        raise FileNotFoundError(f"Spider not found at {root}; run: python -I scripts/download_data.py --only spider")
    return root


@lru_cache(maxsize=None)
def load_split(split: str) -> tuple[Example, ...]:
    rows = json.loads((spider_root() / SPLIT_FILES[split]).read_text(encoding="utf-8"))
    return tuple(Example(split, i, r["db_id"], r["question"].strip(), r["query"].strip())
                 for i, r in enumerate(rows))


@lru_cache(maxsize=None)
def load_tables(include_test: bool = False) -> dict[str, dict]:
    """Schema entries keyed by db_id (table/column names, types, primary and foreign keys)."""
    name = "test_tables.json" if include_test else "tables.json"
    rows = json.loads((spider_root() / name).read_text(encoding="utf-8"))
    return {r["db_id"]: r for r in rows}


def db_path(db_id: str, include_test: bool = False) -> Path:
    folder = "test_database" if include_test else "database"
    path = spider_root() / folder / db_id / f"{db_id}.sqlite"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def databases(split: str) -> list[str]:
    return sorted({e.db_id for e in load_split(split)})
