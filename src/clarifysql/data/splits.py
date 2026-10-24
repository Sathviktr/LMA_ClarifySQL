"""Database-level splits (proposal, Section 8.3).

* train:      the 140 core Spider training databases minus 10 held out   -> fine-tuning data
* validation: 10 held-out training databases                              -> choosing prompts/checkpoints
* test:       the 20 Spider dev databases                                  -> the final comparison
A database belongs to exactly one split, so nothing about a test database is seen before testing.

The validation databases are drawn at random (fixed seed) from training databases that offer enough
material: at least MIN_QUESTIONS questions and MIN_CANDIDATES ambiguity candidates (threshold or ranking),
so that 24 ambiguous validation episodes can be built from them.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

from clarifysql.config import REPO_ROOT
from clarifysql.data import cosql, spider, trustsql
from clarifysql.data.normalize import normalize_sql
from clarifysql.episodes.mining import candidates

SEED = 2026
N_VALIDATION = 10
MIN_QUESTIONS = 20
MIN_CANDIDATES = 3
SPLITS_PATH = REPO_ROOT / "data" / "splits" / "splits.json"


def _split_stats(examples, dbs: list[str]) -> dict:
    exs = [e for e in examples if e.db_id in set(dbs)]
    cands = candidates(exs)
    return {
        "databases": len(dbs),
        "questions": len(exs),
        "distinct_queries": len({(e.db_id, normalize_sql(e.query)) for e in exs}),
        "threshold_candidates": len(cands["threshold"]),
        "ranking_candidates": len(cands["ranking"]),
    }


def make_splits(seed: int = SEED) -> dict:
    train_ex = spider.load_split("train")
    dev_ex = spider.load_split("dev")
    questions = Counter(e.db_id for e in train_ex)
    cands = candidates(train_ex)
    per_db = Counter(e.db_id for kind in cands.values() for e in kind)
    core = spider.databases("train")
    eligible = [db for db in core if questions[db] >= MIN_QUESTIONS and per_db[db] >= MIN_CANDIDATES]
    validation = sorted(random.Random(seed).sample(eligible, N_VALIDATION))
    train = sorted(set(core) - set(validation))
    test = spider.databases("dev")
    trust_dbs = sorted({r["db_id"] for r in trustsql.load_infeasible()})
    cosql_dev_dbs = sorted({r["database_id"] for r in cosql.load_intents("dev")})
    return {
        "seed": seed,
        "rules": {
            "validation_pool": f"core train databases with >= {MIN_QUESTIONS} questions and "
                               f">= {MIN_CANDIDATES} threshold/ranking candidates",
            "eligible_count": len(eligible),
            "n_validation": N_VALIDATION,
            "not_used": "Spider train_others (6 legacy databases) and the official Spider test set",
        },
        "train": train,
        "validation": validation,
        "test": test,
        "stats": {
            "train": _split_stats(train_ex, train),
            "validation": _split_stats(train_ex, validation),
            "test": _split_stats(dev_ex, test),
        },
        "test_extras": {
            "trustsql_infeasible_databases": trust_dbs,
            "cosql_dev_databases_all_in_test": set(cosql_dev_dbs) <= set(test),
        },
    }


def write_splits(splits: dict, path: Path = SPLITS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return path


@lru_cache(maxsize=None)
def load_splits(path: Path = SPLITS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_of(db_id: str) -> str | None:
    s = load_splits()
    for name in ("train", "validation", "test"):
        if db_id in s[name]:
            return name
    return None
