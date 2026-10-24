"""The official test-suite evaluator (Zhong, Yu and Klein, EMNLP 2020), one prediction at a time.

A single database can make a wrong query look right by coincidence. The test suite runs the gold and the
predicted query on *many variants* of each Spider dev database (same schema, different rows) and counts
the prediction as correct only if the results agree on every variant. We call its `eval_exec_match`
without `plug_value`, because predicting the clarified values is part of our task (proposal, Section 10).

Safety: the official code runs queries on ordinary read-write connections. So a prediction is only passed
to it after it has run successfully under all our safety layers on the original database.

Setup (done once, outside the repository):
    git clone https://github.com/taoyds/test-suite-sql-eval  <data>/third_party/test-suite-sql-eval
    python -I scripts/download_data.py --only testsuite       # the database variants
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from clarifysql.config import data_dir
from clarifysql.db.safe_exec import execute

REPO_URL = "https://github.com/taoyds/test-suite-sql-eval.git"
PINNED_COMMIT = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"


def evaluator_dir() -> Path:
    return data_dir() / "third_party" / "test-suite-sql-eval"


def ensure_evaluator() -> Path:
    path = evaluator_dir()
    if not (path / "exec_eval.py").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO_URL, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", PINNED_COMMIT], check=True)
    return path


@lru_cache(maxsize=1)
def _exec_eval():
    path = str(ensure_evaluator())
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("exec_eval")


def suite_root() -> Path:
    """Folder that contains one sub-folder per database id with all its .sqlite variants."""
    base = data_dir() / "testsuite"
    for candidate in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if (candidate / "concert_singer" / "concert_singer.sqlite").exists():
            return candidate
    raise FileNotFoundError(f"test-suite databases not found under {base}; "
                            "run: python -I scripts/download_data.py --only testsuite")


def variants(db_id: str, root: Path | None = None) -> list[Path]:
    folder = (root or suite_root()) / db_id
    return sorted(folder.glob("*.sqlite"))


def testsuite_match(db_id: str, pred_sql: str, gold_sql: str, original_db: str | Path,
                    root: Path | None = None) -> bool:
    """True if the prediction agrees with the gold query on every database variant of `db_id`."""
    if not execute(original_db, pred_sql).ok:  # safety layers first; failures are simply wrong
        return False
    folder = (root or suite_root()) / db_id
    db_file = folder / f"{db_id}.sqlite"
    return bool(_exec_eval().eval_exec_match(str(db_file), pred_sql, gold_sql, plug_value=False,
                                             keep_distinct=False, progress_bar_for_each_datapoint=False))
