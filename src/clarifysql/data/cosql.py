"""CoSQL: real clarification dialogues on the Spider databases, with labelled user intents.

We use user_intent_prediction/cosql_{train,dev}.json (one row per user turn; `intent` is a list such as
["AMBIGUOUS"] or ["CANNOT_ANSWER"]) and cosql_all_info_dialogs.json for the earlier turns of each dialogue.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from clarifysql.config import data_dir


def cosql_root() -> Path:
    root = data_dir() / "cosql" / "cosql_dataset"
    if not root.exists():
        raise FileNotFoundError(f"CoSQL not found at {root}; run: python -I scripts/download_data.py --only cosql")
    return root


@lru_cache(maxsize=None)
def load_intents(split: str) -> tuple[dict, ...]:
    """User turns of the train or dev split, each with database_id, utterance and intent (a list)."""
    rows = json.loads((cosql_root() / "user_intent_prediction" / f"cosql_{split}.json").read_text(encoding="utf-8"))
    return tuple(rows)


def intent_counts(split: str) -> Counter:
    counts: Counter = Counter()
    for row in load_intents(split):
        for label in row.get("intent") or [None]:
            counts[label] += 1
    return counts


@lru_cache(maxsize=None)
def load_dialogues() -> dict:
    return json.loads((cosql_root() / "cosql_all_info_dialogs.json").read_text(encoding="utf-8"))
