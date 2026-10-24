import json

import pytest

from clarifysql.config import data_dir
from clarifysql.data.splits import SPLITS_PATH

SPLITS = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))


def test_no_database_in_two_splits():
    train, val, test = (set(SPLITS[k]) for k in ("train", "validation", "test"))
    assert not (train & val) and not (train & test) and not (val & test)


def test_split_sizes_match_proposal():
    assert len(SPLITS["validation"]) == 10
    assert len(SPLITS["train"]) == 130
    assert len(SPLITS["test"]) == 20


def test_validation_has_material_for_24_ambiguous_episodes():
    st = SPLITS["stats"]["validation"]
    assert st["threshold_candidates"] + st["ranking_candidates"] >= 24


@pytest.mark.skipif(not (data_dir() / "spider").exists(), reason="benchmark data not downloaded")
def test_splits_are_reproducible_from_the_seed():
    from clarifysql.data.splits import make_splits

    assert make_splits(SPLITS["seed"]) == SPLITS
