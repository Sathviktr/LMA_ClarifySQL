"""Checks on the real downloaded data; skipped when the data folder has not been downloaded."""

import pytest

from clarifysql.config import data_dir

pytestmark = pytest.mark.skipif(not (data_dir() / "spider").exists(),
                                reason="benchmark data not downloaded (scripts/download_data.py)")


def test_spider_split_sizes():
    from clarifysql.data import spider

    assert len(spider.load_split("train")) == 7000
    assert len(spider.load_split("dev")) == 1034
    assert len(spider.databases("train")) == 140
    assert len(spider.databases("dev")) == 20


def test_every_spider_database_file_exists():
    from clarifysql.data import spider

    for split in ("train", "dev"):
        for db in spider.databases(split):
            assert spider.db_path(db).is_file()


def test_cosql_and_trustsql_counts():
    from clarifysql.data import cosql, trustsql

    assert cosql.intent_counts("dev")["AMBIGUOUS"] == 107
    assert len(trustsql.missing_column_questions()) == 300
