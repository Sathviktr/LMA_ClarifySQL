"""The official test-suite evaluator, wrapped; skipped when the evaluator has not been cloned."""

import hashlib
import shutil
import sqlite3

import pytest

from clarifysql.eval.testsuite import evaluator_dir

pytestmark = pytest.mark.skipif(not (evaluator_dir() / "exec_eval.py").exists(),
                                reason="test-suite evaluator not cloned (see eval/testsuite.py)")

GOLD = "SELECT country FROM singer GROUP BY country HAVING avg(age) > 40"
WRONG = "SELECT country FROM singer WHERE age > 40 GROUP BY country"


@pytest.fixture
def suite(tiny_db, tmp_path):
    """A two-variant test suite for a database called 'concert'."""
    root = tmp_path / "suite"
    (root / "concert").mkdir(parents=True)
    shutil.copy(tiny_db, root / "concert" / "concert.sqlite")
    variant = root / "concert" / "concert_v2.sqlite"
    shutil.copy(tiny_db, variant)
    con = sqlite3.connect(variant)
    con.execute("INSERT INTO singer VALUES (5, 'Eve', 'China', 45)")
    con.commit()
    con.close()
    return root


def test_suite_catches_what_one_database_misses(suite, tiny_db):
    from clarifysql.eval.exec_match import exec_match
    from clarifysql.eval.testsuite import testsuite_match

    assert exec_match(tiny_db, WRONG, GOLD).correct                  # one database: fooled
    assert not testsuite_match("concert", WRONG, GOLD, tiny_db, root=suite)  # two variants: caught
    assert testsuite_match("concert", GOLD, GOLD, tiny_db, root=suite)


def test_unsafe_predictions_never_reach_the_official_code(suite, tiny_db):
    from clarifysql.eval.testsuite import testsuite_match

    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (suite / "concert").iterdir()}
    assert not testsuite_match("concert", "DELETE FROM singer", GOLD, tiny_db, root=suite)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (suite / "concert").iterdir()}
    assert before == after
