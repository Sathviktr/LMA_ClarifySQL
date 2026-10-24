import pytest

from clarifysql.eval.exec_match import exec_match, order_matters, results_match


@pytest.mark.parametrize("sql,expected", [
    ("SELECT name FROM singer ORDER BY age", True),
    ("SELECT name FROM singer", False),
    ("SELECT name FROM singer WHERE age = (SELECT max(age) FROM singer)", False),
    ("SELECT name FROM (SELECT name, age FROM singer ORDER BY age LIMIT 2)", False),  # sorted only inside
    ("SELECT name FROM singer WHERE name = 'order by'", False),                      # inside a string
])
def test_order_matters_only_for_outer_order_by(sql, expected):
    assert order_matters(sql) is expected


def test_multiset_comparison_ignores_order_but_counts_duplicates():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert not results_match([(1,), (1,), (2,)], [(1,), (2,), (2,)], ordered=False)


def test_columns_may_be_reordered():
    assert results_match([("a", 1), ("b", 2)], [(1, "a"), (2, "b")], ordered=False)


def test_numbers_nulls_and_text():
    assert results_match([(5.0, None)], [(5, None)], ordered=False)
    assert results_match([(0.333333,)], [(1 / 3,)], ordered=False)
    assert not results_match([("france",)], [("France",)], ordered=False)


def test_empty_results_match_each_other():
    assert results_match([], [], ordered=False)


def test_exec_match_on_a_database(tiny_db):
    gold = "SELECT country, count(*) FROM singer GROUP BY country"
    assert exec_match(tiny_db, "SELECT count(*), country FROM singer GROUP BY country", gold).correct
    r = exec_match(tiny_db, "SELECT country FROM singer", gold)
    assert not r.correct and r.reason == "different_result"
    r = exec_match(tiny_db, "SELECT contry FROM singer", gold)
    assert r.reason == "pred_sql_error"
    r = exec_match(tiny_db, "DELETE FROM singer", gold)
    assert r.reason == "pred_blocked"


def test_one_database_can_be_fooled_but_a_second_variant_is_not(tiny_db, tmp_path):
    """Why the test-suite evaluator exists: a wrong query can return the right rows by coincidence."""
    import shutil
    import sqlite3

    gold = "SELECT country FROM singer GROUP BY country HAVING avg(age) > 40"
    wrong = "SELECT country FROM singer WHERE age > 40 GROUP BY country"
    assert exec_match(tiny_db, wrong, gold).correct  # false positive: both return only France here

    variant = tmp_path / "variant" / "concert.sqlite"
    variant.parent.mkdir()
    shutil.copy(tiny_db, variant)
    con = sqlite3.connect(variant)
    con.execute("INSERT INTO singer VALUES (5, 'Eve', 'China', 45)")  # China: average 40, one singer over 40
    con.commit()
    con.close()
    assert not exec_match(variant, wrong, gold).correct
