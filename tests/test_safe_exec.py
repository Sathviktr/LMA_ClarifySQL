import hashlib

import pytest
from conftest import SAFETY_SET

from clarifysql.db.safe_exec import check_statement, execute

SQL_CASES = [c for c in SAFETY_SET if c["target"] == "sql"]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_safety_set_has_20_cases():
    assert len(SAFETY_SET) == 20
    assert len({c["id"] for c in SAFETY_SET}) == 20


@pytest.mark.parametrize("case", SQL_CASES, ids=lambda c: f"{c['id']}-{c['category']}")
def test_safety_case(case, tiny_db):
    before = fingerprint(tiny_db)
    files_before = sorted(p.name for p in tiny_db.parent.iterdir())
    result = execute(tiny_db, case["sql"], time_limit_s=1.0)
    assert result.status == case["expect"], (case["why"], result.error)
    assert fingerprint(tiny_db) == before, "the database file changed"
    assert sorted(p.name for p in tiny_db.parent.iterdir()) == files_before, "a new file was created"


@pytest.mark.parametrize("sql", [
    "SELECT name FROM singer WHERE age > 30 ORDER BY age DESC",
    "SELECT country, count(*) FROM singer GROUP BY country HAVING count(*) > 1",
    "SELECT T1.name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id",
    "SELECT name FROM singer WHERE singer_id IN (SELECT singer_id FROM singer_in_concert)",
    "SELECT name FROM singer UNION SELECT concert_name FROM concert",
    "SELECT country FROM singer INTERSECT SELECT country FROM singer WHERE age > 40",
    "SELECT country FROM singer EXCEPT SELECT country FROM singer WHERE age > 40",
    "WITH young AS (SELECT * FROM singer WHERE age < 40) SELECT name FROM young",
    "SELECT name FROM singer WHERE country = 'France';",
    'SELECT name FROM singer WHERE country = "France"',
    "/* note */ SELECT 'a;b' AS text_with_semicolon",
])
def test_ordinary_read_queries_are_allowed(sql, tiny_db):
    result = execute(tiny_db, sql)
    assert result.ok, result.error


def test_results_and_columns(tiny_db):
    result = execute(tiny_db, "SELECT country, count(*) AS n FROM singer GROUP BY country HAVING n > 1")
    assert result.columns == ["country", "n"]
    assert result.rows == [("France", 2)]


def test_row_cap_marks_truncation(tiny_db):
    sql = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n LIMIT 50) SELECT x FROM n"
    result = execute(tiny_db, sql, max_rows=10)
    assert result.ok and len(result.rows) == 10 and result.truncated


def test_syntax_errors_are_reported_not_raised(tiny_db):
    result = execute(tiny_db, "SELECT nme FROM singer")
    assert result.status == "sql_error" and "no such column" in result.error


@pytest.mark.parametrize("sql,reason", [
    ("", "empty"),
    ("   -- only a comment", "empty"),
    ("SELECT 1; SELECT 2", "more than one statement"),
    ("EXPLAIN SELECT 1", "only SELECT or WITH"),
])
def test_check_statement_reasons(sql, reason):
    assert reason in check_statement(sql)


def test_semicolon_inside_string_is_not_a_second_statement():
    assert check_statement("SELECT 'x;y' FROM singer;") is None
