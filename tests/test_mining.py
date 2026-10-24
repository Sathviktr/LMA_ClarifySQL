import sqlite3

from clarifysql.data.spider import Example
from clarifysql.db.catalog import schema_info
from clarifysql.db.safe_exec import open_readonly
from clarifysql.episodes.mining import (
    candidates, cutoff_values, dedupe, entity_columns, is_date_ambiguous, is_ranking, is_threshold,
    threshold_values,
)


def ex(question, query, db="concert", idx=0):
    return Example("train", idx, db, question, query)


def test_threshold_needs_a_cutoff_comparison():
    assert threshold_values("singers older than 30", "SELECT name FROM singer WHERE age > 30") == {"30"}
    assert threshold_values("ages between 20 and 40", "SELECT * FROM singer WHERE age BETWEEN 20 AND 40") == {"20", "40"}
    assert not is_threshold(ex("the student whose id is 171", "SELECT * FROM s WHERE student_id = 171"))
    assert not is_threshold(ex("the top 3 singers", "SELECT name FROM singer ORDER BY age DESC LIMIT 3"))
    assert cutoff_values("SELECT * FROM t WHERE x >= 2.50") == {"2.5"}


def test_ranking_needs_order_by_and_limit():
    assert is_ranking(ex("the oldest singer", "SELECT name FROM singer ORDER BY age DESC LIMIT 1"))
    assert not is_ranking(ex("singers by age", "SELECT name FROM singer ORDER BY age"))


def test_dedupe_keeps_one_question_per_query():
    a = ex("How many singers?", "SELECT count(*) FROM singer", idx=1)
    b = ex("Count the singers.", "select count(*) from singer;", idx=2)
    assert dedupe([a, b]) == [a]
    assert candidates([a, b]) == {"threshold": [], "ranking": []}


def test_date_and_entity_candidates(tiny_db):
    con = sqlite3.connect(tiny_db)
    con.executescript("""
        CREATE TABLE booking (id INTEGER, booked_date TEXT, event_date TEXT, guest TEXT);
        INSERT INTO booking VALUES (1, '2015-01-02', '2015-03-04', 'Asha');
    """)
    con.commit()
    con.close()
    schema = schema_info(str(tiny_db))
    date_ex = ex("bookings in 2015", "SELECT id FROM booking WHERE booked_date LIKE '2015%'")
    assert is_date_ambiguous(date_ex, schema)
    assert not is_date_ambiguous(ex("all bookings", "SELECT id FROM booking"), schema)
    ro = open_readonly(tiny_db, guarded=False)
    hits = entity_columns(ex("records of Asha", "SELECT id FROM booking WHERE guest = 'Asha'"), schema, ro)
    assert sorted(hits["Asha"]) == ["booking.guest", "singer.name"]  # the value lives in two columns
