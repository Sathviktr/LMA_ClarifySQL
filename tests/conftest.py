import json
import sqlite3
from pathlib import Path

import pytest

SAFETY_SET = [json.loads(line) for line in
              (Path(__file__).parent / "safety_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def tiny_db(tmp_path) -> Path:
    """A small concert database (made-up rows) in its own temporary folder."""
    path = tmp_path / "concert.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER);
        CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, concert_name TEXT, year INTEGER);
        CREATE TABLE singer_in_concert (concert_id INTEGER REFERENCES concert(concert_id),
                                        singer_id INTEGER REFERENCES singer(singer_id));
        INSERT INTO singer VALUES (1, 'Asha', 'India', 29), (2, 'Ben', 'France', 43),
                                  (3, 'Chen', 'China', 35), (4, 'Dana', 'France', 52);
        INSERT INTO concert VALUES (1, 'Spring Gala', 2014), (2, 'Summer Nights', 2015);
        INSERT INTO singer_in_concert VALUES (1, 1), (1, 2), (2, 2), (2, 3);
    """)
    con.commit()
    con.close()
    return path
