import json

import pytest

from clarifysql.config import data_dir
from clarifysql.episodes.ambiguate import Slot, candidate_measures, ranking_draft, threshold_draft
from clarifysql.episodes.build import describe_slot, plural, singular
from clarifysql.episodes.schema import EPISODE_DIR, Episode


@pytest.mark.parametrize("question,sql,expected,value", [
    ("How many heads of the departments are older than 56 ?", "SELECT count(*) FROM head WHERE age > 56",
     "How many heads of the departments are older?", "> 56"),
    ("How many cities are there that have more than 3 airports?",
     "SELECT city FROM airport GROUP BY city HAVING count(*) > 3",
     "How many cities are there that have many airports?", "> 3"),
    ("Find the official names of cities with population bigger than 1500.",
     "SELECT Official_Name FROM city WHERE Population > 1500",
     "Find the official names of cities with big population.", "> 1500"),
    ("Which campus was opened between 1935 and 1939?", "SELECT campus FROM campuses WHERE YEAR BETWEEN 1935 AND 1939",
     "Which campus was opened within a certain range?", "between 1935 and 1939"),
    ("What are names of the movies made after 2000?", "SELECT title FROM movie WHERE year > 2000",
     "What are names of the movies made recently?", "> 2000"),
])
def test_threshold_drafts(question, sql, expected, value):
    vague, slot = threshold_draft(question, sql)
    assert vague == expected and slot.value == value and slot.kind == "threshold"
    assert slot.answer.startswith("I mean ") and slot.hidden in question


def test_threshold_draft_skips_equality_and_unknown_phrasing():
    assert threshold_draft("the student whose id is 171", "SELECT * FROM s WHERE id = 171") is None


def test_ranking_draft_hides_the_measure(tiny_db):
    from clarifysql.db.catalog import schema_info

    schema = schema_info(str(tiny_db))
    sql = "SELECT name FROM singer ORDER BY age DESC LIMIT 1"
    draft, slot = ranking_draft("What is the name of the singer with the highest age?", sql,
                                candidate_measures(sql, schema))
    assert draft == "What is the name of the singer that ranks highest?"
    assert slot.value == "age" and slot.answer == "Rank them by age."
    assert ranking_draft("What is the carrier of the most expensive phone?",
                         "SELECT carrier FROM phone ORDER BY price DESC LIMIT 1", ["a", "b"])[0] is None


def test_describe_slot_makes_readable_terms_and_questions():
    t = Slot("threshold", "count(*)", "more than 4", "> 4", "many", ["count(*) > some value (not given)"])
    d = describe_slot(t, "Find the policy type used by many customers.", "policy",
                      "Find the policy type used by more than 4 customers.")
    assert d == {"term": "many customers", "definition": "more than 4 customers", "aliases": [],
                 "question": 'How many customers count as "many"?',
                 "meanings": ["number of customers above some value (which?)"]}
    r = Slot("ranking", "Population", "with the largest population", "population", "ranks highest",
             ["city.Area_km_2", "city.Population"])
    d = describe_slot(r, "List the name of the city that ranks highest.", "record")
    assert d["term"] == "city ranks highest" and d["question"] == "Ranked by what: area km 2 or population?"


def test_singular_and_plural():
    assert [singular(w) for w in ("cities", "Students", "undergoes", "class", "Ref_Document_Types")] == \
        ["city", "student", "undergo", "class", "ref document type"]
    assert plural("city") == "cities" and plural("singer") == "singers"


def test_episode_roundtrip_keeps_first_decision():
    ep = Episode("x", "train", "db", "ambiguous", "Show older singers.")
    d = ep.to_dict()
    assert d["first_decision"] == "clarify" and Episode.from_dict(d) == ep


built = pytest.mark.skipif(not (EPISODE_DIR / "train.jsonl").exists(), reason="episodes not built")


@built
def test_built_splits_have_the_planned_sizes():
    from clarifysql.episodes.review import dropped

    sizes = {"train": 340, "validation": 60, "test": 200}
    for split, n in sizes.items():
        n -= len(dropped(split))  # episodes the team's review removes (scripts/apply_reviews.py)
        rows = [json.loads(line) for line in (EPISODE_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == n
        assert len({r["id"] for r in rows}) == n


@built
def test_episodes_never_mix_splits():
    from clarifysql.data.splits import load_splits

    s = load_splits()
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (EPISODE_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {r["db_id"] for r in rows} <= set(s[split])


@built
@pytest.mark.skipif(not (data_dir() / "spider").exists(), reason="benchmark data not downloaded")
def test_every_training_episode_passes_the_checks():
    from clarifysql.episodes.checks import check_episodes
    from clarifysql.episodes.schema import load_episodes

    report = check_episodes(load_episodes("train"))
    assert report["summary"]["passed"] == report["summary"]["episodes"], report["summary"]["problems"]
