import hashlib

import pytest
from conftest import SAFETY_SET

from clarifysql.db.catalog import compact_catalog, create_table_text, schema_info
from clarifysql.tools import ToolContext, run_tool
from clarifysql.tools.search_definitions import Definition, search, tokenize

TOOL_CASES = [c for c in SAFETY_SET if c["target"] == "tool"]
GLOSSARY = [
    Definition("older singer", "a singer whose age is above 40", ("senior singer",)),
    Definition("recent concert", "a concert held in 2015 or later"),
    Definition("popular singer", "a singer who performed in at least 2 concerts"),
]


@pytest.fixture
def ctx(tiny_db):
    return ToolContext(tiny_db, GLOSSARY)


def test_compact_catalog_lists_names_only(tiny_db):
    assert compact_catalog(tiny_db).splitlines()[0] == "singer(singer_id, name, country, age)"
    assert "CREATE TABLE" in create_table_text(tiny_db)


def test_schema_info_reads_keys_and_rows(tiny_db):
    s = schema_info(str(tiny_db))
    assert s.tables["singer"].rows == 4
    assert s.tables["singer"].column("singer_id").primary_key
    assert ("singer_id", "singer", "singer_id") in s.tables["singer_in_concert"].foreign_keys


def test_describe_table(ctx):
    r = run_tool(ctx, "describe_table", {"table": "SINGER"})  # names are matched case-insensitively
    assert r.ok and r.args["table"] == "singer"
    assert "singer_id integer PK" in r.text and "(4 rows)" in r.text
    assert "referenced by singer_in_concert.singer_id" in r.text


def test_sample_values_most_frequent_first(ctx):
    r = run_tool(ctx, "sample_values", {"table": "singer", "column": "country", "limit": 2})
    assert r.ok and r.text == "singer.country most frequent values: 'France' (2), 'China' (1)"


def test_search_definitions_finds_the_right_entry(ctx):
    r = run_tool(ctx, "search_definitions", {"query": "older singers"})
    assert r.ok and r.text.splitlines()[0] == "older singer: a singer whose age is above 40"
    r = run_tool(ctx, "search_definitions", {"query": "senior singers"})  # through the alias
    assert r.text.startswith("older singer:")


def test_search_definitions_says_when_nothing_matches(ctx):
    r = run_tool(ctx, "search_definitions", {"query": "best stadium"})
    assert r.ok and r.text.startswith("no stored definition")


def test_bm25_prefers_rarer_shared_words():
    hits = search(GLOSSARY, "recent concert")
    assert hits[0][0].term == "recent concert"
    assert tokenize("The Students' marks") == ["student", "mark"]


@pytest.mark.parametrize("case", TOOL_CASES, ids=lambda c: f"{c['id']}-{c['tool']}")
def test_safety_set_bad_arguments_are_rejected(case, ctx, tiny_db):
    before = hashlib.sha256(tiny_db.read_bytes()).hexdigest()
    r = run_tool(ctx, case["tool"], case["args"])
    assert not r.ok and r.text.startswith("rejected:"), case["why"]
    assert hashlib.sha256(tiny_db.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("tool,args,message", [
    ("drop_table", {"table": "singer"}, "unknown tool"),
    ("describe_table", {}, "missing argument"),
    ("describe_table", {"table": "singer", "extra": 1}, "unknown argument"),
    ("describe_table", {"table": "singers"}, "Did you mean: singer"),
    ("sample_values", {"table": "singer", "column": "nme"}, "Did you mean: name"),
    ("sample_values", {"table": "singer", "column": "age", "limit": True}, "must be an integer"),
])
def test_validation_messages_help_recovery(ctx, tool, args, message):
    r = run_tool(ctx, tool, args)
    assert not r.ok and message in r.text
