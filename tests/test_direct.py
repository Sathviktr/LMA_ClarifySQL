import pytest

from clarifysql.llm.backend import MockBackend
from clarifysql.llm.prompts import extract_sql, repair_messages, sql_writer_messages
from clarifysql.systems.direct import Item, run_direct, summarize


@pytest.mark.parametrize("text,expected", [
    ("```sql\nSELECT name FROM singer;\n```", "SELECT name FROM singer"),
    ("Here you go:\n```sqlite\nSELECT 1\n```\nThis returns one.", "SELECT 1"),
    ("SELECT name FROM singer WHERE name = 'a;b'; DROP TABLE singer", "SELECT name FROM singer WHERE name = 'a;b'"),
    ("The answer is SELECT count(*) FROM singer", "SELECT count(*) FROM singer"),
    ("```sql\nWITH t AS (SELECT 1) SELECT * FROM t\n```", "WITH t AS (SELECT 1) SELECT * FROM t"),
])
def test_extract_sql(text, expected):
    assert extract_sql(text) == expected


def test_prompts_contain_schema_and_request():
    msgs = sql_writer_messages("CREATE TABLE singer(name TEXT);", "List all singers.")
    assert msgs[0]["role"] == "system" and "one SQLite query" in msgs[0]["content"]
    assert "CREATE TABLE singer" in msgs[1]["content"] and msgs[1]["content"].endswith("Request: List all singers.")
    rep = repair_messages("S", "R", "SELECT nme FROM singer", "no such column: nme")
    assert rep[-2]["role"] == "assistant" and "no such column" in rep[-1]["content"]


def test_run_direct_scores_and_repairs(tiny_db):
    items = [Item("a", "concert", "How many singers are there?", "SELECT count(*) FROM singer"),
             Item("b", "concert", "Names of French singers?", "SELECT name FROM singer WHERE country = 'France'")]
    answers = {"How many singers are there?": ["```sql\nSELECT count(*) FROM singer\n```"],
               "Names of French singers?": ["```sql\nSELECT nme FROM singer WHERE country = 'France'\n```",
                                            "```sql\nSELECT name FROM singer WHERE country = 'France'\n```"]}

    def respond(chat, adapter):
        assert adapter is False  # the SQL writer always runs with the adapter switched off
        request = chat[1]["content"].split("Request: ")[-1]
        return answers[request].pop(0) if len(answers[request]) > 1 else answers[request][0]

    b1 = run_direct(items, MockBackend(respond), lambda db: tiny_db, max_repairs=0)
    assert [r["correct"] for r in b1] == [True, False] and b1[1]["reason"] == "pred_sql_error"

    answers["Names of French singers?"].insert(0, "```sql\nSELECT nme FROM singer WHERE country = 'France'\n```")
    b2 = run_direct(items, MockBackend(respond), lambda db: tiny_db, max_repairs=2)
    assert [r["correct"] for r in b2] == [True, True] and b2[1]["repairs"] == 1 and b2[1]["model_calls"] == 2
    assert [r["first_correct"] for r in b2] == [True, False]  # the answer before repair is kept (B1 view)
    s = summarize(b2)
    assert s["execution_accuracy"] == 1.0 and s["execution_accuracy_before_repair"] == 0.5
    assert s["repaired_items"] == 1
