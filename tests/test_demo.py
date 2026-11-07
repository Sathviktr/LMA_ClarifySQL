"""Exercise the real controller with a person-facing session on fictional data."""

import hashlib
import pytest
from clarifysql.demo.university import create_database
from clarifysql.demo.session import LiveSession
from clarifysql.llm.backend import MockBackend
from clarifysql.db.safe_exec import execute


@pytest.mark.parametrize("index", range(5))
def test_fictional_examples_cover_the_five_request_types(tmp_path, index):
    from clarifysql.agent.controller import Controller
    from clarifysql.agent.mocks import ReferencePolicy, GoldWriter, KeywordUser
    from clarifysql.simuser.simulated_user import SimulatedUser
    from clarifysql.demo.university import examples
    ep = examples()[index]
    db = create_database(tmp_path / "university.sqlite")
    controller = Controller(ReferencePolicy([ep]), GoldWriter([ep]), SimulatedUser(KeywordUser()),
                            policy_adapter=False, db_path_of=lambda _: db)
    run = controller.run([ep])[0]
    if ep.kind == "unanswerable":
        assert run.outcome == "abstained" and "email" in run.abstain_text
    else:
        assert run.sql_status == "ok" and run.sql == ep.gold_sql
        assert execute(db, run.sql).rows
    if ep.kind == "definition":
        assert run.questions == 0 and run.inspections == 1
    if ep.kind == "two_slot":
        assert run.questions == 2


def test_real_reply_is_remembered_and_sql_is_repaired(tmp_path):
    db = create_database(tmp_path / "university.sqlite")
    replies = iter(['{"action":"clarify","question":"What age counts as older?"}',
                    '{"action":"execute","request":"List students older than 22."}',
                    'SELECT nam FROM students WHERE age > 22', 'SELECT name FROM students WHERE age > 22'])
    session = LiveSession(MockBackend(lambda c, a: next(replies)), db, "Show older students.", adapter=False)
    assert session.advance()["pending"] == "user"
    view = session.advance("Above 22")
    assert view["pending"] == "done" and ("Dana",) in view["rows"]
    assert session.run.repairs == 1 and "Above 22" in view["history"]


def test_demo_denies_sample_values_and_database_writes(tmp_path):
    db = create_database(tmp_path / "university.sqlite")
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    actions = iter(['{"action":"inspect","tool":"sample_values","args":{"table":"students","column":"name"}}',
                    '{"action":"execute","request":"Delete students."}', 'DELETE FROM students'])
    session = LiveSession(MockBackend(lambda c, a: next(actions)), db, "Delete students.", adapter=False)
    view = session.advance()
    assert session.run.steps[0]["result"] == "This tool is unavailable in this session."
    assert session.run.sql_status == "blocked" and not view["rows"]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_demo_stops_after_budget_and_explains_missing_information(tmp_path):
    db = create_database(tmp_path / "university.sqlite")
    asker = MockBackend(lambda c, a: '{"action":"clarify","question":"What age?"}')
    session = LiveSession(asker, db, "Older students", adapter=False)
    session.advance()
    session.advance("22")
    view = session.advance("22")
    assert session.run.outcome == "limit" and session.run.questions == 2
    missing = LiveSession(MockBackend(lambda c, a: '{"action":"abstain","missing":"No email addresses are stored."}'),
                          db, "Student emails", adapter=False)
    assert "No email" in missing.advance()["message"]
