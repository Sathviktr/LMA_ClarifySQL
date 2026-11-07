"""Check scripted reply handling, failure evidence and separation from real model outputs."""

import json
import sqlite3
import sys
import types
import zipfile

import pytest

from clarifysql.demo.checks import check_case, demo_cases, run_demo_checks, scripted_reply
from clarifysql.demo.university import create_database
from clarifysql.llm.backend import MockBackend


def test_replies_only_reveal_the_requested_threshold():
    case = next(c for c in demo_cases() if c.id == "demo-two-slot-age20")
    assert scripted_reply(case, "What age counts as older?") == "Use age above 20."
    assert scripted_reply(case, "What marks count as strong?") == "Use marks above 85."
    assert scripted_reply(case, "What age and marks should I use?") == "Use age above 20. Use marks above 85."
    assert scripted_reply(case, "Which file should I read?") is None
    assert scripted_reply(demo_cases()[0], "What threshold?") is None


def test_correct_sql_cannot_hide_a_missing_clarification(tmp_path):
    case = next(c for c in demo_cases() if c.id == "demo-ambiguous")
    outputs = iter([json.dumps({"action": "execute", "request": "Return the best department by average mark."}),
                    case.expected_sql])
    def reply(chat, adapter):
        assert not any(case.expected_sql in m["content"] for m in chat)  # scoring SQL must stay out of prompts
        return next(outputs)
    row = check_case(MockBackend(reply), create_database(tmp_path / "demo.sqlite"), case, adapter=False)
    assert row["checks"]["result_matches_reference"]
    assert not row["passed"] and not row["checks"]["expected_first_decision"]
    assert not row["checks"]["reply_recorded"] and row["scripted_replies"] == []


def test_marks_first_then_changed_age_are_remembered(tmp_path):
    case = next(c for c in demo_cases() if c.id == "demo-two-slot-age20")
    outputs = iter([json.dumps({"action": "clarify", "question": "What marks count as strong?"}),
                    json.dumps({"action": "clarify", "question": "What age counts as older?"}),
                    json.dumps({"action": "execute", "request": "List students older than 20 with marks above 85."}),
                    case.expected_sql])
    row = check_case(MockBackend(lambda c, a: next(outputs)), create_database(tmp_path / "demo.sqlite"), case, adapter=False)
    assert row["passed"] and row["questions"] == 2
    assert set(row["result_rows"]) == {("Asha",), ("Dana",), ("Fara",)}
    assert [r["reply"] for r in row["scripted_replies"]] == ["Use marks above 85.", "Use age above 20."]
    assert row["database_before_sha256"] == row["database_after_sha256"]


def test_runtime_failure_and_unknown_question_are_saved_as_failures(tmp_path):
    db = create_database(tmp_path / "demo.sqlite")
    def failed(chat, adapter):
        raise RuntimeError("model call failed")
    row = check_case(MockBackend(failed), db, demo_cases()[0], adapter=False)
    assert not row["passed"] and row["runtime_error"] == "RuntimeError: model call failed"
    case = next(c for c in demo_cases() if c.id == "demo-two-slot-age20")
    backend = MockBackend(lambda c, a: '{"action":"clarify","question":"Which file should I read?"}')
    row = check_case(backend, db, case, adapter=False)
    assert not row["passed"] and "No predefined reply" in row["runtime_error"]
    assert len(backend.calls) == 1 and not row["scripted_replies"]


def test_all_cases_export_mock_only_resume_and_refuse_changed_database(tmp_path):
    db = create_database(tmp_path / "demo.sqlite")
    outputs = []
    for case in demo_cases():
        if case.kind == "definition":
            outputs.append(json.dumps({"action": "inspect", "tool": "search_definitions", "args": {"query": "high marks"}}))
        elif case.expected_decision == "clarify":
            question = "Which ranking measure?" if case.reply_mode == "ranking" else "What age and marks should I use?"
            outputs.append(json.dumps({"action": "clarify", "question": question}))
        if case.expected_sql:
            outputs += [json.dumps({"action": "execute", "request": "Return the requested records."}), case.expected_sql]
        else:
            outputs.append(json.dumps({"action": "abstain", "missing": "Student email addresses are not stored."}))
    replies = iter(outputs)
    backend = MockBackend(lambda c, a: next(replies))
    output = tmp_path / "checks"
    summary = run_demo_checks(backend, db, output)
    assert summary["complete"] and summary["passed"] == summary["cases"] == 7
    assert summary["backend"] == "mock" and summary["interactive_ui_tested"] is False
    assert (output / "mock/traces.jsonl").exists() and not (output / "summary.json").exists()
    calls = len(backend.calls)
    assert run_demo_checks(backend, db, output) == summary and len(backend.calls) == calls
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO departments VALUES (9,'New fictional department')")
    with pytest.raises(ValueError, match="changed"):
        run_demo_checks(backend, db, output)


def test_notebook_packages_failures_even_when_interface_construction_fails(tmp_path, monkeypatch):
    import nbformat
    from clarifysql.config import REPO_ROOT
    from clarifysql.demo import app as app_module
    from clarifysql.eval.artifacts import write_json
    notebook = nbformat.read(REPO_ROOT / "notebooks/08_demo.ipynb", as_version=4)
    package = next(c.source for c in notebook.cells if c.cell_type == "code" and "with zipfile.ZipFile(ZIP" in c.source)
    optional = next(c.source for c in notebook.cells if c.cell_type == "code" and "if RUN_INTERACTIVE:" in c.source)
    demo = tmp_path / "checkpoint8_demo"
    saved = {"backend": "hf", "complete": True, "passed": 0, "fixture": "synthetic packaging test"}
    write_json(demo / "checks/summary.json", saved)
    (demo / "checks/traces.jsonl").write_text('{"episode":"test-fixture","passed":false}\n')
    def fail_ui(*args):
        raise RuntimeError("synthetic interface construction error")
    def no_input(*args):
        raise AssertionError("A saved default run must not request interactive input")
    monkeypatch.setattr(app_module, "create_app", fail_ui)
    monkeypatch.setattr("builtins.input", no_input)
    monkeypatch.setenv("CLARIFYSQL_DEMO_PASSWORD", "synthetic-test-password")
    # Notebook display helpers are not needed by the CPU development environment.
    display = types.ModuleType("IPython.display")
    display.FileLink = lambda path: path
    display.display = lambda value: None
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    scope = {"WORK": tmp_path, "DEMO": demo, "model": object(), "db": tmp_path / "demo.sqlite",
             "zipfile": zipfile, "write_json": write_json, "RUN_INTERACTIVE": False}
    exec(compile(package, "08_demo:packaging", "exec"), scope)
    exec(compile(optional, "08_demo:optional-interface", "exec"), scope)
    with zipfile.ZipFile(tmp_path / "checkpoint8_results.zip") as archive:
        assert json.loads(archive.read("checks/summary.json")) == saved
        interface = json.loads(archive.read("interface.json"))
        assert interface["constructed"] is False and "synthetic interface" in interface["error"]
        assert interface["browser_interaction_checked"] is False
        assert "checks/traces.jsonl" in archive.namelist()
        assert not any(b"synthetic-test-password" in archive.read(name) for name in archive.namelist())
