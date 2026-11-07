"""Save reproducible demo checks with predefined replies, separately from benchmarks and live users."""

from __future__ import annotations

import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path

from clarifysql.agent.controller import trace
from clarifysql.config import MODEL_ID, REPO_ROOT
from clarifysql.db.safe_exec import execute
from clarifysql.demo.session import LiveSession
from clarifysql.demo.university import examples
from clarifysql.eval.artifacts import RunStore, digest, write_json
from clarifysql.eval.exec_match import exec_match
from clarifysql.llm.backend import HFBackend, MockBackend


@dataclass(frozen=True)
class DemoCase:
    id: str
    kind: str
    request: str
    expected_sql: str | None
    expected_decision: str
    reply_mode: str = "none"
    ranking_reply: str = ""
    age: int = 22
    mark: int = 85


def demo_cases() -> list[DemoCase]:
    cases = [DemoCase(ep.id, ep.kind, ep.request, ep.gold_sql, ep.first_decision) for ep in examples()]
    clear, definition, ranking, thresholds, missing = cases
    ranking = replace(ranking, reply_mode="ranking", ranking_reply="Use the highest average mark per department.")
    placement = replace(ranking, id="demo-ranking-placements",
        ranking_reply="Use the highest placement rate: placed students divided by all students in the department.",
        expected_sql="SELECT d.name FROM departments d JOIN students s ON s.department_id=d.id "
                     "JOIN placements p ON p.student_id=s.id GROUP BY d.id "
                     "ORDER BY 1.0*SUM(p.placed)/COUNT(*) DESC LIMIT 1")
    thresholds = replace(thresholds, reply_mode="thresholds")
    changed_age = replace(thresholds, id="demo-two-slot-age20", age=20,
                         expected_sql=thresholds.expected_sql.replace("age > 22", "age > 20"))
    return [clear, definition, ranking, placement, thresholds, changed_age, missing]


def scripted_reply(case: DemoCase, question: str) -> str | None:
    """Only the two ambiguous case types have replies. No language-model user is called."""
    if case.reply_mode == "ranking":
        return case.ranking_reply
    if case.reply_mode == "thresholds":
        age = bool(re.search(r"\b(age|ages|older|old|young|younger)\b", question, re.I))
        marks = bool(re.search(r"\b(mark|marks|score|scores|strong|strength)\b", question, re.I))
        parts = ([f"Use age above {case.age}."] if age else [])
        parts += [f"Use marks above {case.mark}."] if marks else []
        return " ".join(parts) or None
    return None


def check_case(backend, database: Path, case: DemoCase, *, adapter: bool) -> dict:
    """Reference SQL is used after execution, never supplied to the policy or writer."""
    database = Path(database)
    before = digest(database)
    session = LiveSession(backend, database, case.request, adapter=adapter)
    replies, error = [], None
    started = time.perf_counter()
    try:
        view = session.advance()
        for _ in range(session.controller.budgets.max_questions):
            if view["pending"] != "user":
                break
            answer = scripted_reply(case, session.run.question)
            if answer is None:
                error = "No predefined reply for this question; the check stopped without inventing an answer."
                break
            replies.append({"question": session.run.question, "reply": answer, "source": "predefined_demo_script"})
            view = session.advance(answer)
    except Exception as exc:
        # Retain the case and continue; a failed model call is an observed failure, not a passing result.
        error = f"{type(exc).__name__}: {exc}"
    run = session.run
    after = digest(database)
    checks = {"finished": run.pending == "done", "expected_first_decision": run.decision == case.expected_decision,
              "no_runtime_error": error is None, "database_unchanged": before == after}
    observed = execute(database, run.sql) if run.sql and run.sql_status == "ok" else None
    result_reason = None
    if case.expected_sql:
        match = exec_match(database, run.sql or "", case.expected_sql)
        checks["result_matches_reference"] = run.outcome == "executed" and match.correct
        result_reason = match.reason
    else:
        checks["named_missing_email"] = run.outcome == "abstained" and bool(re.search(r"e[- ]?mail", run.abstain_text or "", re.I))
    if case.expected_decision == "clarify":
        checks["reply_recorded"] = bool(replies) and run.questions == len(replies)
    else:
        checks["no_unnecessary_question"] = not any(s["action"]["action"] == "clarify" for s in run.steps)
    if case.kind == "definition":
        checks["definition_retrieved"] = any(s["action"].get("tool") == "search_definitions" and s.get("tool_ok")
            and "80" in (s.get("result") or "") for s in run.steps)
    record = trace(run)
    record.update({"episode": case.id, "kind": case.kind, "gold_decision": case.expected_decision,
                   "reply_source": "predefined_demo_script", "scripted_replies": replies,
                   "expected_sql": case.expected_sql, "result_columns": observed.columns if observed else [],
                   "result_rows": observed.rows if observed else [], "result_match_reason": result_reason,
                   "checks": checks, "passed": all(checks.values()), "runtime_error": error,
                   "database_before_sha256": before, "database_after_sha256": after,
                   "wall_seconds": round(time.perf_counter()-started, 3)})
    return record


def _source_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment(backend):
    libraries = {}
    for name in ("torch", "transformers", "peft", "gradio"):
        try:
            libraries[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libraries[name] = None
    out = {"python": platform.python_version(), "platform": platform.platform(), "libraries": libraries,
           "source_commit": _source_commit()}
    if isinstance(backend, HFBackend):
        import torch
        out.update({"model_revision": getattr(backend.model.config, "_commit_hash", None),
                    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    "load_seconds": backend.load_seconds})
    return out


def run_demo_checks(backend, database: Path, output: Path, *, adapter_path: Path | None = None) -> dict:
    """Resume exact matching cases; mock output has its own directory and is explicitly labelled."""
    if isinstance(backend, HFBackend):
        backend_name = "hf"
        if not backend.has_adapter or adapter_path is None:
            raise ValueError("Real demo checks require the trained adapter and its directory.")
        adapter_path = Path(adapter_path)
        weights = {p.name: digest(p) for p in sorted(adapter_path.iterdir()) if p.is_file()}
        if not {"adapter_config.json", "adapter_model.safetensors"} <= weights.keys():
            raise ValueError("The adapter directory is incomplete.")
    elif isinstance(backend, MockBackend):
        backend_name, weights = "mock", {}
    else:
        raise TypeError("Use HFBackend for real outputs or MockBackend for software checks.")
    cases = demo_cases()
    source = sorted((REPO_ROOT / "src/clarifysql").rglob("*.py"))
    environment = _environment(backend)
    if backend_name == "hf" and not environment.get("model_revision"):
        raise ValueError("The loaded base-model revision is unknown; record it before saving a reproducible run.")
    identity = {"protocol": "checkpoint8-demo-v1", "backend": backend_name, "model": MODEL_ID,
                "model_revision": environment.get("model_revision"),
                "inference_libraries": {k: v for k, v in environment["libraries"].items() if k != "gradio"},
                "model_dtype": str(backend.model.dtype) if backend_name == "hf" else None,
                "four_bit": bool(getattr(backend.model, "is_loaded_in_4bit", False)) if backend_name == "hf" else False,
                "cases": [asdict(case) for case in cases], "database_sha256": digest(database),
                "adapter_files": weights, "decoding": "greedy", "reply_source": "predefined_demo_script",
                "source_files": {p.relative_to(REPO_ROOT).as_posix(): digest(p) for p in source}}
    folder = Path(output) / "mock" if backend_name == "mock" else Path(output)
    store = RunStore(folder, identity)
    ids = [case.id for case in cases]
    if not store.completed <= set(ids):
        raise ValueError("Saved demo contains unexpected case IDs.")
    if not (folder / "environment.json").exists():
        write_json(folder / "environment.json", environment)
    for case in cases:
        if case.id in store.completed:
            continue
        row = check_case(backend, database, case, adapter=backend_name == "hf")
        store.append([row])
        print(f"{case.id}: {'PASS' if row['passed'] else 'FAIL'} ({len(store.rows)}/{len(cases)} saved)", flush=True)
        if not row["checks"]["database_unchanged"]:
            raise RuntimeError("The demo database changed; stop and inspect the saved trace.")
    rows = sorted(store.rows, key=lambda r: ids.index(r["episode"]))
    summary = {"protocol": "checkpoint8-demo-v1", "backend": backend_name, "complete": len(rows) == len(cases),
               "cases": len(rows), "passed": sum(r["passed"] for r in rows),
               "failed": [r["episode"] for r in rows if not r["passed"]],
               "reply_source": "predefined_demo_script", "interactive_ui_tested": False,
               "scope": "Seven fixed fictional-data checks. This is a scripted demo check, not a benchmark estimate or a live-user study.",
               "strict_result_comparison": "Result rows must match the reference; extra columns count as a mismatch.",
               "cases_detail": [{"episode": r["episode"], "passed": r["passed"], "checks": r["checks"],
                                 "runtime_error": r["runtime_error"]} for r in rows]}
    write_json(folder / "summary.json", summary)
    lines = ["# Demo checks", "", f"Backend: {backend_name}. Passed: {summary['passed']}/{len(cases)}.",
             "Predefined replies are used; the interactive interface has not been tested by this script.", "",
             "| Case | Result | Failed checks |", "|---|---|---|"]
    for r in rows:
        failures = ", ".join(k for k, value in r["checks"].items() if not value)
        lines.append(f"| {r['episode']} | {'PASS' if r['passed'] else 'FAIL'} | {failures or '-'} |")
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
