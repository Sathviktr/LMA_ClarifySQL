"""Verify the received checkpoint-8 archive and inspect saved SQL on changed fictional data (CPU)."""

import argparse
import hashlib
import io
import json
import re
import sqlite3
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from clarifysql.agent.actions import parse_action
from clarifysql.config import REPO_ROOT
from clarifysql.demo.checks import demo_cases, scripted_reply
from clarifysql.eval.artifacts import digest, json_hash, write_json
from clarifysql.eval.exec_match import exec_match

FILES = {"checks/environment.json", "checks/manifest.json", "checks/report.md", "checks/summary.json",
         "checks/traces.jsonl", "interface.json", "source.json", "university.sqlite"}


def read_bundle(path):
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        if len({i.filename for i in entries}) != len(entries):
            raise ValueError("Duplicate archive paths")
        for entry in entries:
            p = PurePosixPath(entry.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in entry.filename or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Unsafe archive path")
        if {i.filename for i in entries} != FILES or sum(i.file_size for i in entries) > 10_000_000:
            raise ValueError("Unexpected archive contents or size")
        return {i.filename: z.read(i) for i in entries}


def verify_source(revision, files):
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Expected the full recorded Git commit")
    names = subprocess.run(["git", "ls-tree", "-rz", "--name-only", revision, "--", "src/clarifysql"],
                           cwd=REPO_ROOT, check=True, capture_output=True).stdout
    paths = sorted(p.decode("utf-8") for p in names.split(b"\0") if p.endswith(b".py"))
    if set(paths) != set(files):
        raise ValueError("Source file set differs from the recorded Git revision")
    # Read Git blobs directly: an archive or working tree can convert LF to CRLF on Windows.
    requests = "".join(f"{revision}:{p}\n" for p in paths).encode()
    data = subprocess.run(["git", "cat-file", "--batch"], input=requests, cwd=REPO_ROOT,
                          check=True, capture_output=True).stdout
    stream, actual = io.BytesIO(data), {}
    for path in paths:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("Could not read a recorded source blob")
        content = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("Truncated source blob stream")
        actual[path] = hashlib.sha256(content).hexdigest()
    if actual != files:
        raise ValueError("Source hashes do not match the recorded Git revision")


def placement_variants(database, predicted_sql, reference_sql, work):
    """Two explicit post-hoc counterexamples; no claim about a population of databases."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    changes = {"mathematics_best": {3, 4}, "physics_best": {5, 6}}
    results = []
    for name, placed_ids in changes.items():
        destination = work / f"{name}.sqlite"
        if destination.exists():
            raise ValueError("Refusing to replace an existing variant database")
        source = sqlite3.connect(f"{Path(database).resolve().as_uri()}?mode=ro", uri=True)
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
            target.execute("UPDATE placements SET placed=0")
            target.executemany("UPDATE placements SET placed=1 WHERE student_id=?", [(i,) for i in sorted(placed_ids)])
            target.commit()
        finally:
            source.close()
            target.close()
        match = exec_match(destination, predicted_sql, reference_sql)
        results.append({"variant": name, "placed_student_ids": sorted(placed_ids), "matches": match.correct,
                        "prediction_status": match.pred.status, "predicted_rows": match.pred.rows,
                        "reference_status": match.gold.status, "expected_rows": match.gold.rows})
    return results


def verify_bundle(archive, output):
    payload = read_bundle(archive)
    read = lambda name: json.loads(payload[name])
    manifest, environment, source = (read(n) for n in ("checks/manifest.json", "checks/environment.json", "source.json"))
    config, saved = manifest["identity"], read("checks/summary.json")
    cases = [asdict(case) for case in demo_cases()]
    if manifest["fingerprint"] != json_hash(config) or config["backend"] != "hf" or config["cases"] != cases:
        raise ValueError("Invalid manifest, mock backend or changed case definitions")
    revision = source["source"].get("commit")
    if environment["source_commit"] != revision or environment["model_revision"] != config["model_revision"]:
        raise ValueError("Inconsistent source or model revisions")
    if config["inference_libraries"] != {k: environment["libraries"][k] for k in config["inference_libraries"]}:
        raise ValueError("Inference-library versions differ between manifest and environment")
    verify_source(revision, config["source_files"])
    raw_db_hash = hashlib.sha256(payload["university.sqlite"]).hexdigest()
    if raw_db_hash != config["database_sha256"]:
        raise ValueError("The received database hash does not match the manifest")
    baseline = REPO_ROOT / "results/final/test_sft_s1/manifest.json"
    adapter_check = "not checked: checkpoint-7 manifest unavailable"
    if baseline.exists():
        prior = json.loads(baseline.read_text(encoding="utf-8"))["identity"]["adapter_files"]
        if config["adapter_files"] != prior:
            raise ValueError("The demo adapter differs from checkpoint-7 seed 1")
        baseline_environment = json.loads((baseline.parent / "environment.json").read_text(encoding="utf-8"))
        if environment["model_revision"] != baseline_environment["model_revision"]:
            raise ValueError("The demo base model differs from checkpoint 7")
        adapter_check = "matches checkpoint-7 seed 1"
    rows = [json.loads(line) for line in payload["checks/traces.jsonl"].decode("utf-8").splitlines() if line.strip()]
    by_id = {r["episode"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {c["id"] for c in cases}:
        raise ValueError("Incomplete or duplicate demo cases")
    scratch_root = REPO_ROOT / "outputs"
    scratch_root.mkdir(exist_ok=True)
    reproduced = []
    with tempfile.TemporaryDirectory(dir=scratch_root, prefix="verify_demo_") as temporary:
        work = Path(temporary).resolve()
        assert work.is_relative_to(scratch_root.resolve())
        database = work / "university.sqlite"
        database.write_bytes(payload["university.sqlite"])
        for case in cases:
            row = by_id[case["id"]]
            if (row["request"], row["kind"], row["gold_decision"], row["expected_sql"]) != (
                    case["request"], case["kind"], case["expected_decision"], case["expected_sql"]):
                raise ValueError("Trace metadata differs from the frozen case")
            actions = [parse_action(e["raw"])[0] for e in row["events"] if e["role"] == "policy"]
            first = next((a["action"] for a in actions if a and a["action"] != "inspect"), "none")
            users = [e for e in row["events"] if e["role"] == "user"]
            if first != row["decision"] or len(users) != row["questions"]:
                raise ValueError("Decision or reply count differs from the recorded events")
            if row["calls"] != sum(e["role"] in ("policy", "writer") for e in row["events"]):
                raise ValueError("Model-call count differs from the event log")
            expected_replies = [{"question": e["question"], "reply": scripted_reply(
                next(c for c in demo_cases() if c.id == case["id"]), e["question"]),
                "source": "predefined_demo_script"} for e in users]
            if expected_replies != row["scripted_replies"] or any(e["reply"] != r["reply"] for e, r in zip(users, expected_replies)):
                raise ValueError("Replies do not match the declared demo script")
            checks = {"finished": row["outcome"] in ("executed", "abstained", "invalid", "limit"),
                      "expected_first_decision": first == case["expected_decision"],
                      "no_runtime_error": row["runtime_error"] is None,
                      "database_unchanged": row["database_before_sha256"] == row["database_after_sha256"] == raw_db_hash}
            if case["expected_sql"]:
                match = exec_match(database, row["sql"] or "", case["expected_sql"])
                checks["result_matches_reference"] = row["outcome"] == "executed" and match.correct
                observed = [list(r) for r in match.pred.rows] if match.pred.ok and row["sql_status"] == "ok" else []
                if row["result_rows"] != observed:
                    raise ValueError("Saved result rows do not reproduce")
            else:
                checks["named_missing_email"] = row["outcome"] == "abstained" and bool(re.search(r"e[- ]?mail", row["abstain_text"] or "", re.I))
            if case["expected_decision"] == "clarify":
                checks["reply_recorded"] = bool(users) and row["questions"] == len(expected_replies)
            else:
                checks["no_unnecessary_question"] = not any(s["kind"] == "clarify" for s in row["steps"])
            if case["kind"] == "definition":
                checks["definition_retrieved"] = any(s["kind"] == "inspect" and s.get("tool_ok")
                    and json.loads(s["action"]).get("tool") == "search_definitions"
                    and "80" in (s["result"] or "") for s in row["steps"])
            if checks != row["checks"] or all(checks.values()) != row["passed"]:
                raise ValueError("Saved case checks do not reproduce")
            reproduced.append({"episode": case["id"], "passed": row["passed"], "checks": checks,
                               "runtime_error": row["runtime_error"]})
        actual = {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
                  "failed": [r["episode"] for r in rows if not r["passed"]], "cases_detail": reproduced,
                  "complete": len(rows) == len(cases)}
        if any(saved[k] != v for k, v in actual.items()):
            raise ValueError("Saved summary does not match the independently recomputed checks")
        placement = by_id["demo-ranking-placements"]
        variants = placement_variants(database, placement["sql"], placement["expected_sql"], work / "variants")
        assert digest(database) == raw_db_hash
    verification = {"archive_sha256": digest(archive), "source_commit": revision,
                    "source_hashes": "verified against recorded Git revision", "database_sha256": raw_db_hash,
                    "model_revision": config["model_revision"], "adapter": adapter_check,
                    "original_cases_reproduced": len(rows), "original_passed": actual["passed"],
                    "original_failed": actual["failed"], "model_calls": sum(r["calls"] for r in rows),
                    "generation_seconds": round(sum(r["seconds"] for r in rows), 3),
                    "interface": read("interface.json"),
                    "scope": "Frozen original checks and result rows reproduced on CPU. Interface construction is reported from the archive, not rerun locally. No model inference or browser interaction was performed here."}
    counterfactual = {"case": "demo-ranking-placements", "original_result_match": placement["checks"]["result_matches_reference"],
                      "predicted_sql": placement["sql"], "reference_sql": placement["expected_sql"], "variants": variants,
                      "all_variants_match": all(r["matches"] for r in variants),
                      "scope": "Additional CPU checks introduced after inspecting the prediction; they do not replace the original 5/7 result or estimate general performance."}
    write_json(Path(output) / "verification.json", verification)
    write_json(Path(output) / "placement_variants.json", counterfactual)
    lines = ["# Checkpoint 8: measured findings", "", "The original saved model run passes **5/7** fixed checks; that result is preserved.",
             "The seven cases are a scripted demonstration, not a benchmark or human study. All seven result records reproduce on CPU.", "",
             "| Case | Original result | Observation |", "|---|---|---|"]
    observations = ["Correct department names.", "Retrieves the above-80 glossary definition and returns the correct names.",
                    "Returns Mathematics after the fixed average-mark reply, although its offered age/ID options are poorly grounded.",
                    "Returns Computing on the seed data, but its SQL fails the additional placement variants below.",
                    "Does not ask; uses marks above 80 and guesses age above 18 instead of obtaining the missing age cutoff.",
                    "Does not ask, so the changed age-20 reply is never supplied. Repeats the same incorrect query.",
                    "Correctly abstains and identifies missing email addresses."]
    for case, observation in zip(cases, observations):
        lines.append(f"| {case['id']} | {'PASS' if by_id[case['id']]['passed'] else 'FAIL'} | {observation} |")
    lines += ["", "## Placement SQL: a matching answer can still use the wrong rule", "",
              "The prediction orders by COUNT(p.student_id)/COUNT(s.id). Every student has a placement row, including those with placed=0, so all departments receive rate 1. Computing wins the tie on this seed database.",
              "The intended rate uses the placed indicator, for example SUM(p.placed)/COUNT(*), with floating-point division.", "",
              "The additional checks change only the fictional placed flags and retain the original predicted SQL:", "",
              "| Variant | Expected | Predicted | Match |", "|---|---|---|---|"]
    for r in variants:
        expected = ", ".join(str(row[0]) for row in r["expected_rows"])
        predicted = ", ".join(str(row[0]) for row in r["predicted_rows"])
        lines.append(f"| {r['variant']} | {expected} | {predicted} | {r['matches']} |")
    lines += ["", "These are post-hoc counterexamples, not a new benchmark score. The original 5/7 remains unchanged.", "",
              "## Other limitations", "",
              "- Both ranking questions offer age or ID; departments has no age column. A cooperative scripted reply supplies a better criterion than the offered choices. Passing execution does not establish useful question wording.",
              "- The glossary retrieval matches strong marks to high marks, but older still has no supplied cutoff. The policy executes prematurely and the writer supplies 18. Even allowing the glossary match does not resolve age.",
              "- The missing numeric value cannot be caught by a check that only verifies numbers already present in replies, because no reply was requested.",
              "- Gradio construction succeeded. Browser clicks, real replies and sharing have not been demonstrated by this archive.", "",
              f"The run uses {verification['model_calls']} model calls and {verification['generation_seconds']:.1f} elapsed generation seconds; the model load and installation are separate.",
              "Source and adapter hashes match the recorded code and checkpoint-7 seed 1. The base revision matches the frozen test run.", "",
              "## Remaining work", "", "Try the interactive interface, retain these model limitations in the final report, and keep the independent human study status separate. No model retraining or checkpoint-7 rerun is needed to reproduce this analysis."]
    (Path(output) / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verification, counterfactual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=Path("results/demo/checkpoint8/received.zip"))
    ap.add_argument("--output", type=Path, default=Path("results/demo/checkpoint8/analysis"))
    a = ap.parse_args()
    verification, variants = verify_bundle(a.archive, a.output)
    print(f"Verified {verification['original_cases_reproduced']} cases: original {verification['original_passed']}/7 preserved.")
    print("Placement variant matches:", [r["matches"] for r in variants["variants"]])


if __name__ == "__main__":
    main()
