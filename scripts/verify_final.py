"""Independently check received final traces, frozen inputs and saved summary calculations (CPU)."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from clarifysql.config import REPO_ROOT
from clarifysql.data import spider
from clarifysql.episodes.schema import load_episodes
from clarifysql.eval.artifacts import read_jsonl, digest, json_hash, write_json
from clarifysql.eval.metrics import score_trace, summarize
from clarifysql.eval.cosql_slice import load_slice, summarize_slice


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/final"))
    a = ap.parse_args()
    compute = json.loads((a.root / "compute.json").read_text(encoding="utf-8"))
    revision = compute["source"]["commit"]
    episodes = {ep.id: ep for ep in load_episodes("test")}
    cosql = {r["episode"]: r for r in load_slice()}
    blobs, dbs, checks, weight_hashes, base_revisions = {}, {}, {}, {}, set()
    for folder in sorted(p for p in a.root.iterdir() if (p / "manifest.json").is_file()):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        config = manifest["identity"]
        if manifest["fingerprint"] != json_hash(config) or config["backend"] != "hf":
            raise ValueError(f"{folder}: invalid manifest or non-real backend")
        for relative, expected in config["files"].items():
            if relative not in blobs:
                blob = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=REPO_ROOT,
                                      capture_output=True, check=True).stdout
                blobs[relative] = hashlib.sha256(blob).hexdigest()
            if blobs[relative] != expected:
                raise ValueError(f"{folder}: source/input differs from recorded Git revision: {relative}")
        for db_id, expected in config["database_files"].items():
            if db_id not in dbs:
                dbs[db_id] = digest(spider.db_path(db_id))
            if dbs[db_id] != expected:
                raise ValueError(f"{folder}: local database bytes differ: {db_id}")
        rows = read_jsonl(folder / "traces.jsonl")
        ids = [r["episode"] for r in rows]
        truth = episodes if config["split"] == "test" else cosql
        if len(set(ids)) != len(ids) or set(ids) != truth.keys() or set(ids) != set(config["ids"]):
            raise ValueError(f"{folder}: incomplete or duplicate episodes")
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if not summary["complete"]:
            raise ValueError(f"{folder}: incomplete summary")
        if config["split"] == "test":
            rescored = []
            for row in rows:
                ep = episodes[row["episode"]]
                if (row["db_id"], row["kind"], row["gold_decision"]) != (ep.db_id, ep.kind, ep.first_decision):
                    raise ValueError(f"{folder}: episode metadata mismatch")
                scored = score_trace(row, ep)
                for key in ("correct", "reason", "slots_right"):
                    if key in scored and scored[key] != row.get(key):
                        raise ValueError(f"{folder}/{ep.id}: {key} does not reproduce")
                rescored.append(scored)
            calculated = summarize(rescored)
            keys = ("episodes", "by_kind", "execution_accuracy", "decision_macro_f1", "decision_accuracy",
                    "unnecessary_question_rate_clear", "correct_abstention_rate", "slot_accuracy", "total_model_calls")
            ans = [r for r in rows if r["kind"] != "unanswerable"]
            if any(type(r.get("testsuite_correct")) is not bool for r in ans):
                raise ValueError(f"{folder}: missing test-suite flags")
            suite_accuracy = sum(r["testsuite_correct"] for r in ans) / len(ans)
            if not math.isclose(suite_accuracy, summary["testsuite_accuracy"], abs_tol=1e-12):
                raise ValueError(f"{folder}: test-suite summary mismatch")
        else:
            for row in rows:
                expected = cosql[row["episode"]]
                if row["gold_decision"] != expected["gold_decision"] or row["db_id"] != expected["db_id"]:
                    raise ValueError(f"{folder}: reviewed CoSQL label mismatch")
                if row["correct"] != (row["decision"] == row["gold_decision"]):
                    raise ValueError(f"{folder}: decision correctness mismatch")
            calculated = summarize_slice(rows)
            keys = ("turns", "decision_macro_f1", "decision_accuracy", "calls")
        for key in keys:
            if calculated[key] != summary[key]:
                raise ValueError(f"{folder}: {key} summary does not reproduce")
        environment = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
        base_revisions.add(environment.get("model_revision"))
        if config["adapter_files"]:
            weight_hashes[folder.name] = config["adapter_files"]["adapter_model.safetensors"]
        checks[folder.name] = {"records": len(rows), "input_hashes": "verified",
                               "database_hashes": "verified", "summary": "reproduced"}
        print("Verified", folder.name, len(rows), flush=True)
    expected_names = {f"test_{n}" for n in ("b1", "b2", "oracle", "b3", "sft_s1", "sft_s2", "a2", "a3")}
    expected_names |= {"cosql_b3", "cosql_sft_s1"}
    if set(checks) != expected_names or len(base_revisions) != 1 or None in base_revisions:
        raise ValueError("Unexpected condition set or inconsistent base-model revisions")
    if not (weight_hashes["test_sft_s1"] == weight_hashes["test_a3"] == weight_hashes["cosql_sft_s1"]):
        raise ValueError("Seed-1/A3/CoSQL adapter weights differ")
    if len({weight_hashes[f"test_{n}"] for n in ("sft_s1", "sft_s2", "a2")}) != 3:
        raise ValueError("Expected three distinct trained adapters")
    write_json(a.root / "analysis/verification.json", {
        "source_commit": revision, "base_model_revision": next(iter(base_revisions)), "conditions": checks,
        "scope": "Code/data hashes and SQLite execution/summary metrics reproduced locally. Test-suite means recomputed from saved flags; official variant queries were not repeated.",
        "human_reviews": "pending"})


if __name__ == "__main__":
    main()
