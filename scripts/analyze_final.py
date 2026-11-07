"""Validate the final result matrix, write tables and paired database confidence intervals."""

import argparse
import csv
import json
from pathlib import Path
from clarifysql.episodes.schema import load_episodes
from clarifysql.eval.artifacts import read_jsonl, write_json
from clarifysql.eval.bootstrap import paired_bootstrap
from clarifysql.eval.metrics import summarize

NAMES = ("b1", "b2", "oracle", "b3", "sft_s1", "sft_s2", "a2", "a3")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/final"))
    ap.add_argument("--repeats", type=int, default=5000)
    a = ap.parse_args()
    expected = {e.id for e in load_episodes("test")}
    runs, summaries = {}, []
    for name in NAMES:
        folder = a.root / f"test_{name}"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        rows = read_jsonl(folder / "traces.jsonl")
        if manifest["identity"]["backend"] != "hf" or not summary["complete"]:
            raise ValueError(f"{folder}: only complete real-model runs can enter final tables")
        if len(rows) != len(expected) or {r["episode"] for r in rows} != expected:
            raise ValueError(f"{folder}: missing, duplicate or extra test episodes")
        runs[name] = rows
        summaries.append({"system": name, **{k: summary.get(k) for k in ["execution_accuracy",
            "decision_macro_f1", "testsuite_accuracy", "unnecessary_question_rate_clear",
            "correct_abstention_rate", "avg_clarification_turns", "slot_accuracy", "total_model_calls"]}})
    table_dir = a.root / "analysis"
    table_dir.mkdir(exist_ok=True)
    with (table_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    stats = {}
    for metric in ("execution_accuracy", "decision_macro_f1"):
        stats[metric] = {name: paired_bootstrap(runs["b3"], [runs[name]], metric, a.repeats)
                         for name in ("sft_s1", "sft_s2")}
        stats[metric]["two_seed_mean"] = paired_bootstrap(runs["b3"], [runs["sft_s1"], runs["sft_s2"]],
                                                        metric, a.repeats)
        stats[metric]["exploratory_a2_minus_sft_s1"] = paired_bootstrap(runs["sft_s1"], [runs["a2"]], metric, a.repeats)
        stats[metric]["exploratory_a3_minus_sft_s1"] = paired_bootstrap(runs["sft_s1"], [runs["a3"]], metric, a.repeats)
    write_json(table_dir / "paired_bootstrap.json", stats)
    kinds = sorted({r["kind"] for r in runs["b3"]})
    write_json(table_dir / "by_kind.json", {name: {kind: summarize([r for r in rows if r["kind"] == kind])
                                                for kind in kinds} for name, rows in runs.items()})
    from clarifysql.eval.cosql_slice import load_slice
    cosql_ids = {r["episode"] for r in load_slice()}
    cosql = {}
    for name in ("b3", "sft_s1"):
        folder = a.root / f"cosql_{name}"
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        rows = read_jsonl(folder / "traces.jsonl")
        if summary["backend"] != "hf" or not summary["complete"] or len(rows) != len(cosql_ids) or {r["episode"] for r in rows} != cosql_ids:
            raise ValueError(f"{folder}: incomplete real-dialogue comparison")
        cosql[name] = summary
    write_json(table_dir / "cosql.json", cosql)
    lines = ["# Final test results", "", "All conditions cover the same reviewed test episodes.", "",
             "| System | Execution accuracy | Decision macro-F1 | Test-suite accuracy |",
             "|---|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['system']} | {s['execution_accuracy']} | {s['decision_macro_f1']} | "
                     f"{s['testsuite_accuracy'] if s['testsuite_accuracy'] is not None else 'pending'} |")
    lines += ["", "Confirmatory comparisons: SFT versus B3 on execution accuracy and decision macro-F1.",
              "Database-level intervals and both seeds are in paired_bootstrap.json.",
              "Question usefulness and simulator accuracy require the separate human review sheets."]
    (table_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", table_dir)


if __name__ == "__main__":
    main()
