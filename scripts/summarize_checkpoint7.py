"""Explain the frozen checkpoint-7 results and export figures without rerunning a model."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from clarifysql.eval.artifacts import read_jsonl, write_json
from clarifysql.eval.metrics import macro_f1

NAMES = ("b1", "b2", "b3", "sft_s1", "sft_s2", "a2", "a3", "oracle")
LABELS = ("B1", "B2", "B3", "SFT 1", "SFT 2", "A2", "A3", "Oracle")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/final"))
    a = ap.parse_args()
    out = a.root / "analysis"
    out.mkdir(exist_ok=True)
    summaries = {n: json.loads((a.root / f"test_{n}/summary.json").read_text(encoding="utf-8")) for n in NAMES}
    traces = {n: read_jsonl(a.root / f"test_{n}/traces.jsonl") for n in NAMES}
    intervals = json.loads((out / "paired_bootstrap.json").read_text(encoding="utf-8"))
    compute = json.loads((a.root / "compute.json").read_text(encoding="utf-8"))
    lines = ["# Checkpoint 7: measured findings", "", "These are the frozen final test results, not validation scores.",
             "All eight conditions cover 188 episodes on 20 databases: 148 answerable and 40 unanswerable.", "",
             "| System | Correct / 148 | Exec. % | Test-suite % | Decision F1 | Ask on clear % | Correct abstention / 40 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for n, label in zip(NAMES, LABELS):
        s, rows = summaries[n], traces[n]
        ans = [r for r in rows if r["kind"] != "unanswerable"]
        unans = [r for r in rows if r["kind"] == "unanswerable"]
        lines.append(f"| {label} | {sum(r['correct'] for r in ans)} | {100*s['execution_accuracy']:.2f} | "
                     f"{100*s['testsuite_accuracy']:.2f} | {s['decision_macro_f1']:.4f} | "
                     f"{100*s['unnecessary_question_rate_clear']:.2f} | {sum(r['correct'] for r in unans)} |")
    lines += ["", "## Primary comparison: training versus prompting", "",
              "Intervals resample entire databases, use 5,000 paired draws and preserve each seed separately.",
              "The two-seed average uses the same draws; it does not count the questions twice.", "",
              "| Metric | Comparison | Difference | 95% paired interval |", "|---|---|---:|---:|"]
    for metric, unit, factor in [("execution_accuracy", "percentage points", 100), ("decision_macro_f1", "F1", 1)]:
        for key, label in [("sft_s1", "Seed 1 minus B3"), ("sft_s2", "Seed 2 minus B3"),
                           ("two_seed_mean", "Two-seed mean minus B3")]:
            r = intervals[metric][key]
            lines.append(f"| {metric} | {label} | {factor*r['difference']:+.4f} {unit} | "
                         f"[{factor*r['ci95'][0]:+.4f}, {factor*r['ci95'][1]:+.4f}] |")
    lines += ["", "Both seeds improve execution and decision F1 on this test; the paired intervals above exclude zero.",
              "This supports the primary comparison within the experiment and its database-resampling assumptions.",
              "It does not establish general performance on unseen domains or validate the simulated user.", "",
              "## Important tradeoffs and exploratory comparisons", "",
              "- Correct abstention is lower for both SFT seeds than for B3. The system is better at answering but is less reliable at rejecting missing-information requests under the automatic metric.",
              "- A2 solves one more answerable case than SFT seed 1. Its paired execution and decision intervals include zero. Explicit slots have not demonstrated an execution benefit.",
              "- A3 does not improve the point estimates. Its execution interval touches zero; its decision interval includes zero. This is limited evidence about this full-context variant, not proof that retrieval is always necessary.",
              "- A single test case is about 0.68 percentage points on the 148 answerable episodes.",
              "- The oracle is a reference run, not a mathematical upper bound.", "",
              "## Where the execution gain comes from", "",
              "| Request type | Cases | B3 correct | SFT seed 1 correct | SFT seed 2 correct |", "|---|---:|---:|---:|---:|"]
    for kind in ("clear", "ambiguous", "two_slot", "definition"):
        groups = [[r for r in traces[n] if r["kind"] == kind] for n in ("b3", "sft_s1", "sft_s2")]
        lines.append(f"| {kind} | {len(groups[0])} | " + " | ".join(str(sum(r["correct"] for r in g)) for g in groups) + " |")
    lines += ["", "The gains include fewer unnecessary questions on clear requests and better definition handling.",
              "The partly answered cases remain difficult. Question usefulness still needs the blinded review.", "",
              "## Real dialogue slice (CoSQL)", "", "| System | Correct / 59 | Decision F1 | Clarify recall / 13 |",
              "|---|---:|---:|---:|"]
    for n in ("b3", "sft_s1"):
        s = json.loads((a.root / f"cosql_{n}/summary.json").read_text(encoding="utf-8"))
        rows = read_jsonl(a.root / f"cosql_{n}/traces.jsonl")
        clarify = [r for r in rows if r["gold_decision"] == "clarify"]
        lines.append(f"| {n} | {sum(r['correct'] for r in rows)} | {s['decision_macro_f1']:.4f} | {sum(r['correct'] for r in clarify)} |")
    majority = sum(r["gold_decision"] == "execute" for r in rows)
    majority_f1 = macro_f1([r["gold_decision"] for r in rows], ["execute"]*len(rows))[0]
    lines.append(f"| Always execute (label-only baseline) | {majority} | {majority_f1:.4f} | 0 |")
    lines += ["", "SFT improves overall decisions on the reviewed slice but detects fewer of its genuine clarification cases.",
              "Its accuracy is still below the always-execute baseline because the slice is dominated by execute labels; its macro-F1 is higher. The accuracy gain over B3 alone does not establish strong natural-dialogue generalization.",
              "The single abstention case cannot support a general abstention claim. This slice is exploratory.", "",
              "## Cost and output reliability", "", "| System | Total model calls | Total input tokens | Generation seconds |",
              "|---|---:|---:|---:|"]
    for n in ("b3", "sft_s1", "sft_s2", "a2", "a3"):
        s = summaries[n]
        lines.append(f"| {n} | {s['total_model_calls']} | {s['total_tokens_in']:,} | {s['total_generation_seconds']:.1f} |")
    lines += ["", "These totals include policy, writer and simulated-user inference. Generation seconds are elapsed generation time, not CUDA kernel profiling.",
              f"The two-GPU inference cell took {compute['wall_seconds']/60:.1f} minutes. CPU scoring time is not included in that cell measurement.",
              "Invalid outputs and limits should be reported per system; one successful retry does not erase the original invalid attempt.", ""]
    behavior = {}
    for n in ("b3", "sft_s1", "sft_s2", "a2", "a3"):
        failures = []
        counts = Counter()
        for r in traces[n]:
            if r["correct"]:
                continue
            if r["kind"] == "unanswerable":
                category = "missing_information_" + r["reason"]
            elif r["outcome"] == "abstained":
                category = "wrong_abstention_on_answerable"
            elif r["outcome"] != "executed":
                category = r["outcome"]
            elif r.get("sql_status") != "ok":
                category = "SQL_" + str(r.get("sql_status"))
            else:
                category = "executable_SQL_wrong_result"
            counts[category] += 1
            failures.append({"episode": r["episode"], "kind": r["kind"], "category": category,
                             "request": r["request"], "decision": r["decision"], "sql": r.get("sql"),
                             "abstain_text": r.get("abstain_text"), "answered_slots": r.get("answered_slots")})
        behavior[n] = {"counts": dict(counts), "cases": failures,
                       "note": "Observable failure types, not causal attribution of blame to a model role."}
    write_json(out / "error_analysis.json", behavior)
    lines += ["## Still required", "", "Complete the independent question-usefulness and simulator reviews, run the live demo, and write the final report with the tradeoffs above.",
              "Automatic missing-information and slot scores use lexical checks and have not been validated by those human reviews."]
    (out / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#adb5bd", "#8c99a8", "#e29b4c", "#217c83", "#4b9da0", "#8872aa", "#9f87b5", "#56616f"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), constrained_layout=True)
    for ax, metric, label, factor in zip(axes, ("execution_accuracy", "testsuite_accuracy", "decision_macro_f1"),
                                       ("Execution accuracy / 148", "Test-suite accuracy / 148", "Decision macro-F1 / 188"), (100, 100, 1)):
        values = [summaries[n][metric]*factor for n in NAMES]
        bars = ax.bar(LABELS, values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.1f}" if factor == 100 else f"{v:.3f}" for v in values], fontsize=7, padding=3)
        ax.set_ylim(0, factor*1.12)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Percent" if factor == 100 else "F1")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"test_results.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3), constrained_layout=True)
    for ax, metric, factor, label in zip(axes, ("execution_accuracy", "decision_macro_f1"), (100, 1),
                                       ("Execution difference (percentage points)", "Decision macro-F1 difference")):
        for y, (key, title) in enumerate([("sft_s1", "SFT seed 1 - B3"), ("sft_s2", "SFT seed 2 - B3"),
                                          ("two_seed_mean", "Two-seed mean - B3")]):
            r = intervals[metric][key]
            point, low, high = (factor*r['difference'], factor*r['ci95'][0], factor*r['ci95'][1])
            ax.errorbar(point, y, xerr=[[point-low], [high-point]], fmt="o", capsize=4, color="#217c83")
        ax.set_yticks(range(3), ["Seed 1", "Seed 2", "Two-seed mean"])
        ax.axvline(0, color="#888888", linestyle="--")
        ax.set_xlabel(label, fontsize=9)
        ax.set_title("95% paired database intervals", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"paired_intervals.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Compact, readable versions for the portrait learning PDF.
    selected = ("b3", "sft_s1", "sft_s2", "a2", "a3")
    selected_labels = ("B3", "SFT 1", "SFT 2", "A2", "A3")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3), constrained_layout=True)
    for ax, metric, title in zip(axes, ("execution_accuracy", "correct_abstention_rate"),
                                ("Correct execution / 148", "Correct abstention / 40")):
        values = [summaries[n][metric]*100 for n in selected]
        bars = ax.bar(selected_labels, values, color=["#e29b4c", "#217c83", "#4b9da0", "#8872aa", "#9f87b5"])
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=8, padding=3)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Percent")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"execution_abstention.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 2.5), constrained_layout=True)
    for y, key in enumerate(("sft_s1", "sft_s2", "two_seed_mean")):
        r = intervals["execution_accuracy"][key]
        point, low, high = (100*r['difference'], 100*r['ci95'][0], 100*r['ci95'][1])
        ax.errorbar(point, y, xerr=[[point-low], [high-point]], fmt="o", capsize=4, color="#217c83")
    ax.set_yticks(range(3), ["Seed 1 - B3", "Seed 2 - B3", "Two-seed mean - B3"])
    ax.axvline(0, color="#888888", linestyle="--")
    ax.set_xlabel("Execution accuracy difference (percentage points)")
    ax.set_title("95% paired intervals, resampling 20 databases", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"execution_intervals.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Wrote findings, descriptive error categories and standalone figures to", out)


if __name__ == "__main__":
    main()
