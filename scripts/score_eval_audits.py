"""Score completed independent review sheets, refusing blanks and unsigned judgments."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from clarifysql.eval.artifacts import write_json
from clarifysql.eval.agreement import cohen_kappa

CRITERIA = ("targets_uncertainty", "schema_supported", "not_repeated", "resolves_intent")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def binary(value):
    if value.strip().lower() in ("1", "yes", "true"):
        return 1
    if value.strip().lower() in ("0", "no", "false"):
        return 0
    raise ValueError("Each judgment must be yes/no or 1/0; blanks are not judgments")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheets", type=Path, default=Path("annotation/evaluation"))
    ap.add_argument("--out", type=Path, default=Path("results/final/analysis/human_audits.json"))
    a = ap.parse_args()
    key = {r["item"]: r for r in read(a.sheets / "question_key.csv")}
    scores = {}
    grouped = defaultdict(lambda: defaultdict(list))
    for who in ("A", "B"):
        rows = read(a.sheets / f"question_usefulness_{who}.csv")
        if len(rows) != len(key) or {r["item"] for r in rows} != key.keys():
            raise ValueError("Each rater must review every blinded item exactly once")
        scores[who] = {}
        for row in rows:
            if not row["annotator"].strip():
                raise ValueError(f"{who}/{row['item']}: unsigned judgment")
            values = {c: binary(row[c]) for c in CRITERIA}
            scores[who][row["item"]] = values
            group = grouped[key[row["item"]]["system"]]
            for c, value in values.items():
                group[c].append(value)
            group["all_four"].append(int(all(values.values())))
    usefulness = {system: {c: {"yes": sum(v), "ratings": len(v), "rate": sum(v)/len(v)}
                            for c, v in criteria.items()} for system, criteria in grouped.items()}
    agreement = {c: cohen_kappa([str(scores['A'][id_][c]) for id_ in key],
                              [str(scores['B'][id_][c]) for id_ in key]) for c in CRITERIA}
    simulator = {}
    for path in sorted(a.sheets.glob("simulator_*.csv")):
        rows = read(path)
        if any(not r["annotator"].strip() for r in rows):
            raise ValueError(f"{path}: unsigned simulator judgments")
        values = [binary(r["correct"]) for r in rows]
        simulator[path.stem] = {"correct": sum(values), "reviewed": len(values),
                                "accuracy": sum(values)/len(values) if values else None}
    write_json(a.out, {"question_usefulness": usefulness, "rater_kappa": agreement,
                       "simulator_accuracy": simulator,
                       "note": "Question rates pool two independent ratings; preserve per-rater sheets and disagreements."})
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
