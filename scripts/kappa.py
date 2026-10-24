"""Agreement of the two blind annotators (Cohen's kappa) and with the intended first decision.

Usage:  python -I scripts/kappa.py validation        (or: test)
Reads annotation/<split>_label_A.csv, _label_B.csv and _label_key.csv; writes
annotation/<split>_disagreements.csv (to resolve together) and results/episodes/<split>_agreement.json.
"""

from __future__ import annotations

import argparse
import csv
import json

from clarifysql.config import REPO_ROOT, results_dir
from clarifysql.episodes.schema import load_episodes
from clarifysql.eval.agreement import cohen_kappa, confusion

ANNOTATION = REPO_ROOT / "annotation"
DECISIONS = {"execute", "clarify", "abstain"}


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def decision(row) -> str:
    key = next(k for k in row if k.startswith("decision"))
    return row[key].strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("split", choices=["validation", "test"])
    args = ap.parse_args()
    key = {r["shown_id"]: r["episode_id"] for r in read(ANNOTATION / f"{args.split}_label_key.csv")}
    a = {r["episode_id"]: decision(r) for r in read(ANNOTATION / f"{args.split}_label_A.csv")}
    b = {r["episode_id"]: decision(r) for r in read(ANNOTATION / f"{args.split}_label_B.csv")}
    episodes = {ep.id: ep for ep in load_episodes(args.split)}
    ids = [s for s in a if a[s] in DECISIONS and b.get(s) in DECISIONS]
    missing = len(a) - len(ids)
    la, lb = [a[s] for s in ids], [b[s] for s in ids]
    intended = [episodes[key[s]].first_decision for s in ids]
    report = {
        "labelled_by_both": len(ids), "unlabelled_or_invalid": missing,
        "kappa_A_B": round(cohen_kappa(la, lb), 3),
        "agreement_A_B": round(sum(x == y for x, y in zip(la, lb)) / len(ids), 3),
        "kappa_A_intended": round(cohen_kappa(la, intended), 3),
        "kappa_B_intended": round(cohen_kappa(lb, intended), 3),
        "confusion_A_B": confusion(la, lb),
    }
    disagreements = [[s, key[s], a[s], b[s], episodes[key[s]].first_decision, episodes[key[s]].request, ""]
                     for s in ids if a[s] != b[s] or a[s] != episodes[key[s]].first_decision]
    with open(ANNOTATION / f"{args.split}_disagreements.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shown_id", "episode_id", "A", "B", "intended", "request", "resolved_decision"])
        w.writerows(disagreements)
    out = results_dir() / "episodes" / f"{args.split}_agreement.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_A_B"}, indent=2))
    print(f"{len(disagreements)} rows to resolve in {ANNOTATION / f'{args.split}_disagreements.csv'}")


if __name__ == "__main__":
    main()
