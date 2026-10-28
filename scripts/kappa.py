"""Agreement of the two blind annotators (Cohen's kappa) and with the intended first decision.

Usage:  python -I scripts/kappa.py validation        (or: test)
Reads annotation/<split>_label_A.csv, _label_B.csv and _label_key.csv; writes
annotation/<split>_disagreements.csv (to resolve together; it shows the glossary, both comments and what the user
really meant, since the labels are no longer blind at that point) and results/episodes/<split>_agreement.json.
"""

from __future__ import annotations

import argparse
import csv
import json

from clarifysql.config import results_dir
from clarifysql.episodes.review import ANNOTATION, DECISIONS, cell, read_sheet, resolved_decisions
from clarifysql.episodes.schema import FIRST_DECISION, load_episodes
from clarifysql.eval.agreement import cohen_kappa, confusion


def decision(row) -> str:
    return cell(row, "decision").lower()


def intended(episode_id: str) -> str:
    """The intended first decision, from the id ("test-ambiguous-906"), so dropped episodes still count."""
    return FIRST_DECISION[episode_id.split("-")[1]]


def user_meant(ep) -> str:
    """What the user really meant, in the simulated user's words."""
    if ep is None:
        return ""
    if ep.missing:
        return ep.missing["explanation"]
    return " / ".join(s["answer"] for s in ep.slots) or "(nothing hidden: the request is as meant)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("split", choices=["validation", "test"])
    args = ap.parse_args()
    key = {r["shown_id"]: r["episode_id"] for r in read_sheet(f"{args.split}_label_key.csv")}
    rows_a = {r["episode_id"]: r for r in read_sheet(f"{args.split}_label_A.csv")}
    rows_b = {r["episode_id"]: r for r in read_sheet(f"{args.split}_label_B.csv")}
    a = {s: decision(r) for s, r in rows_a.items()}
    b = {s: decision(r) for s, r in rows_b.items()}
    episodes = {ep.id: ep for ep in load_episodes(args.split)}
    ids = [s for s in a if a[s] in DECISIONS and b.get(s) in DECISIONS]
    if not ids:
        raise SystemExit("no row has a valid decision from both annotators yet")
    la, lb = [a[s] for s in ids], [b[s] for s in ids]
    gold = [intended(key[s]) for s in ids]
    report = {
        "labelled_by_both": len(ids), "unlabelled_or_invalid": len(a) - len(ids),
        "kappa_A_B": round(cohen_kappa(la, lb), 3),
        "agreement_A_B": round(sum(x == y for x, y in zip(la, lb)) / len(ids), 3),
        "kappa_A_intended": round(cohen_kappa(la, gold), 3),
        "kappa_B_intended": round(cohen_kappa(lb, gold), 3),
        "confusion_A_B": confusion(la, lb),
    }
    done = resolved_decisions(args.split)  # running this again keeps what the team has already resolved
    disagreements = [[s, key[s], a[s], b[s], g, rows_a[s]["request"], rows_a[s]["glossary"],
                      user_meant(episodes.get(key[s])), cell(rows_a[s], "comment"), cell(rows_b[s], "comment"),
                      done.get(key[s], "")]
                     for s, g in zip(ids, gold) if a[s] != b[s] or a[s] != g]
    with open(ANNOTATION / f"{args.split}_disagreements.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shown_id", "episode_id", "A", "B", "intended", "request", "glossary", "user_meant",
                    "comment_A", "comment_B", "resolved_decision"])
        w.writerows(disagreements)
    out = results_dir() / "episodes" / f"{args.split}_agreement.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_A_B"}, indent=2))
    print(f"{len(disagreements)} rows to resolve in {ANNOTATION / f'{args.split}_disagreements.csv'}")


if __name__ == "__main__":
    main()
