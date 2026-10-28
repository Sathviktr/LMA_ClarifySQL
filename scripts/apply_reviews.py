"""Apply the team's reviews to the episodes.

Usage:  python -I scripts/apply_reviews.py validation        (or: test, train)
validation / test: reads annotation/<split>_edit.csv: rows signed in the `editor` column are applied (a changed
`final_request` replaces the request; DROP removes the episode). If annotation/<split>_disagreements.csv has
`resolved_decision` values, they are stored with the episode, and an episode whose resolved decision differs from
its intended first decision is dropped.
train: reads annotation/train_audit.csv: signed `drop` verdicts remove the episode; the keep rate per stratum goes to
results/episodes/train_audit.json. Run scripts/build_sft.py afterwards.
The automatic checks run again on what is left; episodes that still fail are listed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from clarifysql.config import results_dir
from clarifysql.episodes.checks import check_episode
from clarifysql.episodes.review import cell, dropped, read_sheet, resolved_decisions
from clarifysql.episodes.schema import EPISODE_DIR, load_episodes, write_episodes


def apply_edits(split: str, by_id: dict) -> None:
    edited = 0
    for row in read_sheet(f"{split}_edit.csv"):
        ep = by_id.get(row["episode_id"])
        final, editor = cell(row, "final_request"), cell(row, "editor")
        if ep is None or not final or not editor or final.upper() == "DROP":
            continue  # a row counts as reviewed only when the editor has signed it
        if final != ep.request or ep.draft.get("needs_writing"):
            ep.request = final
            ep.draft.update(method="human", needs_writing=False)
            edited += 1
        ep.draft.update(needs_review=False, editor=editor)
    for eid, decision in resolved_decisions(split).items():
        if eid in by_id:
            by_id[eid].draft["resolved_decision"] = decision
    print(f"{edited} requests updated")


def apply_audit(by_id: dict) -> None:
    rows = [r for r in read_sheet("train_audit.csv") if cell(r, "auditor") and cell(r, "verdict").lower() in
            ("keep", "drop")]
    for r in rows:
        if r["episode_id"] in by_id:
            by_id[r["episode_id"]].draft["audit"] = cell(r, "verdict").lower()
    # counted from the sheet, so running this twice gives the same report
    audited = Counter(f"{r['kind']}/{r['made_by']}" for r in rows)
    drops = Counter(f"{r['kind']}/{r['made_by']}" for r in rows if cell(r, "verdict").lower() == "drop")
    report = {"audited": len(rows), "dropped": sum(drops.values()),
              "by_stratum": {k: {"audited": n, "dropped": drops[k], "kept_rate": round(1 - drops[k] / n, 3)}
                             for k, n in sorted(audited.items())}}
    out = results_dir() / "episodes" / "train_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} training episodes audited, {report['dropped']} marked drop -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["validation", "test", "train"])
    args = ap.parse_args()
    episodes = load_episodes(args.split)
    by_id = {ep.id: ep for ep in episodes}
    if args.split == "train":
        apply_audit(by_id)
    else:
        apply_edits(args.split, by_id)
    drop = dropped(args.split)
    kept = [ep for ep in episodes if ep.id not in drop]
    write_episodes(kept, EPISODE_DIR / f"{args.split}.jsonl")
    failing = {ep.id: p for ep in kept if (p := check_episode(ep))}
    print(f"{len(episodes) - len(kept)} episodes dropped now ({len(drop)} by the review in total), {len(kept)} left; "
          f"{len(failing)} episodes still fail the checks")
    for eid, problems in failing.items():
        print(f"  {eid}: {'; '.join(problems)}")


if __name__ == "__main__":
    main()
