"""Apply the team's edits to the validation / test episodes.

Usage:  python -I scripts/apply_reviews.py validation        (or: test)
Reads annotation/<split>_edit.csv: rows signed in the `editor` column are applied (a changed `final_request`
replaces the request; DROP removes the episode), and the automatic checks run again. Episodes that still fail are listed.
If annotation/<split>_disagreements.csv has `resolved_decision` values, they are stored with the episode.
"""

from __future__ import annotations

import argparse
import csv

from clarifysql.config import REPO_ROOT
from clarifysql.episodes.checks import check_episode
from clarifysql.episodes.schema import EPISODE_DIR, load_episodes, write_episodes

ANNOTATION = REPO_ROOT / "annotation"


def read(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("split", choices=["validation", "test"])
    args = ap.parse_args()
    episodes = load_episodes(args.split)
    by_id = {ep.id: ep for ep in episodes}
    edited, dropped = 0, []
    for row in read(ANNOTATION / f"{args.split}_edit.csv"):
        ep = by_id.get(row["episode_id"])
        final = (row.get("final_request") or "").strip()
        editor = (row.get("editor") or "").strip()
        if ep is None or not final or not editor:
            continue  # a row counts as reviewed only when the editor has signed it
        if final.upper() == "DROP":  # the editor found no sensible vague version
            dropped.append(ep.id)
            continue
        if final != ep.request or ep.draft.get("needs_writing"):
            ep.request = final
            ep.draft.update(method="human", needs_writing=False)
            edited += 1
        ep.draft.update(needs_review=False, editor=editor)
    for row in read(ANNOTATION / f"{args.split}_disagreements.csv"):
        ep = by_id.get(row["episode_id"])
        if ep and row.get("resolved_decision", "").strip():
            ep.draft["resolved_decision"] = row["resolved_decision"].strip().lower()
    episodes = [ep for ep in episodes if ep.id not in set(dropped)]
    write_episodes(episodes, EPISODE_DIR / f"{args.split}.jsonl")
    failing = {ep.id: p for ep in episodes if (p := check_episode(ep))}
    print(f"{edited} requests updated, {len(dropped)} episodes dropped; {len(failing)} episodes still fail the checks")
    for eid, problems in failing.items():
        print(f"  {eid}: {'; '.join(problems)}")


if __name__ == "__main__":
    main()
