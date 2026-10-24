"""Build data/episodes/{train,validation,test}.jsonl and results/episodes/build_stats.json.

Usage:  python -I scripts/build_episodes.py [--split train]
Validation and test episodes are drafts until the team has reviewed them (see annotation/).
"""

from __future__ import annotations

import argparse
import json

from clarifysql.config import results_dir
from clarifysql.episodes.build import build_split
from clarifysql.episodes.checks import check_episodes
from clarifysql.episodes.schema import EPISODE_DIR, write_episodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "validation", "test"], nargs="+",
                    default=["train", "validation", "test"])
    args = ap.parse_args()
    stats_path = results_dir() / "episodes" / "build_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    for split in args.split:
        episodes, st = build_split(split)
        report = check_episodes(episodes)
        st["checks"] = report["summary"]
        write_episodes(episodes, EPISODE_DIR / f"{split}.jsonl")
        stats[split] = st
        print(f"{split:<10} built {st['built']}  (targets {st['targets']})")
        print(f"{'':<10} checks: {report['summary']}")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"written to {EPISODE_DIR} and {stats_path}")


if __name__ == "__main__":
    main()
