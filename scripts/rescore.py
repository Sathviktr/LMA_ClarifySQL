"""Score saved agent traces again, against the current episodes and the current metrics.

Usage:  python -I scripts/rescore.py validation        (every results/agent/validation_*.jsonl trace file)
Rewrites each <split>_<name>_summary.json and keeps its run information (system, adapter, wall time). Needed when
episodes are removed after a run (music_2 after checkpoint 4) or a metric is added (slot accuracy, checkpoint 5).
The traces themselves are never changed.
"""

from __future__ import annotations

import argparse
import json

from clarifysql.config import results_dir
from clarifysql.episodes.schema import load_episodes
from clarifysql.eval.metrics import score_trace, summarize

RUN_KEYS = ("system", "split", "adapter", "backend", "wall_seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["validation", "test"])
    args = ap.parse_args()
    episodes = {ep.id: ep for ep in load_episodes(args.split)}
    for path in sorted((results_dir() / "agent").glob(f"{args.split}_*.jsonl")):
        if path.stem.endswith("_user_log"):
            continue
        summary_path = path.with_name(f"{path.stem}_summary.json")
        old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        traces = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        scored = [score_trace(t, episodes[t["episode"]]) for t in traces if t["episode"] in episodes]
        name = path.stem[len(args.split) + 1:]
        summary = {**{k: old[k] for k in RUN_KEYS if k in old}, "name": name, **summarize(scored)}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"{name:16} {len(scored)} of {len(traces)} episodes | execution {summary['execution_accuracy']} | "
              f"decision F1 {summary['decision_macro_f1']} | slots {summary['slot_accuracy']}")


if __name__ == "__main__":
    main()
