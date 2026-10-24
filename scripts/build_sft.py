"""Build the fine-tuning data: conversation states -> next action (proposal, Section 9).

Usage:  python -I scripts/build_sft.py
Writes data/sft/{train,validation}.jsonl (slot version, our main adapter) and
data/sft/{train,validation}_noslot.jsonl (no slots or meanings: ablation A2), plus results/sft/sft_stats.json
with the action mix and the token lengths (prompt + target, Qwen3 chat format; no PyTorch needed).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter

from clarifysql.config import REPO_ROOT, results_dir
from clarifysql.episodes.schema import load_episodes
from clarifysql.episodes.states import episode_states
from clarifysql.llm.tokens import count_chat_tokens

MAX_TOKENS = 1024


def main() -> None:
    out_dir = REPO_ROOT / "data" / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"max_tokens": MAX_TOKENS}
    for split in ("train", "validation"):
        # an episode whose request the team still has to write would show the original question
        episodes = [ep for ep in load_episodes(split) if not ep.draft.get("needs_writing")]
        for with_slots, suffix in ((True, ""), (False, "_noslot")):
            rows = [s for ep in episodes for s in episode_states(ep, with_slots)]
            lengths = []
            for r in rows:
                r["tokens"] = count_chat_tokens(r["messages"] + [{"role": "assistant", "content": r["target"]}])
                lengths.append(r["tokens"])
            kept = [r for r in rows if r["tokens"] <= MAX_TOKENS]  # longer states are dropped, not truncated
            with open(out_dir / f"{split}{suffix}.jsonl", "w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            v = sorted(lengths)
            stats[f"{split}{suffix}"] = {
                "states": len(kept), "dropped_over_limit": len(rows) - len(kept), "episodes": len(episodes),
                "target_actions": dict(Counter(r["target_action"] for r in kept)),
                "states_by_kind": dict(Counter(r["kind"] for r in kept)),
                "tokens": {"median": statistics.median(v), "p90": v[int(0.9 * (len(v) - 1))], "max": v[-1],
                           "over_limit": sum(x > MAX_TOKENS for x in v)},
            }
            print(f"{split}{suffix:<8} {stats[f'{split}{suffix}']}")
    path = results_dir() / "sft" / "sft_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"written to {out_dir} and {path}")


if __name__ == "__main__":
    main()
