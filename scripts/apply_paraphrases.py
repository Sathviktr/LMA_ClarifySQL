"""Put the accepted paraphrases (from GPU checkpoint 3) into data/episodes/train.jsonl.

Usage:  python -I scripts/apply_paraphrases.py [results/episodes/paraphrases_train.jsonl]
Afterwards rebuild the fine-tuning data:  python -I scripts/build_sft.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clarifysql.config import results_dir
from clarifysql.episodes.paraphrase import accept
from clarifysql.episodes.schema import EPISODE_DIR, load_episodes, write_episodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", nargs="?", type=Path, default=results_dir() / "episodes" / "paraphrases_train.jsonl")
    args = ap.parse_args()
    rows = {r["episode"]: r for r in map(json.loads, args.file.read_text(encoding="utf-8").splitlines()) if r}
    episodes = load_episodes("train")
    applied = 0
    for ep in episodes:
        r = rows.get(ep.id)
        if not r or not r["accepted"]:
            continue
        ok, _ = accept(ep, r["paraphrase"])  # re-check here, in case the episodes changed since the GPU run
        if ok and r["paraphrase"] != ep.request:
            ep.draft.update(method="model_paraphrase", rule_draft=ep.request)
            ep.request = r["paraphrase"]
            applied += 1
    write_episodes(episodes, EPISODE_DIR / "train.jsonl")
    print(f"{applied} training requests replaced by accepted paraphrases (of {len(rows)} paraphrased)")


if __name__ == "__main__":
    main()
