"""Paraphrase the training requests with the base model (GPU checkpoint 3) and check every rewrite.

Usage (GPU):     python scripts/paraphrase_train.py
Laptop check:    python -I scripts/paraphrase_train.py --backend mock     (the "model" returns the rule draft)
Writes results/episodes/paraphrases_train.jsonl: one line per episode with the draft, the rewrite, whether
it was accepted and why not. scripts/apply_paraphrases.py then puts accepted rewrites into the episodes.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from clarifysql.config import results_dir
from clarifysql.episodes.paraphrase import PARAPHRASE_KINDS, accept, clean, paraphrase_messages
from clarifysql.episodes.schema import load_episodes
from clarifysql.llm.backend import MockBackend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    episodes = [ep for ep in load_episodes("train") if ep.kind in PARAPHRASE_KINDS]
    if args.backend == "hf":
        from clarifysql.llm.backend import HFBackend

        backend = HFBackend()
    else:
        drafts = {paraphrase_messages(ep)[1]["content"]: ep.request for ep in episodes}
        backend = MockBackend(lambda chat, adapter: drafts[chat[1]["content"]])
    t0 = time.perf_counter()
    gens = backend.generate([paraphrase_messages(ep) for ep in episodes], max_new_tokens=96, adapter=False,
                            batch_size=args.batch_size)
    rows = []
    for ep, g in zip(episodes, gens):
        ok, reasons = accept(ep, g.text)
        rows.append({"episode": ep.id, "kind": ep.kind, "draft": ep.request, "paraphrase": clean(g.text),
                     "accepted": ok, "reasons": reasons})
    out = results_dir() / "episodes" / f"paraphrases_train{'' if args.backend == 'hf' else '_mock'}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    acc = Counter((r["kind"], r["accepted"]) for r in rows)
    print(json.dumps({"episodes": len(rows), "accepted": sum(r["accepted"] for r in rows),
                      "by_kind": {f"{k}/{'ok' if a else 'rejected'}": n for (k, a), n in sorted(acc.items())},
                      "rejection_reasons": Counter(x for r in rows for x in r["reasons"]).most_common(6),
                      "seconds": round(time.perf_counter() - t0, 1)}, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
