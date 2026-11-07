"""Create blank evaluation review sheets after real test runs; never fill human judgments."""

import argparse
import csv
import json
import random
from pathlib import Path
from clarifysql.episodes.schema import load_episodes
from clarifysql.eval.artifacts import read_jsonl


def write_sheet(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/final"))
    ap.add_argument("--out", type=Path, default=Path("annotation/evaluation"))
    a = ap.parse_args()
    names = ("b3", "sft_s1", "a2")
    simulator_names = ("b3", "sft_s1", "sft_s2", "a2", "a3")
    episodes = {ep.id: ep for ep in load_episodes("test")}
    a.out.mkdir(parents=True, exist_ok=True)
    if any(p.name != "README.md" or not p.is_file() for p in a.out.iterdir()):
        raise ValueError("Output folder is not empty; refusing to overwrite or partially replace reviews")
    traces = {}
    for name in simulator_names:
        folder = a.root / f"test_{name}"
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if summary["backend"] != "hf" or not summary["complete"]:
            raise ValueError("Human audits require complete real-model traces")
        traces[name] = {r["episode"]: r for r in read_jsonl(folder / "traces.jsonl")}
        if traces[name].keys() != episodes.keys():
            raise ValueError("Trace episodes do not match the complete test set")
    rng = random.Random(2026)
    chosen = rng.sample(sorted(id_ for id_, ep in episodes.items() if ep.kind in ("ambiguous", "two_slot")), 40)
    entries = [(name, id_) for id_ in chosen for name in names]
    rng.shuffle(entries)
    blind, keys = [], []
    for i, (name, id_) in enumerate(entries, 1):
        ep, t = episodes[id_], traces[name][id_]
        dialogue = "\n".join(f"Question: {s['action']}\nReply: {s['result']}" for s in t["steps"] if s["kind"] == "clarify")
        from clarifysql.data import spider
        from clarifysql.db.catalog import compact_catalog
        blind.append({"item": f"Q{i:03d}", "request": ep.request, "intended_request": ep.source_question,
                      "tables": compact_catalog(spider.db_path(ep.db_id)),
                      "dialogue": dialogue or "No clarification question was asked.",
                      "targets_uncertainty": "", "schema_supported": "", "not_repeated": "",
                      "resolves_intent": "", "annotator": "", "notes": ""})
        keys.append({"item": f"Q{i:03d}", "system": name, "episode": id_})
    for who in ("A", "B"):
        write_sheet(a.out / f"question_usefulness_{who}.csv", blind, list(blind[0]))
    write_sheet(a.out / "question_key.csv", keys, list(keys[0]))
    coverage = {}
    for name in simulator_names:
        candidates = []
        for id_, t in traces[name].items():
            ep = episodes[id_]
            if ep.kind not in ("ambiguous", "two_slot"):
                continue
            for e in t["events"]:
                if e["role"] == "user":
                    candidates.append({"episode": id_, "request": ep.request, "question": e["question"],
                        "hidden_details": json.dumps(ep.slots, ensure_ascii=False), "simulator_choice": e.get("choice"),
                        "simulator_reply": e["reply"], "human_slot_indices": "", "correct": "",
                        "annotator": "", "notes": ""})
        picked = rng.sample(candidates, min(50, len(candidates)))
        coverage[name] = {"available": len(candidates), "sampled": len(picked), "target": 50}
        fields = list(picked[0]) if picked else ["episode", "correct", "annotator", "notes"]
        write_sheet(a.out / f"simulator_{name}.csv", picked, fields)
    (a.out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print("Created blank review sheets in", a.out, "; do not show question_key.csv to the raters.")
    print("Simulator coverage (shortfalls are reported, not filled by repeated examples):", coverage)


if __name__ == "__main__":
    main()
