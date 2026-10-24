"""Create the team's review sheets in annotation/ (CSV files that open in Excel or Google Sheets).

Usage:
    python -I scripts/make_review_sheets.py edit     # 1. write / edit validation and test requests
    python -I scripts/make_review_sheets.py label    # 2. blind labels by annotators A and B (after step 1)
    python -I scripts/make_review_sheets.py cosql    # the CoSQL dev slice to clean by hand
See annotation/README.md for the instructions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

from clarifysql.config import REPO_ROOT
from clarifysql.data import cosql, spider
from clarifysql.data.splits import load_splits
from clarifysql.db.catalog import compact_catalog
from clarifysql.episodes.checks import check_episode
from clarifysql.episodes.schema import load_episodes

ANNOTATION = REPO_ROOT / "annotation"
COSQL_MAP = {"AMBIGUOUS": "clarify", "CANNOT_ANSWER": "abstain", "INFORM_SQL": "execute"}


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel then reads UTF-8 correctly
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def shown_id(episode_id: str) -> str:
    """A stable id that does not reveal the episode kind (labels must be blind)."""
    split = episode_id.split("-")[0]
    return f"{split}-{hashlib.sha1(episode_id.encode()).hexdigest()[:8]}"


def hidden_detail(ep) -> str:
    if ep.kind == "unanswerable":
        return ep.missing["explanation"]
    return " | ".join(f"{s['kind']}: {s['hidden'] or s['value']} (user means: {s['value']})" for s in ep.slots)


def edit_sheets() -> None:
    header = ["episode_id", "kind", "db_id", "original_question", "hidden_detail", "draft_request",
              "automatic_check_problems", "final_request", "editor", "notes"]
    for split in ("validation", "test"):
        rows = []
        for ep in load_episodes(split):
            if ep.kind == "clear":
                continue  # original Spider questions: nothing to write
            problems = "; ".join(check_episode(ep))
            draft = "" if ep.draft.get("needs_writing") else ep.request
            rows.append([ep.id, ep.kind, ep.db_id, ep.source_question or "", hidden_detail(ep), draft, problems,
                         draft, "", ""])
        print(write_csv(ANNOTATION / f"{split}_edit.csv", header, rows), f"({len(rows)} rows)")


def label_sheets() -> None:
    header = ["episode_id", "db_id", "tables", "glossary", "request", "decision (execute / clarify / abstain)",
              "comment"]
    for split in ("validation", "test"):
        episodes = load_episodes(split)
        rng = random.Random(f"label-{split}")
        rng.shuffle(episodes)  # mixed order, so the kinds are not grouped together
        rows = []
        for ep in episodes:
            glossary = "\n".join(f"{g['term']}: {g['definition']}" for g in ep.glossary)
            rows.append([shown_id(ep.id), ep.db_id,
                         compact_catalog(spider.db_path(ep.db_id)), glossary, ep.request, "", ""])
        # the shown id hides the episode kind; keep the mapping for scripts/kappa.py
        mapping = [[r[0], ep.id] for r, ep in zip(rows, episodes)]
        write_csv(ANNOTATION / f"{split}_label_key.csv", ["shown_id", "episode_id"], mapping)
        for who in ("A", "B"):
            print(write_csv(ANNOTATION / f"{split}_label_{who}.csv", header, rows), f"({len(rows)} rows)")


def cosql_sheet(n_ambiguous: int = 25, n_cannot: int = 10, n_clear: int = 25, seed: int = 2026) -> None:
    test_dbs = set(load_splits()["test"])
    found = {"AMBIGUOUS": [], "CANNOT_ANSWER": [], "INFORM_SQL": []}
    for dialog_id, d in cosql.load_dialogues().items():
        if d.get("db_id") not in test_dbs:
            continue
        context = []
        for t in d.get("turns", []):
            text = (t.get("text") or "").strip()
            if t.get("isUser"):
                labels = t.get("label") or []
                for lab in found:
                    if lab in labels and len(labels) == 1:
                        found[lab].append((dialog_id, d["db_id"], t.get("turn_index"), list(context[-4:]), text, lab))
                context.append(f"user: {text}")
            elif not t.get("isSql") and text and not text.startswith("v2/"):
                context.append(f"system: {text}")
    rng = random.Random(seed)
    picked = []
    for lab, n in (("AMBIGUOUS", n_ambiguous), ("CANNOT_ANSWER", n_cannot), ("INFORM_SQL", n_clear)):
        pool = found[lab]
        picked += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(picked)
    header = ["turn_id", "db_id", "earlier_turns", "user_turn", "cosql_label", "mapped_decision",
              "cleaned_decision (execute / clarify / abstain / drop)", "annotator", "notes"]
    rows = [[f"{dlg}#{idx}", db, "\n".join(ctx), text, lab, COSQL_MAP[lab], "", "", ""]
            for dlg, db, idx, ctx, text, lab in picked]
    print(write_csv(ANNOTATION / "cosql_dev_slice.csv", header, rows), f"({len(rows)} rows)",
          {k: len(v) for k, v in found.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", choices=["edit", "label", "cosql"])
    args = ap.parse_args()
    {"edit": edit_sheets, "label": label_sheets, "cosql": cosql_sheet}[args.sheet]()


if __name__ == "__main__":
    main()
