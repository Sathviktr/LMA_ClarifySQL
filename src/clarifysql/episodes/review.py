"""The team's review sheets in annotation/ (proposal, Section 8.3): reading them and what they drop.

Only signed rows count, so a pre-filled sheet that nobody has touched changes nothing. An episode is dropped
when the editor writes DROP, when the two annotators' resolved decision differs from the episode's intended
first decision (it is not a clean example of its kind), or when the training audit marks it `drop`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from clarifysql.config import REPO_ROOT

ANNOTATION = REPO_ROOT / "annotation"
DECISIONS = ("execute", "clarify", "abstain")


def read_sheet(name: str, directory: Path = ANNOTATION) -> list[dict]:
    """Rows of a CSV sheet; [] if it does not exist. Excel's plain "CSV" (not "CSV UTF-8") is cp1252."""
    path = directory / name
    if not path.exists():
        return []
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: save it again as 'CSV UTF-8'")


def cell(row: dict, prefix: str) -> str:
    """The value of the column whose header starts with `prefix` (headers carry hints like '(keep / drop)')."""
    return next(((v or "").strip() for k, v in row.items() if k and k.startswith(prefix)), "")


def edit_drops(split: str, directory: Path = ANNOTATION) -> set[str]:
    return {r["episode_id"] for r in read_sheet(f"{split}_edit.csv", directory)
            if cell(r, "final_request").upper() == "DROP" and cell(r, "editor")}


def resolved_decisions(split: str, directory: Path = ANNOTATION) -> dict[str, str]:
    return {r["episode_id"]: cell(r, "resolved_decision").lower()
            for r in read_sheet(f"{split}_disagreements.csv", directory) if cell(r, "resolved_decision")}


def label_drops(split: str, directory: Path = ANNOTATION) -> set[str]:
    intended = {r["episode_id"]: cell(r, "intended") for r in read_sheet(f"{split}_disagreements.csv", directory)}
    return {eid for eid, d in resolved_decisions(split, directory).items() if d != intended.get(eid)}


def audit_drops(directory: Path = ANNOTATION) -> set[str]:
    return {r["episode_id"] for r in read_sheet("train_audit.csv", directory)
            if cell(r, "verdict").lower() == "drop" and cell(r, "auditor")}


def dropped(split: str, directory: Path = ANNOTATION) -> set[str]:
    """Every episode of a split that the team's review removes."""
    if split == "train":
        return audit_drops(directory)
    return edit_drops(split, directory) | label_drops(split, directory)
