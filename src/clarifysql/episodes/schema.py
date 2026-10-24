"""The episode record (proposal, Section 8.2).

An episode is one database, one user request, a hidden intent (the gold SQL and the slots the user has in
mind), the correct first decision, and a reference path of actions. Episode kinds:

  clear         an original Spider question                          -> execute
  ambiguous     one detail removed (threshold or ranking)            -> inspect, clarify, execute
  two_slot      two details removed; the user answers one at a time   -> inspect, clarify, clarify, execute
  definition    one detail removed, but a glossary entry defines it  -> inspect, execute (no question)
  unanswerable  the database does not store what is asked            -> inspect, abstain

Fields starting with `hidden_` and the `slots`, `gold_sql`, `reference` and `missing` fields must never be
shown to the policy; tests/test_episodes.py checks that the policy's input cannot contain them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from clarifysql.config import REPO_ROOT

EPISODE_DIR = REPO_ROOT / "data" / "episodes"
FIRST_DECISION = {"clear": "execute", "ambiguous": "clarify", "two_slot": "clarify", "definition": "execute",
                  "unanswerable": "abstain"}


@dataclass
class Episode:
    id: str
    split: str
    db_id: str
    kind: str
    request: str                      # what the user types (shown to the policy)
    glossary: list[dict] = field(default_factory=list)  # shown only through search_definitions
    gold_sql: str | None = None       # hidden intent (None for unanswerable)
    source_question: str | None = None  # the original, fully specified question (oracle input)
    source_id: str | None = None
    slots: list[dict] = field(default_factory=list)    # hidden details, with the simulated user's answers
    missing: dict | None = None       # unanswerable: what the database does not store
    reference: list[dict] = field(default_factory=list)  # the reference path of actions
    draft: dict = field(default_factory=dict)  # how the request was made and whether it needs review

    @property
    def first_decision(self) -> str:
        return FIRST_DECISION[self.kind]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["first_decision"] = self.first_decision
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        d = {k: v for k, v in d.items() if k != "first_decision"}
        return cls(**d)


def write_episodes(episodes: list[Episode], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_episodes(split: str, directory: Path = EPISODE_DIR) -> list[Episode]:
    path = directory / f"{split}.jsonl"
    return [Episode.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
