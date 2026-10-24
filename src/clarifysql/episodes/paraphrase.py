"""Polish training requests with the base model (proposal, Section 8.2: "The local Qwen3-4B paraphrases vague
questions for training data only").

The model rewrites from the *original* question plus a plain description of the detail to hide (the rule
draft is only a hint), so details lost by a clumsy rule are restored. Every rewrite must pass the same
automatic checks as the rule draft; otherwise the draft is kept. Definition episodes are not paraphrased,
because their vague phrase must stay identical to the glossary term that retrieval has to find.
"""

from __future__ import annotations

import copy
import re

from clarifysql.episodes.checks import check_episode
from clarifysql.episodes.schema import Episode

PARAPHRASE_KINDS = ("ambiguous", "two_slot", "unanswerable")

SYSTEM = ("You rewrite questions that people type to a university database assistant. "
          "Follow the instructions exactly and answer with the rewritten question only, on one line.")


def _hide_instruction(slot: dict) -> str:
    if slot["kind"] == "threshold":
        return (f'- do not state the exact cut-off "{slot["hidden"]}"; use a vague word such as "{slot["vague"]}" '
                f"instead, so that the reader cannot know the value")
    return (f'- do not say what is used for ranking ("{slot["value"]}"); say something vague such as '
            f'"{slot["vague"]}" or "the top one" instead')


def paraphrase_messages(ep: Episode) -> list[dict]:
    if ep.kind == "unanswerable":
        user = (f"Question: {ep.request}\n"
                "Rewrite it so that it sounds natural, as a staff member might type it. Keep its meaning, keep the "
                f'words "{ep.missing["attribute"]}", and do not add new details.')
    else:
        rules = "\n".join(_hide_instruction(s) for s in ep.slots)
        user = (f"Original question: {ep.source_question}\n"
                f"Rewrite it so that it is vague in this way:\n{rules}\n"
                "- keep every other detail (what to list, other conditions, the order) and sound natural\n"
                f"A clumsy first attempt, for reference: {ep.request}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def clean(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    first = re.sub(r"^(rewritten question|question)\s*:\s*", "", lines[0], flags=re.I)
    return first.strip().strip('"').strip()


def accept(ep: Episode, text: str) -> tuple[bool, list[str]]:
    """Would this rewrite pass every check? Returns (accepted, reasons for rejection)."""
    text = clean(text)
    reasons = []
    if not text:
        return False, ["empty"]
    base = ep.source_question or ep.request
    if base.rstrip().endswith(("?", ".")) and not text.endswith(("?", ".")):
        reasons.append("does not end like a question or request")
    if not 0.5 <= len(text) / max(len(base), 1) <= 1.6:
        reasons.append("length very different from the original")
    if ep.kind == "unanswerable" and ep.missing["attribute"].lower() not in text.lower():
        reasons.append("lost the missing attribute")
    trial = copy.deepcopy(ep)
    trial.request = text
    reasons += check_episode(trial)
    return not reasons, reasons
