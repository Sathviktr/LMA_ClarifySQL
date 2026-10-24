"""A two-stage simulated user (proposal, Section 10, following BIRD-Interact).

Stage 1: the base model (adapter off) reads ONLY the agent's natural-language question and picks, from a fixed
         multiple-choice list, which hidden detail of the episode it asks about, or "none of these".
Stage 2: a template answers with only that detail (the slot's prepared answer, e.g. "I mean more than 4.").
         Questions about nothing hidden get "I'm not sure what you mean."

The user never sees the agent's slot fields, so systems with and without slots are judged the same way
(important for RQ2). Every decision is recorded so that a person can audit 50 of them per system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NOT_SURE = "I'm not sure what you mean."
NOTHING_MORE = "I'm not sure what you mean. Please answer the request as it is."
LETTERS = "ABCDEFGH"

SYSTEM = ("You are checking a conversation between a user and a database assistant. "
          "Answer with a single letter only.")


@dataclass
class Reply:
    text: str
    slot_index: int | None  # which hidden slot was answered (None: nothing answered)
    choice: str | None      # the letter the model chose (None when no model call was needed)
    raw: str = ""


def describe_option(slot: dict) -> str:
    if slot["kind"] == "ranking":
        return f'what "{slot["term"]}" should be ranked by (the user means: {slot["value"]})'
    return f'the cut-off meant by "{slot["term"]}" (the user means: {slot["hidden"]})'


def choice_messages(request: str, question: str, slots: list[dict]) -> list[dict]:
    options = [f"{LETTERS[i]}) {describe_option(s)}" for i, s in enumerate(slots)]
    options.append(f"{LETTERS[len(slots)]}) none of these: the question is about something else")
    user = (f'The user asked: "{request}"\n'
            f'The assistant replied with this clarifying question: "{question}"\n\n'
            "Which hidden detail of the user's request is the assistant asking about?\n" + "\n".join(options) +
            "\n\nAnswer with the letter only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_choice(text: str, n_slots: int) -> int | None:
    m = re.search(r"\b([A-H])\b", text.strip().upper())
    if not m:
        return None
    idx = LETTERS.index(m.group(1))
    return idx if idx < n_slots else None


class SimulatedUser:
    def __init__(self, backend):
        self.backend = backend
        self.log: list[dict] = []

    def answer_batch(self, items: list[dict], batch_size: int = 16) -> list[Reply]:
        """items: {"episode", "request", "question", "slots", "answered": set of slot indices already given}."""
        replies: list[Reply | None] = [None] * len(items)
        need_model = []
        for i, it in enumerate(items):
            if not it["slots"]:  # clear or unanswerable episode: nothing hidden to reveal
                replies[i] = Reply(NOTHING_MORE, None, None)
            else:
                need_model.append(i)
        if need_model:
            chats = [choice_messages(items[i]["request"], items[i]["question"], items[i]["slots"]) for i in need_model]
            gens = self.backend.generate(chats, max_new_tokens=4, adapter=False, batch_size=batch_size)
            for i, g in zip(need_model, gens):
                idx = parse_choice(g.text, len(items[i]["slots"]))
                text = items[i]["slots"][idx]["answer"] if idx is not None else NOT_SURE
                replies[i] = Reply(text, idx, g.text.strip()[:1] or None, g.text)
        for it, r in zip(items, replies):
            self.log.append({"episode": it["episode"], "question": it["question"], "choice": r.choice,
                             "slot_index": r.slot_index, "reply": r.text})
        return replies  # type: ignore[return-value]
