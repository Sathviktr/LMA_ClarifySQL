"""A two-stage simulated user (proposal, Section 10, following BIRD-Interact).

Stage 1: the base model (adapter off) reads ONLY the agent's natural-language question and picks, from a fixed
         multiple-choice list, which hidden details of the episode it asks about (one or several), or "none of these".
Stage 2: a template answers with only those details (the slot's prepared answer, e.g. "I mean more than 4."). One
         question about two details gets both answers, each naming its phrase; a detail asked again gets the same
         answer again, as a real user would repeat it. Questions about nothing hidden get "I'm not sure what you mean."

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
    slot_indices: list[int]  # which hidden slots were answered (empty: nothing answered)
    choice: str | None       # the letters the model chose, e.g. "A,B" (None when no model call was needed)
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
            "\n\nAnswer with the letter only. If the question asks about several of these details, give each "
            "letter, separated by commas (for example: A,B).")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_choices(text: str, n_slots: int) -> list[int]:
    """The slots the chosen letters stand for, in order; [] for "none of these" or no letter."""
    out = []
    for letter in re.findall(r"\b([A-H])\b", text.strip().upper()):
        idx = LETTERS.index(letter)
        if idx < n_slots and idx not in out:
            out.append(idx)
    return out


def reply_text(slots: list[dict], indices: list[int]) -> str:
    if not indices:
        return NOT_SURE
    if len(indices) == 1:
        return slots[indices[0]]["answer"]
    return " ".join(f'For "{slots[i]["term"]}": {slots[i]["answer"]}' for i in indices)


class SimulatedUser:
    def __init__(self, backend):
        self.backend = backend
        self.log: list[dict] = []

    def answer_batch(self, items: list[dict], batch_size: int = 16) -> list[Reply]:
        """items: {"episode", "request", "question", "slots": every hidden slot, already answered or not}."""
        replies: list[Reply | None] = [None] * len(items)
        need_model = []
        for i, it in enumerate(items):
            if not it["slots"]:  # clear or unanswerable episode: nothing hidden to reveal
                replies[i] = Reply(NOTHING_MORE, [], None)
            else:
                need_model.append(i)
        if need_model:
            chats = [choice_messages(items[i]["request"], items[i]["question"], items[i]["slots"]) for i in need_model]
            gens = self.backend.generate(chats, max_new_tokens=8, adapter=False, batch_size=batch_size)
            for i, g in zip(need_model, gens):
                idx = parse_choices(g.text, len(items[i]["slots"]))
                letters = ",".join(re.findall(r"\b([A-H])\b", g.text.strip().upper())) or None
                replies[i] = Reply(reply_text(items[i]["slots"], idx), idx, letters, g.text)
        for it, r in zip(items, replies):
            self.log.append({"episode": it["episode"], "question": it["question"], "choice": r.choice,
                             "slot_indices": r.slot_indices, "reply": r.text})
        return replies  # type: ignore[return-value]
