"""Turning conversation states into training examples (proposal, Section 9).

A state (data/sft/*.jsonl, made by scripts/build_sft.py) holds the policy's prompt messages and the reference
next action as compact JSON. The model trains on exactly the text it reads at run time: the chat template applied
to the prompt with the assistant turn opened (as llm/backend.py sends it), then the target and Qwen's end-of-turn
token. Only the target tokens carry a loss (assistant-only loss): the model learns to write the action, not to
repeat the prompt, and the end-of-turn token teaches it to stop.

No PyTorch here: a tokenizer is enough.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from clarifysql.config import REPO_ROOT

SFT_DIR = REPO_ROOT / "data" / "sft"
END_OF_TURN = "<|im_end|>"
IGNORE = -100  # the label value the loss skips


def load_states(name: str) -> list[dict]:
    """name: train, validation, train_noslot or validation_noslot."""
    lines = (SFT_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@dataclass
class Example:
    id: str
    input_ids: list[int]
    labels: list[int]  # IGNORE on the prompt, the token ids on the target

    @property
    def target_tokens(self) -> int:
        return sum(t != IGNORE for t in self.labels)


def prompt_text(tok, messages: list[dict]) -> str:
    """The prompt as the agent sends it at run time: the chat template with the assistant turn opened."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(tok, state: dict) -> Example:
    prompt = prompt_text(tok, state["messages"])
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(prompt + state["target"] + END_OF_TURN, add_special_tokens=False)["input_ids"]
    if ids[:len(prompt_ids)] != prompt_ids:  # the target would be learned in a different context
        raise ValueError(f"{state['id']}: the prompt's tokens change when the target is appended")
    return Example(state["id"], ids, [IGNORE] * len(prompt_ids) + ids[len(prompt_ids):])


def encode_all(tok, states: list[dict], max_len: int = 1024) -> tuple[list[Example], list[str]]:
    """The encoded examples, and the ids of states longer than max_len.

    Long states are left out, never cut: a cut target teaches a broken action, and a cut prompt hides part of
    the state the action depends on. The count of left-out states is the pilot's "truncation" measure.
    """
    kept, over = [], []
    for state in states:
        ex = encode(tok, state)
        if len(ex.input_ids) <= max_len:
            kept.append(ex)
        else:
            over.append(state["id"])
    return kept, over
