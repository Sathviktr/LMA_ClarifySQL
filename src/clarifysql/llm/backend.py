"""Model backends: one interface, a real Hugging Face model for the GPU and a scripted mock for tests.

`generate(chats, adapter=...)` takes a batch of chat conversations (lists of {"role", "content"} messages)
and returns one Generation per conversation. With `adapter=True` the LoRA adapter is used (the clarification
policy); with `adapter=False` the same weights run without it (the SQL writer and the simulated user).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass

from clarifysql.config import MODEL_ID
from clarifysql.llm.hf import greedy_config, load_model

Chat = list[dict]


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    new_tokens: int
    seconds: float  # this item's share of its batch's wall time


class MockBackend:
    """Answers from a Python function, so the agent can be tested on a laptop without any model."""

    def __init__(self, respond: Callable[[Chat, bool], str]):
        self.respond = respond
        self.calls: list[tuple[Chat, bool]] = []

    def generate(self, chats: list[Chat], max_new_tokens: int = 256, adapter: bool = False,
                 batch_size: int = 8) -> list[Generation]:
        out = []
        for chat in chats:
            self.calls.append((chat, adapter))
            text = self.respond(chat, adapter)
            words = sum(len(m["content"].split()) for m in chat)
            out.append(Generation(text, words, len(text.split()), 0.0))
        return out


class HFBackend:
    """Qwen3-4B in FP16 on one GPU, greedy decoding, batched; optionally with a LoRA adapter."""

    def __init__(self, model_id: str = MODEL_ID, adapter_path: str | None = None, mode: str = "fp16"):
        self.model, self.tok, self.load_seconds = load_model(model_id, mode=mode)
        self.tok.padding_side = "left"
        self.has_adapter = False
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()
            self.has_adapter = True

    def _adapter_context(self, adapter: bool):
        if adapter and not self.has_adapter:
            raise ValueError("adapter=True but no adapter was loaded")
        if self.has_adapter and not adapter:
            return self.model.disable_adapter()  # same weights, LoRA switched off
        return contextlib.nullcontext()

    def generate(self, chats: list[Chat], max_new_tokens: int = 256, adapter: bool = False,
                 batch_size: int = 8) -> list[Generation]:
        import torch

        texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths -> less padding
        results: list[Generation | None] = [None] * len(texts)
        cfg = greedy_config(self.model, max_new_tokens)
        pad_id = self.tok.pad_token_id
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), self._adapter_context(adapter):
                out = self.model.generate(**enc, generation_config=cfg)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            share = (time.perf_counter() - t0) / len(idx)
            width = enc["input_ids"].shape[1]
            for row, i in enumerate(idx):
                new = out[row, width:]
                n_new = int((new != pad_id).sum()) if pad_id is not None else int(new.shape[0])
                text = self.tok.decode(new, skip_special_tokens=True).strip()
                results[i] = Generation(text, int(enc["attention_mask"][row].sum()), n_new, share)
        return results  # type: ignore[return-value]
