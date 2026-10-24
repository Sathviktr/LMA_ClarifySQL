"""Count Qwen3 tokens without loading PyTorch: the Rust `tokenizers` library plus Qwen's chat format.

Qwen3's chat template (without tools) writes every message as
    <|im_start|>{role}\n{content}<|im_end|>\n
so a conversation's text can be rebuilt exactly and counted with the model's own tokenizer.json.
Used by the data-building scripts, which only need lengths, not the model.
"""

from __future__ import annotations

from functools import lru_cache

from clarifysql.config import MODEL_ID


@lru_cache(maxsize=1)
def qwen_tokenizer():
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from tokenizers import Tokenizer

    path = try_to_load_from_cache(MODEL_ID, "tokenizer.json")
    if not isinstance(path, str):
        path = hf_hub_download(MODEL_ID, "tokenizer.json")
    return Tokenizer.from_file(path)


def chat_text(messages: list[dict]) -> str:
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def count_chat_tokens(messages: list[dict]) -> int:
    return len(qwen_tokenizer().encode(chat_text(messages), add_special_tokens=False).ids)
