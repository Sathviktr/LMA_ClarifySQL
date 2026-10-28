"""Loading the model with Hugging Face, and greedy decoding settings (shared by the pilot and the backends)."""

from __future__ import annotations

import copy
import time
from importlib import metadata
from typing import Any

from clarifysql.config import MODEL_ID


def _torch():
    import torch

    return torch


def _sync() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _dtype_kwarg() -> str:
    """transformers >= 4.56 calls the argument `dtype`; older versions call it `torch_dtype`."""
    try:
        major, minor = (int(x) for x in metadata.version("transformers").split(".")[:2])
    except Exception:
        return "torch_dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_model(model_id: str = MODEL_ID, mode: str = "fp16"):
    """Load tokenizer + model. mode: "fp16", "nf4" (4-bit QLoRA storage) or "fp32" (CPU tests).

    Returns (model, tokenizer, load_seconds).
    """
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict[str, Any] = {}
    key = _dtype_kwarg()
    if mode == "fp16":
        kwargs[key] = torch.float16
    elif mode == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs[key] = torch.float16
    elif mode == "fp32":
        kwargs[key] = torch.float32
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    reset_peak()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    _sync()
    model.eval()
    return model, tok, time.perf_counter() - t0


def greedy_config(model, max_new_tokens: int, min_new_tokens: int = 0):
    """Greedy decoding settings (no sampling), as fixed in the proposal for every system."""
    cfg = copy.deepcopy(model.generation_config)
    cfg.do_sample = False
    cfg.temperature = None
    cfg.top_p = None
    cfg.top_k = None
    cfg.max_new_tokens = max_new_tokens
    cfg.min_new_tokens = min_new_tokens
    cfg.use_cache = True  # also after training switched the model's own cache setting off
    if cfg.pad_token_id is None:
        cfg.pad_token_id = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
    return cfg
