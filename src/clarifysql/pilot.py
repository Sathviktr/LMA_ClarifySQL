"""Compute pilot: measure what one T4 GPU can do before data and evaluation sizes are fixed.

Proposal Section 13 ("expected fit, then measured"). The notebook 01_pilot.ipynb calls these functions to
record:
  * load time and peak GPU memory of the model in FP16 and in 4-bit NF4,
  * whether FP16 logits stay finite (the checkpoint is stored in BF16, which the T4 cannot run natively),
  * generation speed (tokens per second) at batch sizes 1 and 4, and the latency of one policy-sized call,
  * peak memory and tokens per second of QLoRA training steps at 1,024 and 2,048 tokens,
  * the number of trainable LoRA parameters (formula vs. the real model).

torch / transformers / peft are imported inside functions, so the pure-Python helpers need no GPU stack.
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from clarifysql.llm.hf import _sync, _torch, greedy_config, load_model, reset_peak  # noqa: F401

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

SYSTEM_PROMPT = "You are a careful SQL assistant. Answer with one SQLite query only."
SQL_PROMPT = (
    "Schema: marks(dept TEXT, student TEXT, mark INTEGER)\n"
    "Question: Show departments whose average mark is above 80."
)
CATALOG_LINE = "table students(id, name, age, dept_id, enrol_date); table departments(id, name, building); "


# ----------------------------------------------------------------------------------- pure Python
def lora_param_count(cfg: dict, r: int = 16, targets: tuple[str, ...] = LORA_TARGETS) -> int:
    """Trainable LoRA parameters for a Qwen3-style decoder, computed from its config.json.

    A LoRA pair on a linear layer with d_in inputs and d_out outputs adds r * (d_in + d_out) numbers.
    """
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    head_dim = cfg.get("head_dim") or h // cfg["num_attention_heads"]
    q_out = cfg["num_attention_heads"] * head_dim
    kv_out = cfg["num_key_value_heads"] * head_dim
    shapes = {
        "q_proj": (h, q_out), "k_proj": (h, kv_out), "v_proj": (h, kv_out), "o_proj": (q_out, h),
        "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h),
    }
    return cfg["num_hidden_layers"] * sum(r * (shapes[t][0] + shapes[t][1]) for t in targets)


def save_results(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def summarize(results: dict) -> str:
    """A short human-readable summary of a pilot results dict."""
    lines = []
    env = results.get("env", {})
    gpus = env.get("gpus") or []
    if gpus:
        lines.append(f"GPU: {len(gpus)} x {gpus[0]['name']} ({gpus[0]['memory_gb']} GB each)")
    for key, label in (("fp16_load", "FP16 load"), ("nf4_load", "NF4 load")):
        if key in results:
            r = results[key]
            lines.append(f"{label}: {r.get('seconds')} s, peak {r.get('peak_allocated_gb')} GB")
    for key, label in (("fp16_generation", "FP16 generation"), ("nf4_generation", "NF4 generation")):
        for g in results.get(key, []):
            lines.append(f"{label} batch {g['batch_size']}: {g['tokens_per_second']:.1f} tokens/s "
                         f"(peak {g['peak_allocated_gb']} GB)")
    if "fp16_call_latency" in results:
        c = results["fp16_call_latency"]
        lines.append(f"One policy-sized call ({c['prompt_tokens']} in / {c['new_tokens']} out): "
                     f"{c['seconds']:.2f} s")
    if "fp16_sanity" in results:
        lines.append(f"FP16 logits finite: {results['fp16_sanity']['logits_finite']}")
    q = results.get("qlora")
    if q:
        formula = results.get("lora_formula_params")
        extra = f" (formula: {formula:,})" if isinstance(formula, int) else ""
        lines.append(f"Trainable LoRA parameters: {q['trainable_params']:,}{extra}")
        for run in q["runs"]:
            if run["status"] == "ok":
                lines.append(f"QLoRA step @ {run['seq_len']} tokens: {run['seconds_per_step']:.2f} s/step, "
                             f"{run['tokens_per_second']:.0f} tokens/s, peak {run['peak_allocated_gb']} GB")
            else:
                lines.append(f"QLoRA step @ {run['seq_len']} tokens: {run['status']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- GPU helpers
# (loading and greedy decoding live in clarifysql/llm/hf.py)


def gpu_memory(device: int = 0) -> dict[str, float | None]:
    """Peak and current memory that PyTorch has allocated on one GPU, in GB (1e9 bytes)."""
    torch = _torch()
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": None, "allocated_gb": None, "free_gb": None, "total_gb": None}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 1e9, 3),
        "free_gb": round(free / 1e9, 3),
        "total_gb": round(total / 1e9, 3),
    }


def free_gpu() -> None:
    """Release cached memory after `del model` in the caller."""
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def env_report() -> dict[str, Any]:
    torch = _torch()
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    for name in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            info[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info[name] = None
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            major, minor = torch.cuda.get_device_capability(i)
            gpus.append({
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(props.total_memory / 1e9, 2),
                "compute_capability": f"{major}.{minor}",
                "native_bf16": major >= 8,
            })
        info["gpus"] = gpus
    return info


def chat_text(tok, user: str, system: str = SYSTEM_PROMPT) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def fp16_sanity(model, tok, prompt: str = SQL_PROMPT, max_new_tokens: int = 80) -> dict[str, Any]:
    """Run one forward pass and one greedy answer; report whether all logits are finite numbers."""
    torch = _torch()
    enc = tok(chat_text(tok, prompt), return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits
        out = model.generate(**enc, generation_config=greedy_config(model, max_new_tokens))
    answer = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "logits_finite": bool(torch.isfinite(logits).all()),
        "max_abs_logit": round(float(logits.float().abs().max()), 2),
        "prompt": prompt,
        "sample_answer": answer.strip(),
    }


def measure_generation(model, tok, batch_sizes=(1, 4), max_new_tokens: int = 128,
                       prompt: str = SQL_PROMPT) -> list[dict[str, Any]]:
    """Tokens per second for greedy generation of exactly `max_new_tokens` tokens per sequence."""
    torch = _torch()
    tok.padding_side = "left"
    text = chat_text(tok, prompt)
    warm = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**warm, generation_config=greedy_config(model, 4, 4))
    results = []
    for bs in batch_sizes:
        enc = tok([text] * bs, return_tensors="pt", padding=True).to(model.device)
        reset_peak()
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, generation_config=greedy_config(model, max_new_tokens, max_new_tokens))
        _sync()
        seconds = time.perf_counter() - t0
        new_tokens = (out.shape[1] - enc["input_ids"].shape[1]) * bs
        results.append({
            "batch_size": bs,
            "new_tokens": int(new_tokens),
            "seconds": round(seconds, 3),
            "tokens_per_second": round(new_tokens / seconds, 2),
            "peak_allocated_gb": gpu_memory()["peak_allocated_gb"],
        })
    return results


def measure_call_latency(model, tok, prompt_tokens: int = 600, new_tokens: int = 60) -> dict[str, Any]:
    """Seconds for one policy-sized call: about 600 prompt tokens in, 60 tokens (one JSON action) out."""
    torch = _torch()
    filler = CATALOG_LINE * (prompt_tokens // 20 + 1)
    ids = tok(filler, add_special_tokens=False)["input_ids"][:prompt_tokens]
    user = tok.decode(ids) + "\nQuestion: which department has the oldest students?"
    enc = tok(chat_text(tok, user), return_tensors="pt").to(model.device)
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**enc, generation_config=greedy_config(model, new_tokens, new_tokens))
    _sync()
    return {"prompt_tokens": int(enc["input_ids"].shape[1]), "new_tokens": new_tokens,
            "seconds": round(time.perf_counter() - t0, 3)}


def measure_qlora_steps(model, seq_lens=(1024, 2048), steps: int = 4, r: int = 16, alpha: int = 32,
                        lr: float = 2e-4, seed: int = 0):
    """Attach a LoRA adapter (r=16 on attention + MLP) and time a few training steps per sequence length.

    The input is random token ids: the loss value is meaningless, but memory use and speed are realistic
    for sequences of that length. Returns (results, peft_model).
    """
    torch = _torch()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    device = next(model.parameters()).device
    on_gpu = device.type == "cuda"
    ckpt_kwargs = {"use_reentrant": False}
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "hf_quantizer", None) is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs=ckpt_kwargs)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt_kwargs)
        model.enable_input_require_grads()
    peft_model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                                                  target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM"))
    params = [p for p in peft_model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)
    vocab = peft_model.config.vocab_size
    gen = torch.Generator().manual_seed(seed)
    peft_model.train()
    runs = []
    for seq_len in seq_lens:
        rec: dict[str, Any] = {"seq_len": seq_len}
        try:
            reset_peak()
            ids = torch.randint(10, vocab - 10, (1, seq_len), generator=gen).to(device)
            times, losses = [], []
            for _ in range(steps):
                _sync()
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=on_gpu):
                    loss = peft_model(input_ids=ids, labels=ids).loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                _sync()
                times.append(time.perf_counter() - t0)
                losses.append(float(loss.detach()))
            step_s = statistics.median(times[1:]) if len(times) > 1 else times[0]
            rec.update({
                "status": "ok",
                "seconds_per_step": round(step_s, 3),
                "first_step_seconds": round(times[0], 3),
                "tokens_per_second": round(seq_len / step_s, 1),
                "peak_allocated_gb": gpu_memory()["peak_allocated_gb"],
                "loss_first": round(losses[0], 3),
                "loss_last": round(losses[-1], 3),
            })
        except torch.cuda.OutOfMemoryError:
            rec["status"] = "out_of_memory"
            opt.zero_grad(set_to_none=True)
            free_gpu()
        runs.append(rec)
    results = {"r": r, "alpha": alpha, "targets": list(LORA_TARGETS),
               "trainable_params": int(trainable), "runs": runs}
    return results, peft_model
