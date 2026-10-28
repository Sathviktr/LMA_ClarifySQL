"""QLoRA fine-tuning of the clarification policy (proposal, Section 9).

Settings for a free T4 GPU: the base model stored in 4-bit NF4 with FP16 compute (the T4 has no BF16), a LoRA
adapter of rank 16 on the attention and MLP projections (about 33M trainable parameters), micro-batch 1 with
gradient accumulation, gradient checkpointing, and FP16 mixed precision with loss scaling. The loss covers the
target tokens only (sft_data.py). After every epoch the adapter is saved and judged on the validation states
(state_eval.py); the epoch to keep is chosen by those scores, never by the training loss.

torch / transformers / peft are imported inside functions, so this module imports without them.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from clarifysql.agent.controller import POLICY_MAX_TOKENS
from clarifysql.llm.hf import _sync, _torch, reset_peak
from clarifysql.pilot import LORA_TARGETS, env_report, gpu_memory
from clarifysql.train.sft_data import encode_all, load_states
from clarifysql.train.state_eval import score_states, selection_key


@dataclass
class TrainConfig:
    name: str
    data: str = "train"              # data/sft/<data>.jsonl ("train_noslot" for ablation A2)
    val_data: str = "validation"
    seed: int = 1
    epochs: int = 3
    lr: float = 2e-4
    grad_accum: int = 16             # micro-batch 1, so 16 states per optimizer step
    max_len: int = 1024
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    limit: int | None = None         # train on a seeded sample of this many states (the pilot uses about 100)
    eval_generate: bool = True       # write and score the next action of every validation state after each epoch
    eval_limit: int | None = None
    eval_batch_size: int = 16

    @property
    def with_slots(self) -> bool:
        return "noslot" not in self.data


def set_seed(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attach_lora(model, cfg: TrainConfig):
    """Gradient checkpointing plus a fresh LoRA adapter; only the adapter's weights will train."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    ckpt = {"use_reentrant": False}
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "hf_quantizer", None) is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs=ckpt)
    else:  # full-precision model (CPU tests)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt)
        model.enable_input_require_grads()
    model.config.use_cache = False  # training needs no cache; greedy decoding asks for it (llm/hf.py)
    return get_peft_model(model, LoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                                            target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM"))


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warm-up to the full learning rate, then a cosine decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _forward_loss(model, ex, device, on_gpu: bool):
    torch = _torch()
    ids = torch.tensor([ex.input_ids], device=device)
    labels = torch.tensor([ex.labels], device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=on_gpu):
        return model(input_ids=ids, labels=labels).loss


def validation_loss(model, examples, device, on_gpu: bool) -> float:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        losses = [float(_forward_loss(model, ex, device, on_gpu)) for ex in examples]
    return round(statistics.mean(losses), 4)


def predict(model, tok, states: list[dict], batch_size: int = 16) -> list[str]:
    """The policy's next action for each state, greedy, exactly as the agent would ask for it."""
    from clarifysql.llm.backend import HFBackend

    model.eval()
    backend = HFBackend.wrap(model, tok, has_adapter=True)
    gens = backend.generate([s["messages"] for s in states], max_new_tokens=POLICY_MAX_TOKENS, adapter=True,
                            batch_size=batch_size)
    return [g.text for g in gens]


def train(cfg: TrainConfig, model, tok, out_dir: Path, log_path: Path | None = None) -> dict:
    """Train an adapter on `model`; save it after every epoch to out_dir/epoch-<k>/. Returns the run's log."""
    torch = _torch()
    set_seed(cfg.seed)
    states = load_states(cfg.data)
    if cfg.limit:
        states = random.Random(cfg.seed).sample(states, min(cfg.limit, len(states)))
    train_ex, over = encode_all(tok, states, cfg.max_len)
    val_states = load_states(cfg.val_data)[:cfg.eval_limit]
    val_ex, _ = encode_all(tok, val_states, cfg.max_len)

    model = attach_lora(model, cfg)
    device = next(model.parameters()).device
    on_gpu = device.type == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    n = len(train_ex)
    total = math.ceil(n / cfg.grad_accum) * cfg.epochs
    warmup = max(1, round(cfg.warmup_ratio * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log = {"config": asdict(cfg), "env": env_report(), "train_states": n, "over_limit": over,
           "train_tokens_per_epoch": sum(len(e.input_ids) for e in train_ex), "val_states": len(val_states),
           "trainable_params": sum(p.numel() for p in params), "optimizer_steps": total, "warmup_steps": warmup,
           "steps": [], "epochs": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = list(range(n))
        rng.shuffle(order)
        reset_peak()
        _sync()
        t0, tokens, group, epoch_losses = time.perf_counter(), 0, [], []
        for i, k in enumerate(order):
            ex = train_ex[k]
            size = min(cfg.grad_accum, n - (i // cfg.grad_accum) * cfg.grad_accum)  # the last group may be short
            loss = _forward_loss(model, ex, device, on_gpu)
            scaler.scale(loss / size).backward()
            group.append(float(loss.detach()))
            tokens += len(ex.input_ids)
            if len(group) == size:
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                log["steps"].append({"step": step, "epoch": epoch, "loss": round(statistics.mean(group), 4),
                                     "lr": round(sched.get_last_lr()[0], 8), "grad_norm": round(float(norm), 3)})
                epoch_losses += group
                group = []
        _sync()
        seconds = time.perf_counter() - t0
        rec = {"epoch": epoch, "seconds": round(seconds, 1), "tokens_per_second": round(tokens / seconds, 1),
               "peak_allocated_gb": gpu_memory()["peak_allocated_gb"],
               "train_loss": round(statistics.mean(epoch_losses), 4),
               "val_loss": validation_loss(model, val_ex, device, on_gpu)}
        model.save_pretrained(out_dir / f"epoch-{epoch}")
        if cfg.eval_generate:
            t1 = time.perf_counter()
            answers = predict(model, tok, val_states, cfg.eval_batch_size)
            rec["validation"] = score_states(val_states, answers, cfg.with_slots)
            rec["eval_seconds"] = round(time.perf_counter() - t1, 1)
            if log_path is not None:
                answers_path = log_path.with_name(f"{log_path.stem}_epoch-{epoch}_answers.jsonl")
                answers_path.write_text("".join(json.dumps({"id": s["id"], "target": s["target"], "answer": a},
                                                           ensure_ascii=False) + "\n"
                                                for s, a in zip(val_states, answers)), encoding="utf-8")
        log["epochs"].append(rec)
        if log_path is not None:  # written after every epoch, so a stopped session keeps what it finished
            log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    judged = [e for e in log["epochs"] if "validation" in e]
    if judged:
        log["best_epoch"] = max(judged, key=lambda e: selection_key(e["validation"]))["epoch"]
    if log_path is not None:
        log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return log
