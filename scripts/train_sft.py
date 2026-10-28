"""Fine-tune the clarification policy with QLoRA (proposal, Section 9). Needs a GPU (Kaggle / Colab T4).

Usage:
    python scripts/train_sft.py --name pilot --limit 100 --epochs 2 --grad-accum 8      # checkpoint 5 (pilot)
    python scripts/train_sft.py --name sft_lr2e-4_s1 --seed 1                            # an adapter, seed 1
    python scripts/train_sft.py --name a2_s1 --data train_noslot --val-data validation_noslot   # ablation A2
Saves the adapter after every epoch to outputs/adapters/<name>/epoch-<k>/ and writes results/train/<name>.json
(losses, speed, memory and the validation scores of every epoch) plus the validation answers of every epoch.
Two runs can share the two T4s: start each with CUDA_VISIBLE_DEVICES=0 or =1.
"""

from __future__ import annotations

import argparse
import json

from clarifysql.config import MODEL_ID, outputs_dir, results_dir
from clarifysql.llm.hf import load_model
from clarifysql.train.qlora import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="train", choices=["train", "train_noslot"])
    ap.add_argument("--val-data", default="validation", choices=["validation", "validation_noslot"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="train on a seeded sample of this many states")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--no-eval-generate", action="store_true", help="skip scoring the validation actions")
    args = ap.parse_args()
    if ("noslot" in args.data) != ("noslot" in args.val_data):
        ap.error("--data and --val-data must both be the slot or both the no-slot version")

    cfg = TrainConfig(name=args.name, data=args.data, val_data=args.val_data, seed=args.seed, epochs=args.epochs,
                      lr=args.lr, grad_accum=args.grad_accum, max_len=args.max_len, limit=args.limit,
                      eval_limit=args.eval_limit, eval_generate=not args.no_eval_generate)
    model, tok, seconds = load_model(MODEL_ID, mode="nf4")
    print(f"model loaded in 4-bit NF4 in {seconds:.0f} s")
    log_path = results_dir() / "train" / f"{args.name}.json"
    log = train(cfg, model, tok, outputs_dir() / "adapters" / args.name, log_path)
    print(f"{log['train_states']} training states ({len(log['over_limit'])} over {cfg.max_len} tokens left out), "
          f"{log['trainable_params']:,} trainable parameters, {log['optimizer_steps']} optimizer steps")
    for e in log["epochs"]:
        v = e.get("validation", {})
        print(json.dumps({"epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e["val_loss"],
                          "tokens_per_second": e["tokens_per_second"], "peak_gb": e["peak_allocated_gb"],
                          "valid_rate": v.get("valid_rate"), "decision_macro_f1": v.get("decision_macro_f1"),
                          "slot_accuracy": v.get("slot_accuracy"), "action_accuracy": v.get("action_accuracy")}))
    if "best_epoch" in log:
        print(f"best epoch by the validation scores: {log['best_epoch']}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
