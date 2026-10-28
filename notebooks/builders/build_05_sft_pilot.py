"""Builds notebooks/05_sft_pilot.ipynb (Phase 5, GPU checkpoint 5: the QLoRA pilot)."""

from nbtools import INSTALL_CODE, SETUP_CODE, SETUP_MD, code, md, write

CELLS = [
    md("""
    # ClarifySQL · Notebook 05 — The QLoRA pilot

    **Goal** (proposal, Section 9): "Pilot on about 100 examples and measure peak GPU memory, training tokens per
    second and truncation." The pilot also checks, before the long runs, that

    - the adapter learns the action format (valid JSON actions on the validation states),
    - a saved adapter loads into the full agent and plays whole validation episodes,
    - B3 is re-run with the corrected simulated user, so it can be compared with the fine-tuned model.

    **Time:** about 35 minutes. **GPU quota:** about 35 minutes.
    **Settings:** Kaggle → *Session options* → **GPU T4 x2**, **Internet On**. Colab → *T4 GPU*.
    """),
    code("!nvidia-smi"),
    md(SETUP_MD),
    code(SETUP_CODE),
    code(INSTALL_CODE + "!pip install -q -e {REPO} --no-deps  # lets the scripts import clarifysql\n"
         "!pip install -q pytest\n"),
    md("""
    ## 2. Truncation: do all training states fit in 1,024 tokens? (about 1 minute)

    Every state is encoded exactly as it will be trained: the chat template, the prompt and the target action.
    States longer than the limit would be left out (never cut), so this count is the "truncation" measure.
    """),
    code("""
    from transformers import AutoTokenizer
    from clarifysql.config import MODEL_ID, outputs_dir
    from clarifysql.train.sft_data import encode_all, load_states

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    FULL_TOKENS = {}
    for name in ("train", "validation", "train_noslot", "validation_noslot"):
        examples, over = encode_all(tok, load_states(name), max_len=1024)
        FULL_TOKENS[name] = sum(len(e.input_ids) for e in examples)
        print(f"{name:18} {len(examples)} states fit, {len(over)} longer than 1,024 tokens | "
              f"{FULL_TOKENS[name]:,} tokens per epoch")
    """),
    md("""
    ## 3. The training code on a tiny random model (seconds)

    The same loop that trains the real adapter, on a toy model on the CPU: the loss must fall, only the adapter
    may change, and a saved adapter must reload and give the same answers.
    """),
    code("!cd {REPO} && python -m pytest -q tests/test_train.py tests/test_train_torch.py -p no:cacheprovider"),
    md("""
    ## 4. The pilot: 100 training states, 2 epochs (about 12 minutes)

    The base model is loaded in 4-bit NF4, a LoRA adapter (rank 16, attention + MLP) is attached and trained with
    micro-batch 1 and 8 states per optimizer step. After each epoch the adapter is saved and writes its next action
    for each of the 101 validation states.
    """),
    code("!python {REPO}/scripts/train_sft.py --name pilot --limit 100 --epochs 2 --grad-accum 8"),
    md("## 5. What the pilot measured"),
    code("""
    import json

    log = json.load(open(f"{REPO}/results/train/pilot.json"))
    print(f"{log['train_states']} states, {log['trainable_params']:,} trainable parameters, "
          f"{log['optimizer_steps']} optimizer steps")
    for e in log["epochs"]:
        v = e["validation"]
        print(f"epoch {e['epoch']}: {e['seconds']:.0f} s, {e['tokens_per_second']:.0f} tokens/s, "
              f"peak {e['peak_allocated_gb']} GB | loss {e['train_loss']} (validation {e['val_loss']}) | "
              f"valid {v['valid_rate']}, decision F1 {v['decision_macro_f1']}, slots {v['slot_accuracy']}, "
              f"actions {v['action_accuracy']}")
    minutes = FULL_TOKENS["train"] / log["epochs"][-1]["tokens_per_second"] / 60
    print(f"\\nOne full epoch ({FULL_TOKENS['train']:,} tokens) at this speed: about {minutes:.0f} minutes; "
          f"three epochs: about {3 * minutes:.0f} minutes")
    print("loss per optimizer step:", [s["loss"] for s in log["steps"]])
    """),
    code("""
    rows = [json.loads(line) for line in open(f"{REPO}/results/train/pilot_epoch-2_answers.jsonl")]
    for r in rows[:6]:
        print("reference:", r["target"])
        print("pilot:    ", r["answer"], "\\n")
    """),
    md("""
    ## 6. Whole validation episodes (about 12 minutes)

    The agent runs need the Spider databases. First B3 again, now with the corrected simulated user; then the
    pilot adapter inside the full agent. The pilot saw only 100 states, so its scores are not the point: this
    checks that a trained adapter loads and plays episodes before the long runs depend on it.
    """),
    code("""
    os.environ["CLARIFYSQL_DATA"] = "/tmp/clarifysql_data"
    !python {REPO}/scripts/download_data.py --only spider
    ADAPTER = outputs_dir() / "adapters" / "pilot" / "epoch-2"
    !python {REPO}/scripts/run_agent.py --system b3 --split validation
    !python {REPO}/scripts/run_agent.py --system sft --split validation --adapter {ADAPTER} --name pilot_sft
    """),
    md("## 7. Pack the results"),
    code("""
    !cd {REPO} && zip -qr {WORK}/checkpoint5_results.zip results/train results/agent
    !ls -la {WORK}/checkpoint5_results.zip
    """),
    md("""
    ## 8. Send the results back

    Download `checkpoint5_results.zip` (Kaggle: **Output** panel; Colab: **folder icon**) and unzip it **inside
    the repository folder**, so the files land in `results/train/` and `results/agent/`. Then **stop the session**.
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "05_sft_pilot.ipynb"))
