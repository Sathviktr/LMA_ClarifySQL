"""Builds notebooks/06_sft_final.ipynb (Phase 5, GPU checkpoint 6: the final adapters)."""

from nbtools import INSTALL_CODE, SETUP_CODE, SETUP_MD, code, md, write

CELLS = [
    md("""
    # ClarifySQL · Notebook 06 — The final adapters

    **Goal** (proposal, Section 9): "Choose one configuration on validation, then train the final adapter with two
    random seeds ... plus one adapter without slot targets for ablation A2." The two T4s work side by side:

    | Part | GPU 0 | GPU 1 | Time |
    |---|---|---|---|
    | A. train | learning rate 2e-4, seed 1 | learning rate 1e-4, seed 1 | about 65 min |
    | B. choose | the agent on the 53 validation episodes, epochs 1–3 | the same, other learning rate | about 20 min |
    | C. train | the chosen setting, seed 2 | the chosen setting without slots (A2) | about 65 min |
    | D. check | the agent with seed 2 | the agent with A2 | about 6 min |

    The rule for choosing was fixed before this run: the highest **decision macro-F1 on whole validation
    episodes** (rounded to 0.01); slot accuracy decides between near-ties.

    **Time:** about 2 hours 45 minutes. **Kaggle only** (it needs two GPUs).
    **How to run:** *Session options* → **GPU T4 x2**, **Internet On**; then **Save Version → Save & Run All
    (Commit)**. It runs in the background, so you can close the browser. The adapters stay in the version's
    output, where Phase 6 will pick them up.
    """),
    code("""
    !nvidia-smi
    import torch
    assert torch.cuda.device_count() == 2, "this notebook needs Kaggle's GPU T4 x2"
    """),
    md(SETUP_MD),
    code(SETUP_CODE),
    code(INSTALL_CODE + "!pip install -q -e {REPO} --no-deps  # lets the scripts import clarifysql\n"
         "!pip install -q pytest\n"
         "# Kaggle ships an old torchao (0.10), and new peft versions then refuse to build LoRA layers.\n"
         "# We do not use torchao, so it is removed.\n"
         "!pip uninstall -y -q torchao\n"),
    md("""
    ## 2. Data, the model, and a last check of the training code (about 5 minutes)

    The model is downloaded once here, so the two training runs do not both download 8 GB at the same time.
    The tiny-model tests must pass; otherwise the notebook stops before any long run starts.
    """),
    code("""
    os.environ["CLARIFYSQL_DATA"] = "/tmp/clarifysql_data"
    !python {REPO}/scripts/download_data.py --only spider
    from huggingface_hub import snapshot_download
    from clarifysql.config import MODEL_ID, outputs_dir
    snapshot_download(MODEL_ID)
    check = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_train.py", "tests/test_train_torch.py",
                            "-p", "no:cacheprovider"], cwd=REPO, capture_output=True, text=True)
    print(check.stdout[-1500:])
    assert check.returncode == 0, "the training tests failed: stop and send this output"
    """),
    code("""
    import json, time

    LOGS = f"{WORK}/logs"
    os.makedirs(LOGS, exist_ok=True)
    ADAPTERS = outputs_dir() / "adapters"
    TRAIN = "python scripts/train_sft.py"
    AGENT = "python scripts/run_agent.py --split validation"

    def run_parallel(jobs: dict) -> None:
        \"\"\"jobs: {name: (gpu, shell command)}. Runs them side by side, one per GPU, and waits for all.\"\"\"
        procs = {}
        for name, (gpu, cmd) in jobs.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            procs[name] = subprocess.Popen(cmd, shell=True, env=env, cwd=REPO, stderr=subprocess.STDOUT,
                                           stdout=open(f"{LOGS}/{name}.log", "w"))
        t0 = time.time()
        codes = {name: p.wait() for name, p in procs.items()}
        print(f"finished in {(time.time() - t0) / 60:.0f} minutes")
        for name, code in codes.items():
            print(f"--- {name}: exit code {code}")
            print("\\n".join(open(f"{LOGS}/{name}.log").read().splitlines()[-8:]))
        assert all(c == 0 for c in codes.values()), "a job failed: see its log above"
    """),
    md("## 3. Part A: two learning rates, seed 1, side by side (about 65 minutes)"),
    code("""
    run_parallel({
        "sft_lr2e-4_s1": (0, f"{TRAIN} --name sft_lr2e-4_s1 --lr 2e-4 --seed 1"),
        "sft_lr1e-4_s1": (1, f"{TRAIN} --name sft_lr1e-4_s1 --lr 1e-4 --seed 1"),
    })
    """),
    md("""
    ## 4. Part B: every epoch of both runs plays the validation episodes (about 20 minutes)

    Each adapter runs inside the full agent (FP16 model, same tools, same simulated user and limits as B3).
    """),
    code("""
    def epochs_of(run: str, system: str = "sft") -> str:
        return " && ".join(f"{AGENT} --system {system} --adapter {ADAPTERS}/{run}/epoch-{k} --name {run}_e{k}"
                           for k in (1, 2, 3))

    run_parallel({"agent_lr2e-4": (0, epochs_of("sft_lr2e-4_s1")), "agent_lr1e-4": (1, epochs_of("sft_lr1e-4_s1"))})
    """),
    md("## 5. The choice: learning rate and epoch"),
    code("""
    from clarifysql.eval.metrics import selection_key

    KEYS = ["execution_accuracy", "decision_macro_f1", "slot_accuracy", "unnecessary_question_rate_clear",
            "correct_abstention_rate", "wrong_abstention_rate_answerable", "invalid_output_rate"]
    cands = {f"sft_lr{lr}_s1_e{k}": json.load(open(f"{REPO}/results/agent/validation_sft_lr{lr}_s1_e{k}_summary.json"))
             for lr in ("2e-4", "1e-4") for k in (1, 2, 3)}
    print(f"{'candidate':22}" + "".join(f"{k[:16]:>18}" for k in KEYS))
    for name, s in cands.items():
        print(f"{name:22}" + "".join(f"{str(s[k]):>18}" for k in KEYS))
    best = max(cands, key=lambda n: selection_key(cands[n]))
    LR, EPOCH = best.split("_lr")[1].split("_")[0], int(best[-1])
    choice = {"rule": "highest decision macro-F1 on the validation episodes (rounded to 0.01); slot accuracy "
                      "decides between near-ties", "chosen": best, "lr": LR, "epoch": EPOCH,
              "candidates": {n: {k: s[k] for k in KEYS} for n, s in cands.items()}}
    json.dump(choice, open(f"{REPO}/results/train/selection.json", "w"), indent=2)
    print(f"\\nchosen: learning rate {LR}, epoch {EPOCH} ({best})")
    """),
    md("## 6. Part C: the chosen setting with seed 2, and the no-slot adapter (A2), side by side (about 65 minutes)"),
    code("""
    run_parallel({
        f"sft_lr{LR}_s2": (0, f"{TRAIN} --name sft_lr{LR}_s2 --lr {LR} --seed 2"),
        f"a2_lr{LR}_s1": (1, f"{TRAIN} --name a2_lr{LR}_s1 --lr {LR} --seed 1 --data train_noslot "
                             "--val-data validation_noslot"),
    })
    """),
    md("## 7. Part D: seed 2 and A2 play the validation episodes (about 6 minutes)"),
    code("""
    run_parallel({
        "agent_s2": (0, f"{AGENT} --system sft --adapter {ADAPTERS}/sft_lr{LR}_s2/epoch-{EPOCH} "
                        f"--name sft_lr{LR}_s2_e{EPOCH}"),
        "agent_a2": (1, f"{AGENT} --system a2 --adapter {ADAPTERS}/a2_lr{LR}_s1/epoch-{EPOCH} "
                        f"--name a2_lr{LR}_s1_e{EPOCH}"),
    })
    """),
    md("""
    ## 8. The final adapters and the validation table

    The three chosen adapters are copied to `final_adapters/`, which Phase 6 attaches as input.
    """),
    code("""
    import shutil

    FINAL = {"sft_s1": f"sft_lr{LR}_s1", "sft_s2": f"sft_lr{LR}_s2", "a2_s1": f"a2_lr{LR}_s1"}
    for short, run in FINAL.items():
        shutil.copytree(ADAPTERS / run / f"epoch-{EPOCH}", f"{WORK}/final_adapters/{short}", dirs_exist_ok=True)
    json.dump({**choice, "final_adapters": {s: f"{r}/epoch-{EPOCH}" for s, r in FINAL.items()}},
              open(f"{WORK}/final_adapters/choice.json", "w"), indent=2)

    rows = {"B3": "b3", "ours, seed 1": f"sft_lr{LR}_s1_e{EPOCH}", "ours, seed 2": f"sft_lr{LR}_s2_e{EPOCH}",
            "A2 (no slots)": f"a2_lr{LR}_s1_e{EPOCH}"}
    print(f"{'system':16}" + "".join(f"{k[:16]:>18}" for k in KEYS))
    for label, name in rows.items():
        s = json.load(open(f"{REPO}/results/agent/validation_{name}_summary.json"))
        print(f"{label:16}" + "".join(f"{str(s.get(k)):>18}" for k in KEYS))
    !du -sh {WORK}/final_adapters/*
    """),
    md("## 9. Pack the results"),
    code("""
    !cd {REPO} && zip -qr {WORK}/checkpoint6_results.zip results/train results/agent
    !cd {WORK} && zip -qr checkpoint6_results.zip logs
    !ls -la {WORK}/checkpoint6_results.zip
    """),
    md("""
    ## 10. Send the results back

    Open the saved version → **Output**, download `checkpoint6_results.zip` and unzip it **inside the repository
    folder** (the files land in `results/train/` and `results/agent/`). Do **not** download the adapters: they stay
    in this version's output for Phase 6.
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "06_sft_final.ipynb"))
