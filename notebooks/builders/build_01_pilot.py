"""Builds notebooks/01_pilot.ipynb (Phase 1, GPU checkpoint 1: the compute pilot)."""

from nbtools import INSTALL_CODE, SETUP_CODE, SETUP_MD, code, md, write

CELLS = [
    md("""
    # ClarifySQL · Notebook 01 — Compute pilot

    **Goal.** Before we fix data sizes and the training plan, measure what one free T4 GPU can really do
    with `Qwen/Qwen3-4B-Instruct-2507` (proposal, Section 13: *expected fit, then measured*).

    **Time:** about 20–30 minutes; most of it is the one-time 8 GB model download.
    **GPU quota used:** about half an hour.

    **Before you press *Run all***
    - **Kaggle:** in the right-hand panel open *Session options* → **Accelerator: GPU T4 x2**, and turn
      **Internet: On** (needs a phone-verified account).
    - **Colab:** *Runtime → Change runtime type →* **T4 GPU**.

    **At the end**, download `results/pilot/pilot_metrics.json` (last cell explains how) and copy it into the
    repository folder `results/pilot/` on your laptop.
    """),
    md("""
    ## 0. Check the GPU

    `nvidia-smi` lists the GPUs this notebook can use. On Kaggle you should see **two Tesla T4** cards with
    about 15 GB each. They are separate cards: one model must fit on *one* of them. If you see
    *"command not found"* or no GPU, the accelerator is not switched on (see the first cell).
    """),
    code("!nvidia-smi"),
    md(SETUP_MD),
    code(SETUP_CODE),
    code(INSTALL_CODE),
    md("""
    ## 2. Environment report

    Records the versions of Python, PyTorch, CUDA and the Hugging Face libraries, plus each GPU's memory and
    *compute capability*. The T4 has capability 7.5, which means **no native BF16**. That is why we compute
    in FP16.
    """),
    code("""
    from clarifysql import pilot, config

    results = {"model_id": config.MODEL_ID, "env": pilot.env_report()}
    results["env"]
    """),
    md("""
    ## 3. FP16: load the full model in 16-bit

    4.02 billion parameters × 2 bytes ≈ **8 GB** of weights. The first run downloads them from Hugging Face
    (a few minutes). `peak_allocated_gb` is the most memory PyTorch used at any moment.
    """),
    code("""
    model, tok, seconds = pilot.load_model(config.MODEL_ID, mode="fp16")
    results["fp16_load"] = {"seconds": round(seconds, 1), **pilot.gpu_memory()}
    results["fp16_load"]
    """),
    md("""
    ## 4. FP16 sanity check

    The weights were published in BF16. FP16 has a much smaller range (largest number 65,504), so a value can
    overflow to `inf`. We run one forward pass, check that every output number (*logit*) is finite, and look
    at a real greedy answer to a tiny text-to-SQL question.
    """),
    code("""
    results["fp16_sanity"] = pilot.fp16_sanity(model, tok)
    print("logits finite:", results["fp16_sanity"]["logits_finite"])
    print(results["fp16_sanity"]["sample_answer"])
    """),
    md("""
    ## 5. Generation speed

    To produce **one** token the GPU must read **all** 8 GB of weights. With a batch of 4 sequences it produces
    4 tokens per read, so throughput should rise almost 4×. We also time one *policy-sized* call: about 600
    prompt tokens in and 60 tokens (one JSON action) out. This is the unit for estimating the cost of the
    agent's episodes.
    """),
    code("""
    results["fp16_generation"] = pilot.measure_generation(model, tok, batch_sizes=(1, 4), max_new_tokens=128)
    results["fp16_call_latency"] = pilot.measure_call_latency(model, tok)
    results["fp16_generation"], results["fp16_call_latency"]
    """),
    md("## 6. Free the GPU before loading the 4-bit version"),
    code("""
    del model
    pilot.free_gpu()
    pilot.gpu_memory()
    """),
    md("""
    ## 7. NF4: load the model in 4-bit (how QLoRA stores the frozen weights)

    Each weight becomes a 4-bit code plus a shared scale per block of 64 weights, so we expect roughly
    **2.5–3 GB**. Generation in 4-bit is usually *slower*, because every weight must be unpacked before use.
    That is why the proposal runs inference in FP16.
    """),
    code("""
    model4, tok, seconds = pilot.load_model(config.MODEL_ID, mode="nf4")
    results["nf4_load"] = {"seconds": round(seconds, 1), **pilot.gpu_memory()}
    results["nf4_generation"] = pilot.measure_generation(model4, tok, batch_sizes=(1,), max_new_tokens=64)
    results["nf4_load"], results["nf4_generation"]
    """),
    md("""
    ## 8. QLoRA training steps

    We attach a LoRA adapter (rank 16 on the 7 attention and MLP matrices of every layer) and time 4 training
    steps at **1,024** and **2,048** tokens with micro-batch 1 and gradient checkpointing. The input is random
    token ids, so the loss means nothing here. We are measuring memory and speed. The trainable-parameter
    count should equal the formula: **33,030,144**.
    """),
    code("""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(config.MODEL_ID).to_dict()
    results["lora_formula_params"] = pilot.lora_param_count(cfg, r=16)
    results["qlora"], peft_model = pilot.measure_qlora_steps(model4, seq_lens=(1024, 2048), steps=4)
    results["qlora"]
    """),
    md("## 9. Save and summarise"),
    code("""
    import datetime

    results["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    path = pilot.save_results(results, f"{WORK}/results/pilot/pilot_metrics.json")
    print("saved", path, "\\n")
    print(pilot.summarize(results))
    """),
    md("""
    ## 10. Send the results back

    - **Kaggle (interactive session):** in the right-hand panel open **Output** (or the file browser) →
      `/kaggle/working/results/pilot/pilot_metrics.json` → **Download**.
      If you used **Save Version → Save & Run All**, open that version → **Output** tab → download the file.
    - **Colab:** click the **folder icon** in the left bar → `/content/results/pilot/pilot_metrics.json` →
      right-click → **Download**.

    Copy the file to `LMA_ClarifySQL/results/pilot/pilot_metrics.json` on your laptop and commit it: the
    next phase uses these numbers. Then **stop the session** (Kaggle: power icon / *Stop session*), so it does not use up GPU
    quota while idle.
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "01_pilot.ipynb"))
