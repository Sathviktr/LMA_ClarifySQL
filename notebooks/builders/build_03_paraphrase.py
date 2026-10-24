"""Builds notebooks/03_paraphrase.ipynb (Phase 3, GPU checkpoint 3: polish the training requests)."""

from nbtools import INSTALL_CODE, SETUP_CODE, SETUP_MD, code, md, write

CELLS = [
    md("""
    # ClarifySQL · Notebook 03 — Paraphrasing the training requests

    **Goal.** The rule-based drafts of the 180 vague and unanswerable *training* requests are sometimes clumsy
    ("grade high point"). The base model rewrites each one from the original question plus a plain description
    of the detail to hide. Every rewrite must pass the same automatic checks as the drafts (hidden value gone,
    gold SQL still runs, nothing leaked...); otherwise the draft is kept. Validation and test requests are
    **not** touched: the team edits those, so the tuned model is never tested on its own writing style.

    **Time:** about 10 minutes (most of it loading the model). **GPU quota:** about 10 minutes.
    **Settings:** Kaggle → *Session options* → **GPU T4 x2**, **Internet On**. Colab → *T4 GPU*.
    """),
    code("!nvidia-smi"),
    md(SETUP_MD),
    code(SETUP_CODE),
    code(INSTALL_CODE + "!pip install -q -e {REPO} --no-deps  # lets the scripts import clarifysql\n"),
    md("## 2. Download Spider (the checks run the gold SQL on the real databases)"),
    code("""
    os.environ["CLARIFYSQL_DATA"] = "/tmp/clarifysql_data"
    !python {REPO}/scripts/download_data.py --only spider
    """),
    md("""
    ## 3. Check the pipeline without the model (seconds)

    A fake "model" that returns the rule drafts unchanged: almost every draft should be accepted (they passed
    the same checks when the episodes were built).
    """),
    code("""
    !python {REPO}/scripts/paraphrase_train.py --backend mock
    !rm -f {REPO}/results/episodes/paraphrases_train_mock.jsonl
    """),
    md("""
    ## 4. Paraphrase with the base model (FP16, greedy, batches of 16)

    The output shows how many rewrites were accepted and the most common reasons for rejection.
    """),
    code("!python {REPO}/scripts/paraphrase_train.py --batch-size 16"),
    md("## 5. A few examples: draft → rewrite"),
    code("""
    import json
    rows = [json.loads(l) for l in open(f"{REPO}/results/episodes/paraphrases_train.jsonl", encoding="utf-8")]
    for r in rows[:12]:
        mark = "OK " if r["accepted"] else "NO "
        print(f"{mark} draft: {r['draft']}\\n    model: {r['paraphrase']}\\n    {'; '.join(r['reasons'])}\\n")
    """),
    md("## 6. Pack the result"),
    code("""
    !cd {REPO} && zip -q {WORK}/checkpoint3_results.zip results/episodes/paraphrases_train.jsonl
    !ls -la {WORK}/checkpoint3_results.zip
    """),
    md("""
    ## 7. Send the result back

    Download `checkpoint3_results.zip` (Kaggle: **Output** panel; Colab: **folder icon**) and unzip it
    **inside the repository folder**, so that `results/episodes/paraphrases_train.jsonl` lands in place. Then
    **stop the session**.
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "03_paraphrase.ipynb"))
