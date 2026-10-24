"""Builds notebooks/02_sql_writer.ipynb (Phase 2, GPU checkpoint 2: the frozen SQL writer, B1 / oracle)."""

from nbtools import INSTALL_CODE, SETUP_CODE, SETUP_MD, code, md, write

CELLS = [
    md("""
    # ClarifySQL · Notebook 02 — The frozen SQL writer (baseline B1 and the oracle)

    **Goal.** Measure how well the base model, with no adapter and no questions, turns a fully specified
    request into SQL. We use the original Spider questions of our validation (537) and test (1,034)
    databases. On these clear questions B1 and the oracle receive exactly the same input. The result is the
    **upper bound** every later system is compared with, plus the real cost per query.

    Each question gets one answer, and up to **2 repairs** if SQLite reports an error. The files keep the
    answer *before* repair as well, so one run gives both B1 (no repair) and B2 (with repair).

    **Time:** about 45 minutes. **GPU quota:** about 45 minutes.
    **Settings:** Kaggle → *Session options* → **GPU T4 x2**, **Internet On**.
    Colab → *Runtime → Change runtime type → T4 GPU*.
    """),
    code("!nvidia-smi"),
    md(SETUP_MD),
    code(SETUP_CODE),
    code(INSTALL_CODE + "!pip install -q -e {REPO} --no-deps  # lets the scripts import clarifysql\n"),
    md("""
    ## 2. Download Spider (about 1 minute)

    The data goes to `/tmp/clarifysql_data`, which is outside the notebook's output folder, so the saved
    notebook version stays small. The script compares the archive's SHA-256 fingerprint with the one in the
    repository's `data/manifest.json` and says whether they match.

    *If Google Drive refuses ("too many users"):* upload `spider_data.zip` from your laptop
    (`C:\\Users\\sathv\\clarifysql_data\\downloads`) as a private Kaggle Dataset, attach it with
    **Add Input**, and copy it to `/tmp/clarifysql_data/downloads/` before running the cell again.
    """),
    code("""
    os.environ["CLARIFYSQL_DATA"] = "/tmp/clarifysql_data"
    !python {REPO}/scripts/download_data.py --only spider
    """),
    md("""
    ## 3. Check the pipeline without the model (seconds)

    A fake "model" that answers with the gold SQL must score 100%. This checks the prompts, the SQL
    extraction, the safe executor and the result comparison before we spend GPU time.
    """),
    code("""
    !python {REPO}/scripts/run_direct.py --split validation --backend mock-gold --limit 50
    !rm -f {REPO}/results/direct/*mock-gold*
    """),
    md("""
    ## 4. Validation questions (537, about 10 minutes)

    The model is loaded in FP16, and questions are sorted by length and sent in batches of 16 (the pilot
    showed batching multiplies throughput). Greedy decoding throughout.
    """),
    code("!python {REPO}/scripts/run_direct.py --split validation --max-repairs 2 --batch-size 16"),
    md("## 5. Test questions (1,034, about 20 minutes)"),
    code("!python {REPO}/scripts/run_direct.py --split test --max-repairs 2 --batch-size 16"),
    md("""
    ## 6. Official test-suite accuracy (CPU, about 10 minutes)

    Downloads the database variants of the test-suite evaluator (1.3 GB) and re-scores every test
    prediction on all variants of its database. Set `RUN_TESTSUITE = False` to skip this; it can also be
    done later on the laptop.
    """),
    code("""
    RUN_TESTSUITE = True
    if RUN_TESTSUITE:
        !python {REPO}/scripts/download_data.py --only testsuite
        !python {REPO}/scripts/score_testsuite.py {REPO}/results/direct/test_oracle_repair2.jsonl
    """),
    md("## 7. Summary"),
    code("""
    import json, glob
    for path in sorted(glob.glob(f"{REPO}/results/direct/*_summary.json")):
        s = json.load(open(path))
        print(f"{s['split']:<10} items={s['items']}  accuracy before repair={s['execution_accuracy_before_repair']}"
              f"  after repair={s['execution_accuracy']}  test-suite={s.get('testsuite_accuracy', '-')}"
              f"  seconds/item={s['mean_seconds_per_item']}  wall={s['wall_seconds']}s")
    """),
    md("## 8. Pack the results"),
    code("""
    !cd {REPO} && zip -qr {WORK}/checkpoint2_results.zip results/direct
    !ls -la {WORK}/checkpoint2_results.zip
    """),
    md("""
    ## 9. Send the results back

    Download `checkpoint2_results.zip`: on Kaggle from the **Output** panel (`/kaggle/working`), or on
    Colab from the **folder icon** (`/content`). Unzip it **inside the repository folder** on your laptop,
    so the files land in `LMA_ClarifySQL/results/direct/`. Then **stop the session** so it does not use up
    GPU quota while idle.
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "02_sql_writer.ipynb"))
