"""Builds notebooks/04_agent_validation.ipynb (Phase 4, GPU checkpoint 4: baselines on validation episodes)."""

from nbtools import INSTALL_CODE, SETUP_CODE, SETUP_MD, code, md, write

CELLS = [
    md("""
    # ClarifySQL · Notebook 04 — The agent and the baselines on the validation episodes

    **Goal.** Run every system that does not need a trained adapter on the 56 written validation episodes,
    with the same SQL writer, tools, simulated user and limits:

    | system | what it does |
    |---|---|
    | B1 / B2 | the request goes straight to the SQL writer (B2: plus up to 2 repairs) |
    | oracle | the original, fully specified question goes to the SQL writer (upper bound) |
    | B3 | the full agent; the policy is the base model with instructions **and worked examples** |
    | B3-zero | the same, with instructions only (a prompt variant) |

    Validation results are used to choose B3's prompt and to find problems **before** training. Test episodes
    are not touched until the final evaluation.

    **Time:** about 25 minutes. **GPU quota:** about 25 minutes.
    **Settings:** Kaggle → *Session options* → **GPU T4 x2**, **Internet On**. Colab → *T4 GPU*.
    """),
    code("!nvidia-smi"),
    md(SETUP_MD),
    code(SETUP_CODE),
    code(INSTALL_CODE + "!pip install -q -e {REPO} --no-deps  # lets the scripts import clarifysql\n"),
    md("## 2. Download Spider (tools, the SQL writer and the scoring use the real databases)"),
    code("""
    os.environ["CLARIFYSQL_DATA"] = "/tmp/clarifysql_data"
    !python {REPO}/scripts/download_data.py --only spider
    """),
    md("""
    ## 3. Check the whole pipeline without the model (seconds)

    A scripted policy that follows each episode's reference path, a keyword-matching simulated user and a
    gold-SQL writer must score **1.0** on every metric. Anything less is a bug in our code, not in the model.
    """),
    code("""
    !python {REPO}/scripts/run_agent.py --system reference --split validation --backend mock
    !rm -f {REPO}/results/agent/*_mock*
    """),
    md("## 4. Direct baselines B1 / B2 and the oracle (about 3 minutes)"),
    code("""
    !python {REPO}/scripts/run_agent.py --system direct --split validation
    !python {REPO}/scripts/run_agent.py --system oracle --split validation
    """),
    md("""
    ## 5. The prompted agent B3, two prompt variants (about 15 minutes)

    The model plays three roles: the policy (adapter off), the simulated user (adapter off, two-stage) and the
    SQL writer (adapter off). Episodes run in lockstep, so every round is one batched call per role.
    """),
    code("""
    !python {REPO}/scripts/run_agent.py --system b3 --split validation
    !python {REPO}/scripts/run_agent.py --system b3zero --split validation
    """),
    md("## 6. Comparison table"),
    code("""
    import json, glob
    cols = ["execution_accuracy", "decision_macro_f1", "unnecessary_question_rate_clear", "correct_abstention_rate",
            "avg_clarification_turns", "invalid_output_rate", "limit_rate"]
    print(f"{'system':<14}" + "".join(f"{c[:18]:>20}" for c in cols))
    for path in sorted(glob.glob(f"{REPO}/results/agent/validation_*_summary.json")):
        s = json.load(open(path))
        print(f"{s['name']:<14}" + "".join(f"{str(s[c]):>20}" for c in cols))
    """),
    md("## 7. Pack the results"),
    code("""
    !cd {REPO} && zip -qr {WORK}/checkpoint4_results.zip results/agent
    !ls -la {WORK}/checkpoint4_results.zip
    """),
    md("""
    ## 8. Send the results back

    Download `checkpoint4_results.zip` (Kaggle: **Output** panel; Colab: **folder icon**) and unzip it **inside
    the repository folder**, so the files land in `results/agent/`. Then **stop the session**.
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "04_agent_validation.ipynb"))
