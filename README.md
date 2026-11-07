# ClarifySQL

**Teaching a small language model when to ask before it writes SQL** (LMA Group Project, 2026).

People often ask databases vague questions ("Which is the best department?"). Text-to-SQL systems usually
guess silently and can return a confident but wrong answer. ClarifySQL fine-tunes a small open model
(**Qwen3-4B-Instruct-2507**) with **QLoRA** so that, at every step, it picks one action:

| Action | Meaning |
|---|---|
| `inspect` | look something up with a tool (`describe_table`, `sample_values`, `search_definitions`) |
| `clarify` | ask the user one short question about a missing detail |
| `execute` | hand a fully specified request to the (frozen) SQL writer and run it read-only |
| `abstain` | explain that the database does not store the requested information |

The tuned policy is compared with the same model used with a prompt only, with the same tools, SQL writer
and limits. Everything runs on free Kaggle/Colab T4 GPUs, without any paid API.

## Project status

| Phase | Content | Status |
|---|---|---|
| 1 | Setup, foundations, compute pilot | done (pilot: `results/pilot/`) |
| 2 | Data (Spider, CoSQL, TrustSQL), splits, safe read-only SQL layer, tools | done (SQL writer: `results/direct/`) |
| 3 | Reviewed episodes and SFT states | complete: 307 train / 53 validation / 188 test episodes |
| 4 | Agent controller, simulated user, baselines B1–B3 | complete; validation results saved |
| 5 | QLoRA fine-tuning (two seeds + no-slot ablation) | complete; learning rate `2e-4`, epoch 3 selected |
| 6 | Final inference, CoSQL, scoring, bootstrap and review sheets | checkpoint 7 complete and verified; human judgments pending |
| 7 | Fictional university demo and final delivery | checkpoint 8 received and verified: original 5/7; extra SQL check finds a false positive; browser interaction pending |

### Checkpoint 7 results (7 November 2026)

All eight test conditions cover the same 188 episodes (148 answerable, 40 unanswerable), plus two CoSQL
conditions on 59 reviewed turns. The received run used commit `428e743`; its inputs, source hashes,
database bytes and execution/decision summaries have been checked locally.

| System | Execution accuracy | Test-suite accuracy | Decision macro-F1 | Correct abstention |
|---|---:|---:|---:|---:|
| B1 | 56.76% | 46.62% | 0.2207 | 0.0% |
| B2 | 60.14% | 47.30% | 0.2207 | 0.0% |
| B3 | 63.51% | 54.05% | 0.7850 | 87.5% |
| SFT seed 1 | 77.03% | 67.57% | 0.8790 | 72.5% |
| SFT seed 2 | 75.68% | 65.54% | 0.9257 | 72.5% |
| A2: no slots | 77.70% | 68.24% | 0.8923 | 70.0% |
| A3: full context | 72.97% | 62.16% | 0.8533 | 70.0% |
| Oracle | 78.38% | 68.92% | 0.2207 | 0.0% |

The two-seed mean execution gain over B3 is **12.84 percentage points**, with a 95% paired database-bootstrap
interval of **[5.55, 19.08]**. Decision macro-F1 improves by **0.11735**, interval **[0.06545, 0.17945]**.
These intervals use 5,000 paired draws over 20 databases. Both primary comparisons improve, while correct
abstention declines. A2 is only one execution ahead of seed 1 and its difference interval includes zero;
an execution benefit from explicit slots is not established. Human question ratings remain pending.

On CoSQL, accuracy rises from 26/59 to 40/59 and macro-F1 from 0.374 to 0.456, but clarify recall falls
from 8/13 to 3/13. An always-execute baseline gets 45/59 accuracy but only 0.2883 macro-F1 on this
imbalanced slice. Generalization is mixed. See [measured findings](results/final/analysis/findings.md),
[paired intervals](results/final/analysis/paired_bootstrap.json) and [verification scope](results/final/analysis/verification.json).

The [final-report draft](reports/ClarifySQL_Final_Report.pdf) includes the measured test and demo results.
The [independent human reviews](annotation/evaluation/README.md) and browser interaction remain pending.
The mid-submission report remains a historical validation report.

### Checkpoint 8 results (7 November 2026)

All seven real-adapter demo records reproduce on CPU. The original saved checks pass **5/7** and Gradio
construction succeeds. Both age-and-marks cases execute without asking for the age cutoff; the writer
guesses age above 18. Both ranking questions offer poorly grounded age/ID choices, although the fixed
reply supplies a more useful criterion.

The placement case passes the original seed-data comparison, but its SQL counts placement rows rather
than the `placed` indicator. Every student has a placement row, including `placed=0`, so all predicted rates
are 1. Two additional CPU variants make Mathematics and Physics the true winners; the saved query still
returns Computing. These post-hoc checks expose a false positive without replacing the original 5/7.

See [demo findings](results/demo/checkpoint8/analysis/findings.md),
[verification](results/demo/checkpoint8/analysis/verification.json) and
[placement variants](results/demo/checkpoint8/analysis/placement_variants.json).
The [original received ZIP](results/demo/checkpoint8/received.zip) is preserved, including its fictional database.
Run `python scripts/analyze_demo.py` to reproduce this analysis without a GPU. Browser clicks and real
conversation have not yet been demonstrated. No training or checkpoint-7 rerun is needed for this analysis.

### Reproduce checkpoint 8

Import the updated [notebook 08](notebooks/08_demo.ipynb) into Kaggle, enable Internet and a T4 GPU, and
attach the same saved phase6 output (run `363373340`). Keep `RUN_INTERACTIVE = False` and save a run.
The notebook loads `sft_s1` once, checks seven fixed examples and creates
`/kaggle/working/checkpoint8_results.zip`. Return that ZIP; checkpoint 7 and training do not need repeating.

The checks cover a clear request, a glossary definition, two ranking choices, two age cutoffs and missing
email data. Questions, predefined replies, SQL, result rows and failed checks are saved after each case.
Mocks are stored separately and are never real model results. These seven examples are a scripted demo
check, not another benchmark or a live-user study. Strict result comparison flags extra result columns;
an unrecognized clarification can also stop the reply script, so inspect failed traces before assigning blame.

Gradio construction is recorded in `interface.json`; browser interaction remains a separate check.
To try your own replies later, set `RUN_INTERACTIVE = True` and run the last cell interactively.

### Reproduce checkpoint 7

Use [notebook 07](notebooks/07_final_evaluation.ipynb). It runs the required final comparisons without retraining.
Its setup cell clones or updates this GitHub repository and prints the exact commit used. Push the new
evaluation files before running it. No source Dataset upload is required for the default GitHub route.
Attach the saved **Notebook Output** of `sathviktr/phase6`, run **363373340**, containing
`final_adapters/{sft_s1,sft_s2,a2_s1}` and `choice.json`. The downloaded `checkpoint6_results.zip`
contains only logs/metrics and does not include weights. Select GPU T4 x2, enable Internet and save a version
with Save & Run All. Return `checkpoint7_results.zip` for analysis and the independent human review step.

For an optional source-ZIP fallback, run `python scripts/prepare_checkpoint7.py`, upload the resulting
`outputs/clarifysql_checkpoint7_source.zip` as a private Kaggle Dataset, and set `SOURCE = 'archive'`
in the notebook. To reproduce a GitHub run later, set `REVISION` to its printed commit.
The recorded checkpoint-7 code is `428e74397d379c4202a62262885e99ed65e7c62e`; pin that revision when
resuming its original traces. Later demo code changes the source fingerprint and cannot be mixed into that run.

Each condition saves 16-episode batches with data/code/adapter fingerprints. A matching rerun resumes saved
episodes. Mock outputs are kept separately and rejected by the final analysis. Final results are in
`results/final/`; `results/agent/validation_*` files remain the earlier development results.
The evaluation notebook pins Transformers/PEFT/Accelerate/bitsandbytes to the versions recorded in the
completed checkpoint-6 training run; it logs the actual runtime and loaded model revision for each condition.

After the real outputs exist:

```powershell
python scripts/score_final_testsuite.py --root results/final
python scripts/analyze_final.py --root results/final
python scripts/make_eval_audits.py --root results/final
# Two independent raters and a simulator reviewer fill the blank sheets before scoring:
python scripts/score_eval_audits.py
```

The received checkpoint already includes test-suite scoring, bootstrap tables and blank review sheets.
Do not regenerate the sheets after people start filling them. `scripts/verify_final.py` checks the received
results against the recorded Git revision and local database files. `scripts/summarize_checkpoint7.py`
rebuilds the descriptive findings and standalone figures from those saved results.

The eight corrected test slot phrases in `data/episodes/slot_terms.json` affect scoring only. Training data,
requests and simulator inputs are preserved. Both seeds and their mean receive paired database-bootstrap
intervals. The automatic slot score remains a lexical proxy and needs the independent question review.

## Repository layout

```
src/clarifysql/     Python package (config, data, db, tools, episodes, agent, simuser, systems, llm, train, eval, demo)
tests/              pytest tests (run locally on CPU)
notebooks/          Kaggle/Colab notebooks (GPU work); notebooks/builders/ generates them
scripts/            command-line entry points (data download, splits, episode building, ...)
data/               small derived files that are committed (splits, episodes); data/raw/ is gitignored
results/            small result files (pilot metrics, predictions, tables)
annotation/         dataset review sheets; later independent evaluation review sheets
outputs/            local source bundles and large artifacts (gitignored)
```

`learning/` (personal learning notes) is intentionally **gitignored**.
It contains a study index, a project-state handoff, editable sections and one growing PDF per phase.
Training configurations are saved in `results/train/*.json`; final run identities go in their manifests.

## Local setup (Windows, CPU)

```powershell
py -3.13 -m venv C:\Users\<you>\venvs\clarifysql        # keep the venv outside OneDrive
C:\Users\<you>\venvs\clarifysql\Scripts\activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
```

Large raw data (about 1 GB) is stored in the folder named by the `CLARIFYSQL_DATA` environment variable
(default: `data/raw/`, gitignored). Keeping it outside OneDrive avoids syncing it:

```powershell
setx CLARIFYSQL_DATA C:\Users\<you>\clarifysql_data
```

For the current PowerShell session, also set `$env:CLARIFYSQL_DATA = 'C:\Users\<you>\clarifysql_data'`.
Dataset-dependent tests skip if that folder is not configured and downloaded.

## GPU work (Kaggle first, Colab as backup)

Notebooks in `notebooks/` clone this repository, install `requirements-gpu.txt` and run on a T4 GPU.
Start with `notebooks/01_pilot.ipynb`. On Kaggle: *Settings → Accelerator → GPU T4 x2* and
*Settings → Internet → On* (requires a phone-verified account).

### Fictional university demo

[Notebook 08](notebooks/08_demo.ipynb) also loads the code from GitHub and uses the selected `sft_s1` adapter.
One GPU is sufficient. Its default saved run exports the seven scripted checks; the optional interface
takes real replies, shows model text in plain text boxes, and runs queries
only against made-up university records. Its interface blocks sample-value inspection; benchmark tools remain
unchanged. Sharing requires credentials entered at launch. The scripted GPU run and interface construction
are complete; actual browser interaction remains pending. Known failures are recorded above.

For a GPU environment outside Kaggle:

```powershell
python -m pip install -r requirements-demo.txt
python scripts/run_demo_checks.py --adapter <path-to-sft_s1>
python scripts/demo_app.py --adapter <path-to-sft_s1>
```

## Data and licences

Spider and CoSQL are CC BY-SA 4.0 and TrustSQL is CC BY 4.0. Raw benchmark files are downloaded by script
and are not stored in this repository. The derived episode set will be released under CC BY-SA 4.0, as
Spider requires. Only public benchmark data and made-up records are used; no real student data.
