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
| 6 | Final inference, CoSQL, scoring, bootstrap and review sheets | local implementation ready; checkpoint 7 GPU results and human judgments pending |
| 7 | Fictional university demo and final delivery | session engine tested; trained-model/UI demonstration and final report pending |

### Next action: checkpoint 7

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

Each condition saves 16-episode batches with data/code/adapter fingerprints. A matching rerun resumes saved
episodes. Mock outputs are kept separately and rejected by the final analysis. Final inference results are
not yet present in this repository; existing `results/agent/validation_*` files are development results.
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
One GPU is sufficient. The demo takes real replies, shows model text in plain text boxes, and runs queries
only against made-up university records. Its interface blocks sample-value inspection; benchmark tools remain
unchanged. Sharing requires credentials entered at launch. The live model/UI check remains pending.

For a GPU environment outside Kaggle:

```powershell
python -m pip install -r requirements-demo.txt
python scripts/demo_app.py --adapter <path-to-sft_s1>
```

## Data and licences

Spider and CoSQL are CC BY-SA 4.0 and TrustSQL is CC BY 4.0. Raw benchmark files are downloaded by script
and are not stored in this repository. The derived episode set will be released under CC BY-SA 4.0, as
Spider requires. Only public benchmark data and made-up records are used; no real student data.
