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
| 1 | Setup, foundations, compute pilot | in progress |
| 2 | Data (Spider, CoSQL, TrustSQL), splits, safe read-only SQL layer, tools | planned |
| 3 | Building episodes (clear / ambiguous / partly answered / definition / unanswerable) | planned |
| 4 | Agent controller, simulated user, baselines B1–B3 | planned |
| 5 | QLoRA fine-tuning (two seeds + no-slot ablation) | planned |
| 6 | Evaluation, statistics, human audits | planned |
| 7 | Demo (fictional university database) and report | planned |

## Repository layout

```
src/clarifysql/     Python package (config, data, db, tools, episodes, agent, llm, train, eval, demo)
tests/              pytest tests (run locally on CPU)
notebooks/          Kaggle/Colab notebooks (GPU work); notebooks/builders/ generates them
scripts/            command-line entry points (data download, splits, episode building, ...)
configs/            training and evaluation settings
data/               small derived files that are committed (splits, episodes); data/raw/ is gitignored
results/            small result files (pilot metrics, predictions, tables)
```

`learning/` (personal learning notes) is intentionally **gitignored**.

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

## GPU work (Kaggle first, Colab as backup)

Notebooks in `notebooks/` clone this repository, install `requirements-gpu.txt` and run on a T4 GPU.
Start with `notebooks/01_pilot.ipynb`. On Kaggle: *Settings → Accelerator → GPU T4 x2* and
*Settings → Internet → On* (requires a phone-verified account).

## Data and licences

Spider and CoSQL are CC BY-SA 4.0 and TrustSQL is CC BY 4.0. Raw benchmark files are downloaded by script
and are not stored in this repository. The derived episode set will be released under CC BY-SA 4.0, as
Spider requires. Only public benchmark data and made-up records are used; no real student data.
