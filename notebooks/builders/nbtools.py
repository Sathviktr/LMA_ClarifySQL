"""Write Jupyter notebooks from plain Python, so notebook sources stay readable in git diffs.

Each notebook has a builder script next to this file (build_01_pilot.py, ...). Run a builder to
regenerate its .ipynb in notebooks/. tests/test_notebooks.py checks that the two stay in sync.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Sathviktr/LMA_ClarifySQL.git"

SETUP_MD = """\
## 1. Get the project code

This cell clones the project from GitHub (branch `main`) into the notebook's working folder and makes
the `clarifysql` package importable. Running it again later just pulls the newest commits.

*Private repository?* Add a secret called `GITHUB_TOKEN` (Kaggle: **Add-ons → Secrets**; Colab: the
**key icon** in the left bar) and set `USE_TOKEN = True`. Our repository is public, so this is not needed.
"""

SETUP_CODE = f"""\
import os, sys, subprocess

REPO_URL = "{REPO_URL}"
BRANCH = "main"
USE_TOKEN = False  # only needed for a private repository

ON_KAGGLE = os.path.exists("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else "/content"
REPO = f"{{WORK}}/LMA_ClarifySQL"

if USE_TOKEN:
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    else:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    REPO_URL = REPO_URL.replace("https://", f"https://{{token}}@")

if os.path.exists(REPO):
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO], check=True)
print(subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

sys.path.insert(0, f"{{REPO}}/src")
print("Running on:", "Kaggle" if ON_KAGGLE else "Colab / other", "| repo at", REPO)
"""

INSTALL_CODE = """\
# Libraries that are not preinstalled (or too old) on Kaggle/Colab. PyTorch with CUDA is already there.
!pip install -q -r {REPO}/requirements-gpu.txt
"""


def md(text: str):
    return new_markdown_cell(dedent(text).strip())


def code(text: str):
    return new_code_cell(dedent(text).strip())


def build(cells: list, name: str):
    for i, cell in enumerate(cells):
        cell["id"] = f"{Path(name).stem}-{i:02d}"  # stable ids keep regenerated notebooks identical
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
    })
    nbformat.validate(nb)
    return nb


def write(cells: list, name: str) -> Path:
    path = NOTEBOOK_DIR / name
    nbformat.write(build(cells, name), path)
    return path
