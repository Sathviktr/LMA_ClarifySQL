"""Project-wide constants: paths, model id, per-episode limits and seeds.

Every other module reads these values from here, so one change applies everywhere.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

SEEDS = (1, 2)


@dataclass(frozen=True)
class Budgets:
    """Hard limits for one episode (proposal, Figure 1)."""

    max_inspections: int = 3
    max_questions: int = 2
    max_repairs: int = 2
    max_model_calls: int = 8


DEFAULT_BUDGETS = Budgets()


def runtime() -> str:
    """Return "kaggle", "colab" or "local" depending on where the code runs."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    return "local"


def data_dir() -> Path:
    """Folder for large raw data (Spider, CoSQL and TrustSQL with their databases).

    Set the CLARIFYSQL_DATA environment variable to keep this data outside the repository
    (and outside OneDrive). Without it, data goes to <repo>/data/raw, which is gitignored.
    """
    env = os.environ.get("CLARIFYSQL_DATA")
    if env:
        return Path(env).expanduser()
    if runtime() == "kaggle":
        return Path("/kaggle/working/data")
    return REPO_ROOT / "data" / "raw"


def results_dir() -> Path:
    """Small result files (metrics, predictions, logs) that are committed to git."""
    return REPO_ROOT / "results"


def outputs_dir() -> Path:
    """Large run outputs (adapters, caches) that are never committed."""
    if runtime() == "kaggle":
        return Path("/kaggle/working/outputs")
    return REPO_ROOT / "outputs"
