from pathlib import Path

from clarifysql import config


def test_budgets_match_proposal():
    b = config.DEFAULT_BUDGETS
    assert (b.max_inspections, b.max_questions, b.max_repairs, b.max_model_calls) == (3, 2, 2, 8)


def test_data_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLARIFYSQL_DATA", str(tmp_path))
    assert config.data_dir() == tmp_path


def test_data_dir_default_is_gitignored_folder(monkeypatch):
    monkeypatch.delenv("CLARIFYSQL_DATA", raising=False)
    monkeypatch.setattr(config, "runtime", lambda: "local")
    assert config.data_dir() == config.REPO_ROOT / "data" / "raw"
    gitignore = (config.REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/" in gitignore


def test_learning_folder_is_gitignored():
    gitignore = (config.REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "learning/" in gitignore.splitlines()


def test_runtime_detects_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert config.runtime() == "kaggle"


def test_repo_root_contains_pyproject():
    assert (config.REPO_ROOT / "pyproject.toml").exists()
    assert isinstance(config.REPO_ROOT, Path)
