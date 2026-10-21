"""Every notebook must be valid, in sync with its builder script, and contain code that compiles."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
BUILDER_DIR = ROOT / "notebooks" / "builders"
BUILDERS = sorted(BUILDER_DIR.glob("build_*.py"))
sys.path.insert(0, str(BUILDER_DIR))


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def notebook_name(builder: Path) -> str:
    return builder.stem.removeprefix("build_") + ".ipynb"


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda p: p.stem)
def test_notebook_in_sync_with_builder(builder):
    from nbtools import build

    expected = build(load_builder(builder).CELLS, notebook_name(builder))
    actual = nbformat.read(ROOT / "notebooks" / notebook_name(builder), as_version=4)
    nbformat.validate(actual)
    assert [c.source for c in actual.cells] == [c.source for c in expected.cells], \
        f"notebook out of date: run python notebooks/builders/{builder.name}"


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda p: p.stem)
def test_code_cells_compile(builder):
    for i, cell in enumerate(load_builder(builder).CELLS):
        if cell.cell_type != "code":
            continue
        src = re.sub(r"^(\s*)[!%].*$", r"\1pass", cell.source, flags=re.M)  # shell/magic lines
        compile(src, f"{builder.stem}[cell {i}]", "exec")
