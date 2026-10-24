import importlib.util

import pytest

from clarifysql.config import REPO_ROOT
from clarifysql.eval.agreement import cohen_kappa, confusion


def test_perfect_agreement_is_one():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_textbook_example():
    # 50 items; A says yes 25 times, B says yes 30 times; they agree on 20 yes and 15 no.
    a = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    # observed = 35/50 = 0.7; chance = (25*30 + 25*20) / 2500 = 0.5; kappa = 0.2 / 0.5 = 0.4
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert confusion(a, b)["yes"]["no"] == 5


def test_agreement_no_better_than_chance_is_zero():
    assert cohen_kappa(["x", "y", "x", "y"], ["x", "x", "y", "y"]) == pytest.approx(0.0)


def test_shown_ids_are_stable_and_hide_the_kind():
    spec = importlib.util.spec_from_file_location("mrs", REPO_ROOT / "scripts" / "make_review_sheets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod.shown_id("test-ambiguous-906")
    assert a == mod.shown_id("test-ambiguous-906") and "ambiguous" not in a and a.startswith("test-")
