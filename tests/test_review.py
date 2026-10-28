import csv
import importlib.util

import pytest

from clarifysql.config import REPO_ROOT
from clarifysql.episodes.review import audit_drops, cell, dropped, edit_drops, label_drops, read_sheet
from clarifysql.episodes.schema import EPISODE_DIR


def write(path, header, rows, encoding="utf-8-sig"):
    with open(path, "w", encoding=encoding, newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_cell_matches_the_header_prefix():
    assert cell({"verdict (keep / drop)": " Drop ", "auditor": "SR"}, "verdict") == "Drop"
    assert cell({"auditor": "SR"}, "verdict") == ""


def test_a_sheet_saved_as_plain_excel_csv_still_reads(tmp_path):
    write(tmp_path / "x.csv", ["request", "notes"], [["Show the café owners", "é"]], encoding="cp1252")
    assert read_sheet("x.csv", tmp_path)[0]["request"] == "Show the café owners"
    assert read_sheet("missing.csv", tmp_path) == []


def test_only_signed_rows_drop(tmp_path):
    write(tmp_path / "test_edit.csv", ["episode_id", "final_request", "editor"],
          [["test-ambiguous-1", "DROP", "SR"], ["test-ambiguous-2", "DROP", ""], ["test-ambiguous-3", "Show x", "SR"]])
    write(tmp_path / "train_audit.csv", ["episode_id", "verdict (keep / drop)", "auditor"],
          [["train-clear-1", "drop", "AB"], ["train-clear-2", "drop", ""], ["train-clear-3", "keep", "AB"]])
    assert edit_drops("test", tmp_path) == {"test-ambiguous-1"}
    assert audit_drops(tmp_path) == {"train-clear-1"} == dropped("train", tmp_path)


def test_a_resolved_decision_that_differs_from_the_intended_one_drops_the_episode(tmp_path):
    write(tmp_path / "test_disagreements.csv", ["shown_id", "episode_id", "A", "B", "intended", "request",
                                                "resolved_decision"],
          [["s1", "test-ambiguous-1", "clarify", "execute", "clarify", "r", "clarify"],   # intended confirmed
           ["s2", "test-ambiguous-2", "execute", "execute", "clarify", "r", "execute"],   # not really vague
           ["s3", "test-clear-3", "clarify", "execute", "execute", "r", "drop"],
           ["s4", "test-clear-4", "clarify", "execute", "execute", "r", ""]])             # not resolved yet
    assert label_drops("test", tmp_path) == {"test-ambiguous-2", "test-clear-3"} == dropped("test", tmp_path)


def load_sheets_module():
    spec = importlib.util.spec_from_file_location("mrs", REPO_ROOT / "scripts" / "make_review_sheets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not (EPISODE_DIR / "train.jsonl").exists(), reason="episodes not built")
def test_the_audit_covers_every_rule_written_episode_and_samples_the_rest():
    from clarifysql.episodes.schema import load_episodes

    mod = load_sheets_module()
    episodes = [ep for ep in load_episodes("train") if ep.id not in dropped("train")]
    picked = mod.audit_selection(episodes)
    ids = {ep.id for ep in picked}
    rule_written = {ep.id for ep in episodes if ep.draft["method"] in mod.AUDIT_ALL and ep.kind != "clear"}
    assert rule_written <= ids
    for (kind, method), n in mod.AUDIT_SAMPLE.items():
        stratum = [ep for ep in episodes if (ep.kind, ep.draft["method"]) == (kind, method)]
        assert sum(ep.id in ids for ep in stratum) == min(n, len(stratum))
    assert [ep.id for ep in mod.audit_selection(episodes)] == [ep.id for ep in picked]  # same sample every time
