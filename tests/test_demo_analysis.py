"""Replay the real received demo and catch a query that wins only by a seed-data tie."""

import hashlib
import json
import zipfile

import pytest

from clarifysql.config import REPO_ROOT
from clarifysql.demo.checks import demo_cases
from clarifysql.demo.university import create_database
from scripts.analyze_demo import placement_variants, read_bundle, verify_bundle

ARCHIVE = REPO_ROOT / "results/demo/checkpoint8/received.zip"


def test_placement_variants_distinguish_counting_all_rows_from_placement_rate(tmp_path):
    db = create_database(tmp_path / "demo.sqlite")
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    reference = next(c.expected_sql for c in demo_cases() if c.id == "demo-ranking-placements")
    wrong = "SELECT d.name FROM departments d JOIN students s ON d.id=s.department_id " \
            "JOIN placements p ON p.student_id=s.id GROUP BY d.id ORDER BY COUNT(p.student_id)*1.0/COUNT(s.id) DESC LIMIT 1"
    # A tie may match one winner by chance; it cannot correctly switch between both variants.
    assert not all(r["matches"] for r in placement_variants(db, wrong, reference, tmp_path / "wrong"))
    assert all(r["matches"] for r in placement_variants(db, reference, reference, tmp_path / "correct"))
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


@pytest.mark.skipif(not ARCHIVE.exists(), reason="Received checkpoint-8 archive is not included in this source-only bundle")
def test_received_model_run_reproduces_without_turning_a_false_positive_into_a_success(tmp_path):
    before = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    verification, extra = verify_bundle(ARCHIVE, tmp_path / "analysis")
    assert verification["original_cases_reproduced"] == 7 and verification["original_passed"] == 5
    assert extra["original_result_match"] is True and extra["all_variants_match"] is False
    assert verification["interface"]["constructed"] is True
    assert verification["interface"]["browser_interaction_checked"] is False
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == before


def test_archive_reader_rejects_unsafe_paths(tmp_path):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../outside.json", "{}")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        read_bundle(path)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="Received checkpoint-8 archive unavailable")
def test_rejects_a_summary_that_claims_every_original_case_passed(tmp_path):
    data = read_bundle(ARCHIVE)
    summary = json.loads(data["checks/summary.json"])
    summary["passed"] = 7
    data["checks/summary.json"] = json.dumps(summary).encode()
    path = tmp_path / "incorrect-summary.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, value in data.items():
            z.writestr(name, value)
    with pytest.raises(ValueError, match="Saved summary does not match"):
        verify_bundle(path, tmp_path / "analysis")
