"""An experiment may finish with negative quality; missing execution stays visible."""

import importlib.util
from pathlib import Path

import pytest


def subject():
    path = Path(__file__).parents[1] / "scripts/tools/summarize_final_acceptance.py"
    spec = importlib.util.spec_from_file_location("acceptance_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_negative_quality_is_retained_as_completed_experiment():
    summary = subject().validated_outcomes(
        {"status": "completed"},
        {
            "num_trials": 2,
            "completed": 1,
            "quality_passed": False,
            "episodes": [{"case": 0, "completed": True}, {"case": 1, "completed": False}],
        },
    )
    assert summary["completed"] == 1
    assert summary["quality_passed"] is False


def test_native_fallback_is_never_counted_as_functional_execution():
    summary = subject().validated_outcomes(
        {"status": "completed", "engineer_passed": True},
        {
            "num_trials": 1,
            "completed": 1,
            "quality_passed": True,
            "episodes": [{"case": 0, "completed": True}],
            "diagnostics": [{"case": 0, "commands": 0, "trajectories": 0}],
        },
        native=True,
    )
    assert summary["execution_passed"] is False
    assert summary["quality_passed"] is False


@pytest.mark.parametrize("change", ["missing_episode", "count_mismatch", "unfinished"])
def test_summary_rejects_incomplete_or_inconsistent_evidence(change):
    result = {"status": "completed"}
    report = {
        "num_trials": 1,
        "completed": 1,
        "quality_passed": True,
        "episodes": [{"case": 0, "completed": True}],
    }
    if change == "missing_episode":
        report["episodes"] = []
    elif change == "count_mismatch":
        report["completed"] = 0
    else:
        result["status"] = "running"
    with pytest.raises(ValueError):
        subject().validated_outcomes(result, report)


def test_paper_navigation_subset_counts_every_failure_once():
    result = {"status": "completed"}
    report = {
        "num_trials": 2,
        "arrived": 0,
        "collision": 1,
        "timeout": 1,
        "episodes": [
            {"scene_id": "S01", "outcome": "collision"},
            {"scene_id": "D01", "outcome": "timeout"},
        ],
    }
    one = subject().validated_outcomes(result, report, scene_prefix="S")
    assert one["num_trials"] == 1 and one["collision"] == 1 and one["arrived"] == 0


def test_pointcloud_control_uses_canonical_completed_flag_with_numeric_outcome():
    result = {"status": "completed"}
    report = {
        "num_trials": 1,
        "completed": 1,
        "quality_passed": True,
        "episodes": [{"case": 0, "completed": True, "outcome": 1}],
    }
    assert subject().validated_outcomes(result, report)["completed"] == 1
