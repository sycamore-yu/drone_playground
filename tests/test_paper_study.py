"""Bounded study selection and evidence requirements without running trainers."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def study():
    """Load the scheduler without scheduling or touching GPU state."""
    path = Path(__file__).parents[1] / "tools/optimize_paper_ppo.py"
    assert path.exists(), "Paper PPO measured study is missing"
    spec = importlib.util.spec_from_file_location("paper_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_and_incomplete_reports_are_not_objective_values(tmp_path, study):
    """A missing full evaluation is unknown, not an observed zero or improvement."""
    p = tmp_path / "checkpoints/model_000080/probe"
    p.mkdir(parents=True)
    (p / "report.json").write_text(
        json.dumps(
            {
                "complete": True,
                "reports": {
                    "S01": {"episodes": 8, "successes": 8},
                    "D01": {"episodes": 8, "successes": 8},
                },
            }
        )
    )
    assert study.full_screen_evidence(tmp_path) is None
    p = tmp_path / "checkpoints/model_000160/screening"
    p.mkdir(parents=True)
    (p / "report.json").write_text(json.dumps({"complete": False, "reports": {}}))
    assert study.full_screen_evidence(tmp_path) is None
    rows = {scene: {"episodes": 8, "successes": 4} for scene in study.SCENES}
    (p / "report.json").write_text(json.dumps({"complete": True, "reports": rows}))
    evidence = study.full_screen_evidence(tmp_path)
    assert evidence["value"] == 0.625
    assert evidence["successes"] == 24


def test_repeat_accepted_seed_before_opening_new_candidate(study):
    """Spend follow-up seeds on real frozen benchmark successes, not reward heuristics."""
    records = {
        f"{case}_s{seed}": {"status": "missing"} for case in study.CASES for seed in range(3)
    }
    records["fan_equations_s0"] = {"status": "running"}
    records["visfly_source_s0"] = {"status": "accepted"}
    assert study.next_trial(records) == ("visfly_source", 1)


def test_negative_baseline_permits_one_informed_candidate(study):
    """Do not relaunch old baselines or spend seeds 1/2 on failed trials."""
    records = {
        f"{case}_s{seed}": {"status": "missing"} for case in study.CASES for seed in range(3)
    }
    records["fan_equations_s0"] = {"status": "early_stopped"}
    records["visfly_source_s0"] = {"status": "running"}
    assert study.next_trial(records) == ("fan_margin2", 0)
    records["fan_margin2_s0"] = {"status": "early_stopped"}
    assert study.next_trial(records) is None


def test_failed_second_seed_does_not_count_as_stable_acceptance(study):
    """One passing seed cannot prevent a bounded alternative after reproduction fails."""
    records = {
        f"{case}_s{seed}": {"status": "missing"} for case in study.CASES for seed in range(3)
    }
    records["fan_equations_s0"] = {"status": "accepted"}
    records["fan_equations_s1"] = {"status": "early_stopped"}
    records["visfly_source_s0"] = {"status": "running"}
    assert study.next_trial(records) == ("fan_margin2", 0)
