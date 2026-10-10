"""Guard the independent frozen audit and terminal Patience handoff."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def scheduler():
    """Load the script without running its CLI or GPU-admission loop."""
    path = Path(__file__).resolve().parents[1] / "tools/train_scratch_matrix.py"
    spec = importlib.util.spec_from_file_location("_scratch_scheduler_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write(path, contents):
    """Write an isolated synthetic evidence record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents) if not isinstance(contents, str) else contents)


def test_verified_first_checkpoint_frozen_audit_counts_without_rewriting_old_run(
    scheduler, tmp_path
):
    """Count only original scratch weights with an independent frozen pass."""
    run = tmp_path / "depth_shac_s1"
    write(run / "config.yaml", "initial_checkpoint: null\n")
    write(run / "run.json", {"status": "interrupted", "layout_version": 2})
    policy = run / "checkpoints/step-000320/policy.zip"
    write(policy, "original-network-weights")
    write(
        policy.parent / "report.json",
        {
            "mode": "checkpoint_eval",
            "complete": True,
            "passed": True,
            "seed_base": 1_000_000,
        },
    )
    audit = tmp_path / ".scheduler/frozen_audits/depth_shac_s1"
    write(
        audit / "audit.json",
        {
            "status": "accepted",
            "checkpoint": "checkpoints/step-000320/policy.zip",
        },
    )
    write(
        audit / "report.json",
        {
            "mode": "benchmark",
            "complete": True,
            "passed": True,
            "seed_base": 2_000_000,
        },
    )
    assert scheduler.trial_status(run, {}) == "accepted"
    write(
        audit / "report.json",
        {
            "mode": "checkpoint_eval",
            "complete": True,
            "passed": True,
            "seed_base": 1_000_000,
        },
    )
    assert scheduler.trial_status(run, {}) == "invalid"


def test_failed_frozen_audit_is_terminal_and_new_early_stop_is_not_resumed(scheduler, tmp_path):
    """Neither failed tests nor Patience should trigger same-run extension."""
    run = tmp_path / "lidar_apg_s0"
    write(run / "run.json", {"status": "interrupted"})
    write(
        tmp_path / ".scheduler/frozen_audits/lidar_apg_s0/audit.json",
        {"status": "failed", "checkpoint": "checkpoints/step-000480/policy.zip"},
    )
    assert scheduler.trial_status(run, {}) == "frozen_benchmark_failed"
    new = tmp_path / "lidar_ppo_s0_trial4"
    write(new / "run.json", {"status": "early_stopped"})
    assert scheduler.trial_status(new, {}) == "early_stopped"
