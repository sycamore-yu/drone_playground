"""Check that continuation diagnostics reuse only matching, completed source evidence."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "ppo_branch_diagnostic.py"


@pytest.mark.parametrize("change", [None, "checkpoint", "reward", "horizon", "incomplete", "error"])
def test_source_evidence_reuse(tmp_path, change):
    """Optimizer batching may change; rollout inputs and prior checks must remain valid."""
    spec = importlib.util.spec_from_file_location("ppo_branch_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    original = {
        "learning": {"options": {"horizon": 32, "minibatches": 4, "gae_lambda": 0.95}},
        "task": {"reward": "original"},
    }
    branch = copy.deepcopy(original)
    branch["learning"]["options"]["horizon"] = 512
    identity = {
        "status": "completed",
        "checkpoint": str(source),
        "source_config": original,
        "branch_config": branch,
        "restored_state_sha256": "previously-verified-state",
    }
    current = copy.deepcopy(branch)
    current["learning"]["options"].update(minibatches=1, gae_lambda=1.0)
    checks = {
        "reward_finite": 1,
        "log_probability_max_error": 0,
        "recurrent_memory_max_error": 0,
    }
    if change == "checkpoint":
        source = tmp_path / "different-source"
    elif change == "reward":
        current["task"]["reward"] = "different"
    elif change == "horizon":
        current["learning"]["options"]["horizon"] = 128
    elif change == "incomplete":
        identity["status"] = "running"
    elif change == "error":
        checks["log_probability_max_error"] = 0.1
    (tmp_path / "experiment.json").write_text(json.dumps(identity))
    (tmp_path / "rollout-checks.json").write_text(json.dumps(checks))
    if change:
        with pytest.raises(ValueError, match="does not match"):
            module.reuse_source_checks(tmp_path, source, original, current)
    else:
        state_id, reused = module.reuse_source_checks(tmp_path, source, original, current)
        assert state_id == "previously-verified-state"
        assert reused == checks
