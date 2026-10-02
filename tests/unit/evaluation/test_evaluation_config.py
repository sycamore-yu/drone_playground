"""Frozen-policy evaluation configuration selection."""

from copy import deepcopy

from drone_playground.evaluation.run import resolve_evaluation_config


def config_fixture():
    return {
        "mode": "train",
        "evaluation": {"environment": "checkpoint", "role": "eval"},
        "runtime": {"device": "gpu"},
        "method": {"name": "saved", "output": "attitude_thrust"},
        "network": {"name": "saved-network"},
        "algorithm": {"name": "ppo"},
        "env": {"dynamics": {"forward": "lotf_high_fidelity"}},
    }


def test_checkpoint_environment_is_the_default_execution_config():
    saved = config_fixture()
    requested = deepcopy(saved)
    requested["mode"] = "eval"
    requested["runtime"]["device"] = "cpu"
    result = resolve_evaluation_config(requested, {"config": saved})
    assert result["env"]["dynamics"]["forward"] == "lotf_high_fidelity"
    assert result["runtime"]["device"] == "cpu"
    assert result["mode"] == "eval"


def test_explicit_execution_environment_keeps_frozen_policy_identity():
    saved = config_fixture()
    requested = deepcopy(saved)
    requested["mode"] = "eval"
    requested["evaluation"]["environment"] = "config"
    requested["runtime"]["device"] = "cpu"
    requested["env"]["dynamics"]["forward"] = "first_principles"
    requested["network"] = {"name": "requested-network"}
    requested["algorithm"] = {"name": "apg"}
    requested["method"] = {"name": "requested", "output": "attitude_thrust"}

    result = resolve_evaluation_config(requested, {"config": saved})
    assert result["env"]["dynamics"]["forward"] == "first_principles"
    assert result["network"] == saved["network"]
    assert result["algorithm"] == saved["algorithm"]
    assert result["method"] == saved["method"]
