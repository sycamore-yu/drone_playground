"""The public entry resolves checkpoints once; evaluators consume resolved settings."""

import json

from drone_playground.configuration import compose_experiment, resolve_checkpoint_execution


def checkpoint(tmp_path):
    config = compose_experiment("control/ppo")
    config["env"]["dynamics"]["forward"] = "lotf_high_fidelity"
    path = tmp_path / "policy.pkl"
    path.with_suffix(".json").write_text(json.dumps({"config": config}))
    requested = compose_experiment("control/ppo")
    requested.update(mode="eval", checkpoint=str(path))
    requested["runtime"]["device"] = "cpu"
    return config, requested


def test_checkpoint_environment_is_the_default_execution_config(tmp_path):
    _saved, requested = checkpoint(tmp_path)
    result = resolve_checkpoint_execution(requested, ["runtime.device=cpu"])
    assert result["env"]["dynamics"]["forward"] == "lotf_high_fidelity"
    assert result["runtime"]["device"] == "cpu" and result["mode"] == "eval"


def test_explicit_execution_environment_keeps_frozen_policy_identity(tmp_path):
    saved, requested = checkpoint(tmp_path)
    requested["env"]["dynamics"]["forward"] = "first_principles"
    requested["network"] = {"name": "unselected-default"}
    result = resolve_checkpoint_execution(
        requested, ["runtime.device=cpu", "env.dynamics.forward=first_principles"]
    )
    assert result["env"]["dynamics"]["forward"] == "first_principles"
    for name in ("network", "algorithm", "method"):
        assert result[name] == saved[name]
