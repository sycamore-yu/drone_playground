"""Frozen policy identity includes input meaning and explicitly selected groups."""

import json

import pytest

from drone_playground.app import resolve_checkpoint_execution
from drone_playground.composition import compose_experiment, validate_config


def metadata(tmp_path, config):
    path = tmp_path / "frozen.pkl"
    path.with_suffix(".json").write_text(json.dumps({"config": config}))
    return str(path)


def test_method_name_conflict_is_rejected_even_when_both_are_neural(tmp_path):
    saved = compose_experiment("control/ppo", "tracking")
    requested = compose_experiment("control/apg", "tracking")
    requested.update(mode="eval", checkpoint=metadata(tmp_path, saved))
    with pytest.raises(ValueError, match=r"identity"):
        resolve_checkpoint_execution(requested, ["experiment=control/apg"])


def test_mounted_sensor_override_is_seen_and_input_semantics_are_checked(tmp_path):
    saved = compose_experiment("control/ppo", "navigation/static")
    overrides = ["sensor@env.sensor=d435", "env.task.observation.name=navigation_depth"]
    requested = compose_experiment("control/ppo", "navigation/static", overrides)
    requested.update(mode="eval", checkpoint=metadata(tmp_path, saved))
    with pytest.raises(ValueError, match=r"input|sensor|observation"):
        resolve_checkpoint_execution(requested, overrides)


def test_external_worker_cannot_be_declared_as_a_jax_runtime():
    cfg = compose_experiment("papers/super", "navigation/static")
    cfg["runtime"]["backend"] = "jax"
    with pytest.raises(ValueError, match=r"backend"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "override",
    ["env.dynamics.forward=so_rpy", "dynamics@env.dynamics=crazyflow_so_rpy"],
)
def test_single_execution_override_keeps_the_checkpoint_task(tmp_path, override):
    saved = compose_experiment("control/ppo", "racing")
    requested = compose_experiment("control/ppo", overrides=[override])
    requested.update(mode="eval", checkpoint=metadata(tmp_path, saved))
    resolved = resolve_checkpoint_execution(requested, [override])
    assert resolved["env"]["name"] == "racing"
    assert resolved["env"]["task"] == saved["env"]["task"]
    assert resolved["env"]["task"]["observation"] == saved["env"]["task"]["observation"]
    assert resolved["env"]["dynamics"]["forward"] == "so_rpy"
    if override.startswith("env."):
        assert resolved["env"]["dynamics"]["drone"] == "cf21B_500"
    validate_config(resolved)


def test_device_override_keeps_saved_command_delay(tmp_path):
    saved = compose_experiment("control/ppo", "hovering", ["runtime.action_delay_steps=2"])
    requested = compose_experiment("control/ppo", overrides=["runtime.device=cpu"])
    requested.update(mode="eval", checkpoint=metadata(tmp_path, saved))
    resolved = resolve_checkpoint_execution(requested, ["runtime.device=cpu"])
    assert resolved["runtime"]["action_delay_steps"] == 2
    assert resolved["runtime"]["device"] == "cpu"


def test_explicit_environment_replacement_uses_the_selected_task(tmp_path):
    saved = compose_experiment("control/ppo", "navigation/static")
    requested = compose_experiment("control/ppo", "navigation/dynamic")
    requested.update(mode="eval", checkpoint=metadata(tmp_path, saved))
    resolved = resolve_checkpoint_execution(requested, ["env=navigation/dynamic"])
    assert resolved["env"]["scene"]["dynamic"] is True
    assert resolved["method"] == saved["method"]
    validate_config(resolved)
