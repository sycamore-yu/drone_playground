"""Frozen recurrent evaluation keeps the explicitly requested execution conditions."""

import copy
from types import SimpleNamespace

import pytest

from drone_playground.configuration import load_config


def test_frozen_navigation_uses_requested_scene_and_runtime(monkeypatch, tmp_path):
    """Selecting dynamic scenes must not silently restore all eight training scenes."""
    from drone_playground.artifacts import training_state
    from drone_playground.environments import factory as environment
    from drone_playground.evaluation.navigation import recurrent

    trained = load_config(overrides=["experiment=papers/depth_diffphysics"])
    requested = copy.deepcopy(trained)
    requested["checkpoint"] = str(tmp_path / "checkpoint.pkl")
    requested["env"]["scene"]["scene_ids"] = ["D01", "D02", "D03", "D06"]
    requested["runtime"]["output_root"] = str(tmp_path)
    monkeypatch.setattr(
        training_state,
        "load_training_state",
        lambda _: (SimpleNamespace(params={}), {"config": trained}),
    )

    class ConfigurationCheckedError(Exception):
        """Stop after verifying the actual environment-construction boundary."""

    def check_environment(config, device, role):
        assert config["env"] == requested["env"]
        assert config["runtime"] == requested["runtime"]
        assert config["network"] == trained["network"]
        raise ConfigurationCheckedError

    monkeypatch.setattr(environment, "build_environment", check_environment)
    with pytest.raises(ConfigurationCheckedError):
        recurrent.evaluate(requested, tmp_path, "frozen-dynamic")
