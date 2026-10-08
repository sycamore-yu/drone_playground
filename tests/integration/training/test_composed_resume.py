"""Resume a complete composed learner without changing the subsequent updates."""

import importlib

import jax
import numpy as np
import pytest

from drone_playground.configuration import compose_experiment
from drone_playground.environments.factory import build_environment
from drone_playground.learning.brax_configuration import native_training_config


@pytest.mark.parametrize("algorithm", ["bptt", "shac"])
def test_composed_complete_state_restores_identical_next_update(algorithm, tmp_path):
    choices = [
        "runtime.device=cpu",
        "runtime.action_delay_ms=null",
        "training.num_envs=1",
        "training.policy_updates=2",
        "algorithm.horizon_length=2",
        "training.num_evals=3",
        "env.task.duration=0.08",
        "network.hidden_sizes=[8]",
        "network.normalize_observations=false",
    ]
    if algorithm == "shac":
        choices.append("algorithm.critic_updates=1")
    config = compose_experiment("control/" + algorithm, "hovering", choices)
    env = build_environment(config, "cpu", role="train")
    try:
        module = importlib.import_module("drone_playground.learning.algorithms." + algorithm)
        native = native_training_config(config)
        _, expected, _ = module.train(env, native, state_directory=tmp_path)
        _, actual, result = module.train(env, native, restore_state=tmp_path / "update-0000001.pkl")
        assert result["actual_steps"] == 4
        for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            np.testing.assert_array_equal(before, after)
    finally:
        env.close()
