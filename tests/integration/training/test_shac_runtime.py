"""SHAC updates and exact resume on the real tracking environment."""

import tempfile
from pathlib import Path

import jax
import numpy as np

from drone_playground.environments.tasks.tracking.rigid_body import TrackingEnv
from drone_playground.learning.algorithms import shac


def training_config():
    return {
        "task": "figure8",
        "dynamics": "so_rpy",
        "drone": "cf2x_L250",
        "policy_updates": 2,
        "num_envs": 2,
        "horizon_length": 4,
        "num_evals": 3,
        "hidden_sizes": [16, 16],
        "learning_rate": 0.001,
        "critic_learning_rate": 0.001,
        "critic_updates": 2,
        "normalize_observations": False,
        "seed": 5,
    }


def test_real_actor_critic_updates_and_snapshot_budget():
    env = TrackingEnv(device="cpu")
    snapshots = []
    try:
        make_policy, params, result = shac.train(
            env,
            training_config(),
            policy_params_fn=lambda step, maker, p: snapshots.append(step),
        )
        assert result["actual_steps"] == 16
        assert result["critic_parameter_delta_l2"] > 0
        assert result["actor_parameter_delta_l2"] > 0
        assert snapshots == [0, 8, 16]
        action, _ = make_policy(params, deterministic=True)(
            env.reset(jax.random.PRNGKey(4)).obs, jax.random.PRNGKey(0)
        )
        assert action.shape == (4,)
        assert np.isfinite(action).all()
    finally:
        env.close()


def test_full_state_resume_matches_continuous_updates():
    env = TrackingEnv(device="cpu")
    try:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, continuous, _ = shac.train(
                env, training_config(), state_directory=directory
            )
            saved = directory / "update-0000001.pkl"
            loaded, metadata = shac.load_training_state(saved)
            assert int(loaded.updates) == 1
            assert metadata["typed_key_paths"]
            _, resumed, result = shac.train(
                env, training_config(), restore_state=saved
            )
            assert result["actual_steps"] == 16
            for expected, actual in zip(
                jax.tree.leaves(continuous), jax.tree.leaves(resumed), strict=True
            ):
                np.testing.assert_array_equal(actual, expected)
    finally:
        env.close()
