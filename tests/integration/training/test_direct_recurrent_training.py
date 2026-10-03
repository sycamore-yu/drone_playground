"""Actual recurrent updates use composed tasks and restore complete learner state."""

import jax
import numpy as np
import optax
import pytest

from drone_playground.artifacts.training_state import load_training_state, save_training_state
from drone_playground.configuration import load_config
from drone_playground.environments.environment import build_environment
from drone_playground.learning.algorithms.recurrent_bptt import initialize, make_update
from drone_playground.learning.algorithms.recurrent_navigation_bptt import rollout_loss
from drone_playground.learning.algorithms.recurrent_tracking_bptt import loss_function


@pytest.mark.parametrize(
    "recipe",
    [
        "papers/differentiable_pointcloud",
        "navigation/differentiable_pointcloud",
        "papers/depth_diffphysics",
        "control/differentiable_pointcloud_hovering",
    ],
)
def test_real_recurrent_update_and_restore(recipe, tmp_path):
    overrides = [
        "experiment=" + recipe,
        "runtime.device=cpu",
        "training.num_envs=1",
        "algorithm.horizon_length=2",
        "network.hidden_size=8",
    ]
    if recipe != "papers/depth_diffphysics":
        overrides += [
            "env.sensor.azimuth_count=8",
            "env.sensor.elevation_count=3",
            "network.encoder.point_channels=[8,8]",
        ]
    if recipe.startswith("navigation/") or recipe == "papers/depth_diffphysics":
        overrides += ["training.scene_distribution.scene.per_difficulty=2"]
    config = load_config(overrides=overrides)
    env = build_environment(config, role="train", count=1)
    try:
        task = env.task
        state, network, optimizer = initialize(task, config)
        if recipe == "papers/differentiable_pointcloud":
            update = make_update(task, network, optimizer, config)
            after, metrics = update(state)
        else:

            def loss(parameters):
                if recipe.startswith("control/"):
                    return loss_function(task, network, parameters, state.key, 1, 2)
                return rollout_loss(
                    task,
                    network,
                    parameters,
                    state.key,
                    1,
                    2,
                    config["algorithm"].get("velocity_prediction_weight", 0.0),
                )

            (_, metrics), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(state.params)
            delta, optimizer_state = optimizer.update(gradient, state.opt_state, state.params)
            after = state.replace(
                params=optax.apply_updates(state.params, delta),
                opt_state=optimizer_state,
                updates=state.updates + 1,
            )
        assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(after.params))
        )
        path = tmp_path / "training-state.pkl"
        save_training_state(path, after, config)
        restored, metadata = load_training_state(path)
        for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(after)):
            np.testing.assert_array_equal(actual, expected)
        assert metadata["config"]["algorithm"]["name"] == "bptt"
    finally:
        env.close()
