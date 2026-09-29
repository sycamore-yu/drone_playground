"""PointNet/GRU control transfer retains actual point input and actuator delay."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.composition import compose_method, validate_config


@pytest.mark.parametrize("task", ["hovering", "tracking", "racing"])
def test_pointcloud_control_recipe_has_explicit_transfer_identity(task):
    cfg = compose_method("paper/pointcloud_flight", "paper/control/" + task)
    validate_config(cfg)
    assert cfg["env"]["task"]["control_task"] == task
    assert cfg["env"]["task"]["name"] == "pointcloud_control"
    assert cfg["objective"]["name"] == "pointcloud_control"
    assert "position_loss" not in cfg["env"]["task"]
    assert cfg["runtime"]["action_delay_ms"] == [25.0, 50.0]
    assert cfg["network"]["hidden_size"] == 192


def test_pointcloud_control_delay_is_causal_and_differentiable():
    from drone_playground.environments.tasks.pointcloud_control import delayed_step
    from drone_playground.models.point_mass import PointMassLag, PointMassState

    model = PointMassLag(backward="direct")
    state = PointMassState.create(jnp.array([[0.0, 0.0, 1.0]]))
    command = jnp.array([[1.0, 0.0, 0.0]])
    final, positions = delayed_step(
        model, state, command, jnp.zeros_like(command), jnp.array([15]), 0.002, 50
    )
    np.testing.assert_allclose(positions[:15, :, 0], 0.0, atol=0)
    assert float(final.pos[0, 0]) > 0
    gradient = jax.grad(
        lambda a: delayed_step(
            model, state, command * a, jnp.zeros_like(command), jnp.array([15]), 0.002, 50
        )[0].pos[0, 0]
    )(1.0)
    assert np.isfinite(gradient) and gradient > 0


def test_pointcloud_control_objective_uses_qualified_task_and_full_encoder():
    from drone_playground.composition import build_environment
    from drone_playground.learning.algorithms.pointcloud_bptt import initialize
    from drone_playground.learning.algorithms.pointcloud_control import loss_function

    cfg = compose_method(
        "paper/pointcloud_flight",
        "paper/control/hovering",
        [
            "training.num_envs=2",
            "training.policy_updates=2",
            "algorithm.horizon_length=2",
            "env.sensor.azimuth_count=6",
            "env.sensor.elevation_count=2",
        ],
    )
    task = build_environment(cfg, "cpu")
    state, network, _ = initialize(task, cfg)
    loss, gradient = jax.value_and_grad(
        lambda p: loss_function(task, network, p, jax.random.PRNGKey(0), 2, 2)[0]
    )(state.params)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient))
    assert sum(float(jnp.sum(x * x)) for x in jax.tree.leaves(gradient)) > 0
