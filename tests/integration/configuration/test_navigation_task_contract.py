"""Task parameters may vary; a named benchmark still locks its own conditions."""

import copy

import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.composition import build_environment, validate_config
from tests.helpers.configs import differentiable_pointcloud_config as configuration


def custom_navigation_config():
    cfg = copy.deepcopy(configuration())
    cfg["mode"] = "eval"
    cfg["runtime"]["device"] = "cpu"
    cfg["evaluation"]["protocol"] = None
    cfg["evaluation"]["benchmark_id"] = None
    cfg["env"]["task"].update(
        freq=20, physics_freq=500, duration=1.0,
        goal_radius=0.3, body_radius=0.09, max_speed=12.0,
    )
    cfg["env"]["sensor"]["source_rate_hz"] = 20
    cfg["runtime"]["action_delay_ms"] = [25.0, 40.0]
    return cfg


@pytest.mark.parametrize("physics_freq", [500, 1000])
def test_custom_navigation_parameters_drive_the_real_task_and_clock(physics_freq):
    cfg = custom_navigation_config()
    cfg["env"]["task"]["physics_freq"] = physics_freq
    env = build_environment(cfg, "cpu", "eval", 1)
    try:
        assert env.duration == 1.0
        assert env.goal_radius == env.task_definition.goal_radius == 0.3
        assert env.body_radius == 0.09
        assert env.substeps == physics_freq // 20
        assert f"{physics_freq}Hz" in env.physics_engine
        bank = env.select_bank(jnp.array([0]))
        bank = bank.replace(active=jnp.zeros_like(bank.active))
        state = env.initial_state(bank)
        command = jnp.zeros((1, 3))
        result, timestamp, outcome, _ = env.advance_checked(
            bank, state, command, command, jnp.array([0]),
            jnp.zeros(1), jnp.zeros(1, dtype=jnp.int32),
        )
        np.testing.assert_allclose(timestamp, [0.05], atol=1e-7)
        assert np.isfinite(result.vector()).all()
        np.testing.assert_array_equal(outcome, [0])
    finally:
        env.close()


def test_named_navigation_benchmark_still_rejects_changed_task_conditions():
    cfg = configuration()
    cfg["mode"] = "eval"
    cfg["env"]["task"]["duration"] = 1.0
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("field,value", [("goal_radius", 0.0), ("body_radius", 0.0), ("max_speed", 1.0)])
def test_custom_task_still_rejects_invalid_geometry_or_command_bounds(field, value):
    cfg = custom_navigation_config()
    cfg["env"]["task"][field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)
