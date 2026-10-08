"""The trained navigation adapter is distinct from the paper's zero-shot protocol."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.configs import differentiable_pointcloud_config as configuration


def test_new_recipe_preserves_sensor_and_protocol_and_old_training_guard():
    from drone_playground.configuration import compose_experiment, validate_config

    cfg = configuration()
    validate_config(cfg)
    assert cfg["env"]["task"]["duration"] == 300
    assert cfg["env"]["task"]["goal_radius"] == 0.5
    assert cfg["runtime"]["action_delay_ms"] == [25.0, 50.0]
    assert cfg["network"]["encoder"]["point_channels"] == [64, 128, 1024]
    assert cfg["env"]["task"]["max_speed"] == 20.0
    assert cfg["method"]["name"] == "differentiable_pointcloud"
    source = compose_experiment("papers/differentiable_pointcloud")
    assert source["env"]["scene"]["name"] != "navigation"
    assert cfg["training"]["scene_distribution"]["scene"]["name"] == "navigation_training"


def test_training_initialization_covers_course_and_goal_without_obstacle_penetration():
    from drone_playground.environments.factory import build_environment

    config = configuration()
    config["training"]["scene_distribution"] = {"type": "fixed", "scene": None}
    task = build_environment(config, "cpu", role="train").task
    initial = jax.jit(lambda key: task.training_initial(key, 32))(jax.random.PRNGKey(17))
    bank, physical, clocks, speeds, ticks = initial
    clearances = task.clearance(bank, physical, clocks)
    assert np.isfinite(physical.vector()).all()
    assert np.min(clearances) >= 0.15
    assert np.any(np.asarray(physical.pos[:, 0]) < 3)
    assert np.any(np.asarray(physical.pos[:, 0]) > 95)
    assert len(np.unique(np.asarray(bank.subtype))) >= 6
    assert np.all((np.asarray(ticks) >= 13) & (np.asarray(ticks) <= 25))
    assert np.all((np.asarray(speeds) >= 2) & (np.asarray(speeds) <= 4))
    observation_bank = task.select_bank(jnp.array([0, 1]))
    points, valid, proprio, _ = task.measure(
        observation_bank, task.initial_state(observation_bank), jnp.zeros(2), jnp.full(2, 4.0)
    )
    assert points.shape == (2, 5400, 3)
    assert valid.shape == (2, 5400)
    assert proprio.shape == (2, 10)


def test_near_goal_and_penetration_have_actual_differentiable_training_signal():
    from drone_playground.dynamics.point_mass import PointMassState
    from drone_playground.learning.objectives.navigation_acceleration import (
        PointCloudNavigationObjective,
    )

    loss = PointCloudNavigationObjective()
    goal = jnp.array([[98.0, 0.0, 3.0]])
    previous = PointMassState.create(jnp.array([[96.8, 0.0, 3.0]]))

    def value(position, clearance):
        current = previous.replace(pos=position)
        scalar, _ = loss(
            previous,
            current,
            goal,
            jnp.array([2.0]),
            clearance,
            jnp.zeros((1, 3)),
            jnp.zeros((1, 3)),
            0.1,
        )
        return scalar

    gradient = jax.grad(lambda p: value(p, jnp.array([2.0])))(previous.pos)
    assert gradient[0, 0] < 0, "Gradient descent must move a stalled policy toward the goal"
    for distance in [-0.2, 0.2, 1.0]:
        derivative = jax.grad(lambda d: value(previous.pos, jnp.array([d])))(jnp.float32(distance))
        assert derivative < 0 and np.isfinite(derivative)


def test_checked_delay_matches_training_physics_and_preserves_terminal_event():
    from drone_playground.control.delay import delayed_step
    from drone_playground.control.transition import delayed_schedule
    from drone_playground.environments.factory import build_environment

    task = build_environment(configuration(), "cpu", role="eval").task
    bank = task.select_bank(jnp.array([0]))
    state = task.initial_state(bank)
    current = jnp.array([[1.0, 0.0, 0.0]])
    previous = jnp.zeros_like(current)
    ticks = jnp.array([20])
    expected, _ = delayed_step(task.dynamics, state, current, previous, ticks, 0.002, 50)
    schedule = delayed_schedule(current, previous, ticks, task.substeps)
    actual, time, outcome, _, _ = task.transition.checked(
        state,
        schedule,
        task.transition_events(bank),
        timestamp=jnp.zeros(1),
        outcome=jnp.zeros(1, jnp.int32),
    )
    np.testing.assert_allclose(actual.vector(), expected.vector(), atol=1e-6)
    np.testing.assert_allclose(time, 0.1, atol=1e-7)
    np.testing.assert_array_equal(outcome, 0)
    stopped, clock, result, _, _ = task.transition.checked(
        state,
        schedule,
        task.transition_events(bank),
        timestamp=jnp.array([3.0]),
        outcome=jnp.array([1], jnp.int32),
    )
    np.testing.assert_array_equal(stopped.vector(), state.vector())
    np.testing.assert_array_equal(clock, [3.0])
    np.testing.assert_array_equal(result, [1])


def test_evaluator_accepts_runtime_delay_grid_and_speed_without_recompiling():
    from drone_playground.environments.factory import build_environment
    from drone_playground.evaluation.navigation.recurrent import RecurrentNavigationEvaluator

    class ConstantPolicy:
        hidden_size = 1

        def apply(self, params, points, valid, proprio, memory):
            del params, points, valid
            action = jnp.broadcast_to(
                jnp.array([1.0, 0.0, 0.0], dtype=proprio.dtype),
                (proprio.shape[0], 3),
            )
            return action, memory

    task = build_environment(configuration(), "cpu", role="eval").task
    task.bank = task.select_bank(jnp.array([0]))
    task.manifest = {**task.manifest, "scene_ids": ["S01"]}
    task.duration = 3 * task.dt
    network = ConstantPolicy()
    params = {}
    evaluator = RecurrentNavigationEvaluator(task, network, 22000, 1, 4.0)
    a, first = evaluator.run(params, commanded_speed=2.0, delay_ticks=[13])
    b, second = evaluator.run(params, commanded_speed=3.0, delay_ticks=[25])
    assert a["delay_ticks"] == [13] and b["delay_ticks"] == [25]
    assert a["episodes"][0]["command_speed_m_s"] == 2.0
    assert b["episodes"][0]["command_speed_m_s"] == 3.0
    assert float(first["pos"][-1, 0, 0]) > float(second["pos"][-1, 0, 0])
    assert evaluator._run._cache_size() == 1
    for kwargs in ({"delay_ticks": [0]}, {"delay_ticks": [13, 14]}, {"commanded_speed": 21.0}):
        with pytest.raises(ValueError):
            evaluator.run(params, **kwargs)


def test_training_point_jitter_preserves_invalid_points_and_is_seeded():
    from drone_playground.environments.randomization import point_measurement_noise

    def augment_point_measurements(points, valid, key, std):
        return point_measurement_noise(points, valid, key, std)[0]

    points = jnp.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [5.0, 4.0, 3.0]]])
    valid = jnp.array([[True, False, True]])
    key = jax.random.PRNGKey(7)
    np.testing.assert_array_equal(augment_point_measurements(points, valid, key, 0.0), points)
    a = augment_point_measurements(points, valid, key, 0.002)
    np.testing.assert_array_equal(a, augment_point_measurements(points, valid, key, 0.002))
    np.testing.assert_array_equal(a[:, 1], points[:, 1])
    assert not np.array_equal(a, points)
    np.testing.assert_array_equal(
        jax.grad(lambda x: augment_point_measurements(x, valid, key, 0.002).sum())(points),
        np.broadcast_to(np.asarray(valid)[..., None], points.shape).astype(float),
    )
