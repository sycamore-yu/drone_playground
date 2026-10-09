"""Public-interface physics, episode and control contract regression tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.simulation.environment import Environment
from drone_playground.simulation.methods import Setpoint, Trajectory
from drone_playground.simulation.tasks import Event, first_event


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_episode_duration_rejects_invalid_values(duration):
    """Verify episode duration rejects invalid values."""
    with pytest.raises(ValueError, match="finite and positive"):
        Environment(duration=duration)


def test_navigation_contract_records_instantiated_task_rules():
    """Verify navigation contract records instantiated task rules."""
    env = Environment(task="navigation", scene="S01")
    contract = env.task.contract
    assert contract["duration_seconds"] == 300
    assert contract["body_radius_metres"] == 0.07
    assert contract["goal_radius_metres"] == 0.5
    assert contract["bounds_metres"] == [[0, -20, 0.5], [100, 20, 6]]
    assert contract["start_metres"] == [2, 0, 3]
    assert contract["goal_metres"] == [98, 0, 3]


def test_optional_goal_observation_exposes_height_and_remaining_distance():
    """Expose normalized goal quantities while retaining the original ten features."""
    legacy = Environment(task="navigation", scene="S01")
    informed = Environment(task="navigation", scene="S01", navigation_goal_observation=True)
    state = legacy.reset(jax.random.key(7))
    physics = state.physics.replace(
        states=state.physics.states.replace(pos=jnp.array([[[97.0, 0.0, 2.0]]]))
    )
    state = state.replace(physics=physics)
    original = legacy.observe(state)["state"]
    actual = informed.observe(state)["state"]
    assert original.shape == (1, 10)
    assert actual.shape == (1, 12)
    np.testing.assert_array_equal(actual[:, :10], original)
    np.testing.assert_allclose(actual[:, -2:], [[0.1818181818, 0.0147313913]], rtol=1e-6)


def test_forward_matches_official_crazyflow():
    """Verify forward matches official crazyflow."""
    env = Environment(num_envs=3)
    state = env.reset(jax.random.key(9))
    command = jnp.array(
        [[0.10, -0.05, 0.0, 0.32], [-0.08, 0.10, 0.04, 0.35], [0.0, 0.0, 0.0, 0.33]]
    )
    actual = env.step_setpoint(state, Setpoint(command))
    env.sim.data = state.physics
    env.sim.attitude_control(command[:, None])
    env.sim.step(env.substeps)
    for actual_leaf, official_leaf in zip(
        jax.tree.leaves(actual.physics.states), jax.tree.leaves(env.sim.data.states), strict=True
    ):
        np.testing.assert_allclose(actual_leaf, official_leaf, atol=2e-6, rtol=2e-6)


def test_masked_reset_keeps_unfinished_worlds_and_clears_counters():
    """Verify masked reset keeps unfinished worlds and clears counters."""
    env = Environment(num_envs=3)
    before = env.step(env.reset(jax.random.key(3)), jnp.zeros((3, 4)))
    before = before.replace(task=before.task.replace(event=jnp.array([0, 1, 5], jnp.int32)))
    after = env.reset(jax.random.key(10), before)
    np.testing.assert_array_equal(after.physics.states.pos[0], before.physics.states.pos[0])
    np.testing.assert_array_equal(after.physics.core.steps[:, 0], [env.substeps, 0, 0])
    np.testing.assert_array_equal(after.task.event, [0, 0, 0])
    assert not np.array_equal(before.physics.states.pos[1:], after.physics.states.pos[1:])


def test_event_priority_and_termination_vs_truncation():
    """Verify event priority and termination vs truncation."""
    result = first_event(
        jnp.array([1, 0, 0, 0, 0], bool),
        jnp.array([0, 0, 1, 1, 1], bool),
        jnp.array([0, 0, 0, 1, 1], bool),
        jnp.array([1, 1, 1, 1, 0], bool),
        jnp.ones(5, bool),
    )
    np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])
    env = Environment(num_envs=2)
    state = env.reset(jax.random.key(1))
    state = state.replace(
        task=state.task.replace(event=jnp.array([Event.COLLISION, Event.TIMEOUT]))
    )
    np.testing.assert_array_equal(state.terminated, [True, False])
    np.testing.assert_array_equal(state.truncated, [False, True])


def test_real_physics_action_gradient_is_finite_and_nonzero():
    """Verify real physics action gradient is finite and nonzero."""
    env = Environment(num_envs=2)
    state = env.reset(jax.random.key(1))
    gradient = jax.grad(lambda action: jnp.sum(env.step(state, action).physics.states.pos))(
        jnp.zeros((2, 4))
    )
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1e-7


def test_upstream_acceleration_conversion_hover_and_forward():
    """Verify upstream acceleration conversion hover and forward."""
    env = Environment()
    state = env.reset(jax.random.key(1))
    state = state.replace(
        physics=state.physics.replace(
            states=state.physics.states.replace(quat=jnp.array([[[0.0, 0.0, 0.0, 1.0]]]))
        )
    )
    hover = env.controller.acceleration(state.physics, jnp.zeros((1, 3)), jnp.zeros(1))
    np.testing.assert_allclose(hover.value[0, :3], 0, atol=1e-6)
    np.testing.assert_allclose(hover.value[0, 3], state.physics.params.mass[0] * 9.81, rtol=1e-5)
    forward = env.controller.acceleration(state.physics, jnp.array([[2.0, 0.0, 0.0]]), jnp.zeros(1))
    assert forward.value[0, 1] > 0


@pytest.mark.parametrize(
    "task,scene", [("tracking", "empty"), ("racing", "racing"), ("navigation", "D01")]
)
def test_each_task_runs_and_observations_do_not_expose_geometry(task, scene):
    """Verify each task runs and observations do not expose geometry."""
    env = Environment(task=task, scene=scene)
    state = env.reset(jax.random.key(1))
    assert set(env.observe_state(state)) == {"state"}
    next_state = env.step(state, jnp.zeros((1, env.action_size)))
    assert float(next_state.time[0]) > 0
    assert np.isfinite(next_state.physics.states.pos).all()


def test_racing_requires_directed_aperture_crossing():
    """Verify racing requires directed aperture crossing."""
    env = Environment(task="racing", scene="racing")
    state = env.reset(jax.random.key(0))
    center = env.task.gate_positions[0]
    normal = env.task.gate_rotations[0, :, 0]
    before = state.physics.states.replace(pos=(center - 0.1 * normal)[None, None])
    after = before.replace(pos=(center + 0.1 * normal)[None, None])
    result = env.task.update(state.task, before, after, jnp.array([1.0]), jnp.array([1.0]))
    assert int(result.gates[0]) == 1
    reverse = env.task.update(state.task, after, before, jnp.array([1.0]), jnp.array([1.0]))
    assert int(reverse.gates[0]) == 0


def test_trajectory_validity_and_derivative_presence():
    """Verify trajectory validity and derivative presence."""
    trajectory = Trajectory(np.array([1.0, 2.0]), np.array([[0.0, 0, 1], [1.0, 0, 1]]))
    np.testing.assert_array_equal(trajectory.sample(1.5)[0], [0.5, 0, 1])
    assert trajectory.sample(1.5)[1] is None
    with pytest.raises(ValueError, match="not valid"):
        trajectory.sample(2.1)
    with pytest.raises(ValueError, match="increasing"):
        Trajectory(np.array([1.0, 1.0]), np.zeros((2, 3)))


def test_sensor_clock_and_masked_sensor_history_reset():
    """Verify sensor clock and masked sensor history reset."""
    env = Environment(task="navigation", scene="S01", sensor="depth", num_envs=2)
    state = env.reset(jax.random.key(0))
    state = env.step(state, jnp.zeros((2, 3)))
    np.testing.assert_array_equal(state.observation.frame, [0, 0])
    state = env.step(state, jnp.zeros((2, 3)))
    np.testing.assert_array_equal(state.observation.frame, [1, 1])
    np.testing.assert_allclose(state.observation.measurement.acquisition_time, 1 / 30, atol=1e-7)
    reset = env.reset(jax.random.key(1), state, jnp.array([True, False]))
    np.testing.assert_array_equal(reset.observation.frame, [0, 1])
    np.testing.assert_array_equal(
        reset.observation.pose_history[1], state.observation.pose_history[1]
    )
    assert env.observe(reset)["depth"].shape == (2, 12, 16, 1)


def test_finished_world_is_immutable_until_reset():
    """Verify finished world is immutable until reset."""
    env = Environment(duration=0.02)
    state = env.step(env.reset(jax.random.key(0)), jnp.zeros((1, 4)))
    assert bool(state.done[0])
    after = env.step(state, jnp.ones((1, 4)))
    for first, second in zip(jax.tree.leaves(state), jax.tree.leaves(after), strict=True):
        if jax.dtypes.issubdtype(first.dtype, jax.dtypes.prng_key):
            first, second = jax.random.key_data(first), jax.random.key_data(second)
        np.testing.assert_array_equal(first, second)
