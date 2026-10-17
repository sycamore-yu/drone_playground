"""Prevent frame regressions in accepted world-reference Tracking/Racing tasks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.methods import Action


@pytest.mark.parametrize("task,scene", [("tracking", "empty"), ("racing", "racing")])
def test_world_reference_errors_and_attitude_are_all_observable(task, scene):
    """World-error actors must receive world orientation, not only body-state vectors."""
    env = make_environment(task=task, scene=scene, num_envs=2)
    state = env.reset(jax.random.key(35))
    obs = env.observe(state)["state"]
    physics = state.physics.states
    target, velocity_target, acceleration_target = env.task.target(state.time)
    np.testing.assert_allclose(obs[:, :3], target - physics.pos[:, 0], atol=1e-6)
    np.testing.assert_allclose(obs[:, 3:6], velocity_target - physics.vel[:, 0], atol=1e-6)
    np.testing.assert_allclose(obs[:, 6:9], acceleration_target, atol=1e-6)
    np.testing.assert_allclose(obs[:, 9:13], physics.quat[:, 0], atol=1e-6)
    np.testing.assert_allclose(obs[:, 13:16], physics.ang_vel[:, 0], atol=1e-6)
    np.testing.assert_allclose(obs[:, 16:19], physics.vel[:, 0], atol=1e-6)
    for i, dt in enumerate((0.2, 0.5, 1.0)):
        predicted = env.task.target(state.time + dt)[0] - physics.pos[:, 0]
        np.testing.assert_allclose(obs[:, 19 + 3 * i : 22 + 3 * i], predicted, atol=1e-6)
    if task == "racing":
        gate_index = env.task.gate_order[
            jnp.minimum(state.task.gates, len(env.task.gate_order) - 1)
        ]
        expected = env.task.gate_positions[gate_index] - physics.pos[:, 0]
        np.testing.assert_allclose(obs[:, 28:31], expected, atol=1e-6)
    attitude_rotation = Rotation.from_euler("z", jnp.array([[90.0], [45.0]]), degrees=True)
    rotated = state.replace(
        physics=state.physics.replace(
            states=physics.replace(quat=attitude_rotation.as_quat()[:, None])
        )
    )
    observation_rotated = env.observe(rotated)["state"]
    assert not np.allclose(obs[:, 9:13], observation_rotated[:, 9:13])
    assert Action(level="attitude_thrust").contract["frame"] == "world"
    env.sim.close()


def test_attitude_thrust_is_inertial_orientation_plus_scalar_thrust_not_world_acceleration():
    """No body/world vector reinterpretation may alter a native Euler/thrust request."""
    env = make_environment(num_envs=1, action={"level": "attitude_thrust"})
    state = env.reset(jax.random.key(9))
    action = Action(level="attitude_thrust")
    request = jnp.array([[0.15, -0.2, 0.5, 0.8 * action.hover]])
    setpoint = action.setpoint(state.physics, request)
    assert setpoint.level == "attitude_thrust" and setpoint.frame == "world"
    np.testing.assert_array_equal(setpoint.value, request)
    env.sim.close()
