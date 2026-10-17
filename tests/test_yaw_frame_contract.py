"""A body-frame desired velocity must include a body-relative heading command."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.methods import Action


def test_body_velocity_yaw_zero_offset_tracks_current_heading():
    """Same body action under yaw 0/90 needs correctly rotated world attitude."""
    env = make_environment(num_envs=2)
    state = env.reset(jax.random.key(46))
    q = Rotation.from_euler("z", jnp.array([[0.0], [90.0]]), degrees=True).as_quat()
    physics = state.physics.replace(
        states=state.physics.states.replace(
            quat=q[:, None], vel=jnp.zeros_like(state.physics.states.vel)
        )
    )
    policy = Action(level="velocity_yaw", frame="body")
    world = Action(level="velocity_yaw", frame="world")
    command = jnp.array([[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    output = policy.setpoint(physics, command)
    expected = world.setpoint(
        physics, jnp.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, jnp.pi / 2]])
    )
    np.testing.assert_allclose(output.value, expected.value, atol=1e-5, rtol=1e-5)
    assert policy.contract["frame"] == "body"
    assert policy.contract["yaw_reference"] == "current_heading_offset"
    env.sim.close()


def test_body_velocity_yaw_offset_is_relative_even_under_nonzero_roll_pitch():
    """Heading offset is relative to current world yaw, not body rotation magnitude."""
    env = make_environment(num_envs=1)
    state = env.reset(jax.random.key(31))
    q = Rotation.from_euler("xyz", jnp.array([[12.0, -20.0, 110.0]]), degrees=True).as_quat()
    physics = state.physics.replace(
        states=state.physics.states.replace(
            quat=q[:, None], vel=jnp.zeros_like(state.physics.states.vel)
        )
    )
    policy = Action(level="velocity_yaw", frame="body")
    output = policy.setpoint(physics, jnp.array([[0.0, 0.0, 0.0, jnp.deg2rad(15.0)]]))
    desired_yaw = np.deg2rad(125.0)
    np.testing.assert_allclose(output.value[:, 2], [desired_yaw], atol=1e-5)
    env.sim.close()


def test_legacy_world_velocity_yaw_keeps_absolute_world_heading():
    """Previously trained world-action checkpoints retain their action meaning."""
    env = make_environment(num_envs=1)
    state = env.reset(jax.random.key(42))
    q = Rotation.from_euler("z", jnp.array([[90.0]]), degrees=True).as_quat()
    physics = state.physics.replace(states=state.physics.states.replace(quat=q[:, None]))
    result = Action(level="velocity_yaw", frame="world").setpoint(
        physics, jnp.array([[0.0, 0.0, 0.0, 0.0]])
    )
    np.testing.assert_allclose(result.value[:, 2], [0.0], atol=1e-6)
    assert "yaw_reference" not in Action(level="velocity_yaw", frame="world").contract
    env.sim.close()
