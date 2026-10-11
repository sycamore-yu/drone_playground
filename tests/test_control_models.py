"""Independent control interfaces, models, timing and physical execution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.simulation.environment import Environment
from drone_playground.simulation.networks import Actor


def test_tracking_accepts_acceleration_interface():
    """Changing control quantities does not change the task."""
    env = Environment(action={"level": "acceleration"}, num_envs=2)
    assert env.action_size == 3
    state = env.reset(jax.random.key(4))
    after = env.step(state, jnp.zeros((2, 3)))
    assert np.isfinite(after.physics.states.pos).all()
    assert env.task.name == "tracking"


def test_actor_owns_memory_and_output_size():
    """A state actor can output acceleration without a dummy recurrent array."""
    actor = Actor(kind="state", action_size=3, hidden_size=24)
    memory = actor.initialize_memory(2)
    assert memory.shape == (2, 0)
    obs = {"state": jnp.zeros((2, 12))}
    params = actor.init(jax.random.key(0), obs, memory)
    output, next_memory, _ = actor.apply(params, obs, memory)
    assert output.shape == (2, 3)
    assert next_memory.shape == (2, 0)


def test_nominal_action_scale_does_not_read_random_mass():
    """An unknown mass disturbance cannot change the policy's command meaning."""
    env = Environment()
    state = env.reset(jax.random.key(3))
    altered = state.replace(
        physics=state.physics.replace(
            params=state.physics.params.replace(mass=state.physics.params.mass * 1.2)
        )
    )
    action = jnp.zeros((1, 4))
    np.testing.assert_array_equal(
        env.action_setpoint(state, action).value, env.action_setpoint(altered, action).value
    )


def test_fixed_delay_preserves_command_history_and_reset():
    """Commands wait on the physics clock and each world resets independently."""
    env = Environment(num_envs=2, action_delay_s=0.04)
    state = env.reset(jax.random.key(0))
    zero = env.step(state, jnp.zeros((2, 4)))
    changed = env.step(state, jnp.ones((2, 4)) * 0.1)
    np.testing.assert_array_equal(zero.physics.states.pos, changed.physics.states.pos)
    changed = env.step(changed, jnp.zeros((2, 4)))
    changed = env.step(changed, jnp.zeros((2, 4)))
    reset = env.reset(jax.random.key(9), changed, jnp.array([True, False]))
    np.testing.assert_array_equal(reset.commands.values[1], changed.commands.values[1])
    assert float(reset.time[0]) == 0


@pytest.mark.parametrize("model,level", [("point_mass_lag", "acceleration"), ("lotf", "body_rate")])
def test_backward_model_preserves_forward_and_unmodeled_output(model, level):
    """Substitute modeled derivatives while retaining real attitude or motor derivatives."""
    from drone_playground.learning.dynamics import training_step

    env = Environment(action={"level": level})
    state = env.reset(jax.random.key(4))
    action = jnp.zeros((1, env.action_size))
    step = training_step(env, model)
    actual, expected = step(state, action), env.step(state, action)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)

    def get_motor(fn, u):
        return fn(state, u).physics.states.rotor_vel.sum()

    native = jax.grad(lambda u: get_motor(env.step, u))(action)
    custom = jax.grad(lambda u: get_motor(step, u))(action)
    np.testing.assert_allclose(custom, native, rtol=1e-5, atol=1e-7)
    gradient = jax.grad(lambda u: step(state, u).physics.states.vel.sum())(action)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0


def test_delayed_depth_is_invisible_until_delivery():
    """A captured frame is not a delivered observation."""
    env = Environment(
        task="navigation",
        scene="S01",
        sensor="depth",
        sensor_config={"width": 64, "height": 48, "max_range": 10, "latency": 0.04},
    )
    state = env.reset(jax.random.key(0))
    assert not np.asarray(state.observation.measurement.mask).any()
    state = env.step(state, jnp.zeros((1, 4)))
    assert not np.asarray(state.observation.measurement.mask).any()
    state = env.step(state, jnp.zeros((1, 4)))
    assert np.asarray(state.observation.measurement.mask).any()


def test_physical_randomization_is_per_world_and_resettable():
    """Mass samples change real physics, not the controller's nominal mass."""
    env = Environment(num_envs=3, randomization={"mass": [0.9, 1.1]})
    state = env.reset(jax.random.key(3))
    mass = np.asarray(state.physics.params.mass).reshape(-1)
    assert len(mass) == 3 and len(np.unique(mass)) == 3
    reset = env.reset(jax.random.key(5), state, jnp.array([True, False, False]))
    np.testing.assert_array_equal(reset.physics.params.mass[1:], state.physics.params.mass[1:])
    assert not np.array_equal(reset.physics.params.mass[0], state.physics.params.mass[0])
