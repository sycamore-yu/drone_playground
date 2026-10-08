"""The common physics loop preserves schedules and first-terminal-event state."""

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.control.transition import ActionTransition


def test_checked_execution_freezes_only_the_terminated_instance():
    execution = ActionTransition(
        lambda state, value: value, lambda state, value, dt: state + value * dt, 5, 0.002
    )

    def event(old, new, time, memory):
        del old, time
        hit = (new[:, 0] >= 0.003) & (new[:, 0] < 0.005)
        return jnp.ones(2, bool), hit.astype(jnp.int32) * 2, 0.003 - new[:, 0], memory

    commands = jnp.broadcast_to(jnp.array([[1.0], [0.1]]), (5, 2, 1))
    result, clock, outcome, memory, clearance = jax.jit(
        lambda s: execution.checked(
            s, commands, event, timestamp=jnp.zeros(2), outcome=jnp.zeros(2, jnp.int32)
        )
    )(jnp.zeros((2, 1)))
    np.testing.assert_allclose(result[:, 0], [0.004, 0.001], atol=1e-8)
    np.testing.assert_allclose(clock, [0.004, 0.010], atol=1e-8)
    np.testing.assert_array_equal(outcome, [2, 0])
    assert memory == ()
    np.testing.assert_allclose(clearance, [-0.001, 0.002], atol=1e-8)


def test_collision_is_retained_when_execution_continues_to_interval_end():
    execution = ActionTransition(
        lambda state, value: value, lambda state, value, dt: state + value * dt, 5, 0.002
    )

    def event(old, new, time, memory):
        del old, time
        hit = (new[0] > 0.003) & (new[0] < 0.005)
        return jnp.array(True), hit.astype(jnp.int32) * 2, jnp.abs(new[0] - 0.004) - 0.0005, memory

    result, _, outcome, _, clearance = execution.checked(
        jnp.zeros(1),
        jnp.ones((5, 1)),
        event,
        timestamp=jnp.float32(0),
        outcome=jnp.int32(0),
        freeze=False,
    )
    np.testing.assert_allclose(result, [0.01], atol=1e-8)
    assert outcome == 2
    assert clearance < 0


def test_dense_sampling_keeps_the_original_interval_map_and_gradient():
    execution = ActionTransition(
        lambda state, value: value, lambda state, value, dt: state + value * dt**2, 5, 0.02
    )

    def event(old, new, time, memory):
        del old, new, time
        return jnp.array(True), jnp.int32(0), jnp.float32(1), memory

    def endpoint(value):
        return execution.checked(
            jnp.zeros(1),
            jnp.full((5, 1), value),
            event,
            timestamp=jnp.float32(0),
            outcome=jnp.int32(0),
            sample_from_start=True,
        )[0][0]

    np.testing.assert_allclose(jax.jit(endpoint)(2.0), 0.02, atol=1e-7)
    np.testing.assert_allclose(jax.grad(endpoint)(2.0), 0.01, atol=1e-7)
