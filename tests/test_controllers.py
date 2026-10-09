"""Tracking implementations share references without changing task rules."""

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.simulation.controllers import IdealTracking, SO3Controller, so3_control
from drone_playground.simulation.environment import Environment
from drone_playground.simulation.methods import PlannerController
from drone_playground.simulation.runner import rollout


def test_so3_hover_matches_original_force_and_rotation():
    """The upstream formula produces mg and identity orientation at hover."""
    p = jnp.array([[0.0, 0.0, 1.0]])
    zero = jnp.zeros_like(p)
    force, quaternion = jax.jit(so3_control)(
        p,
        zero,
        zero,
        (p, zero, zero),
        mass=0.04,
        kx=jnp.ones(3),
        kv=jnp.ones(3),
        yaw=0.0,
    )
    np.testing.assert_allclose(force, [[0, 0, 0.04 * 9.81]], atol=1e-7)
    np.testing.assert_allclose(quaternion, [[0, 0, 0, 1]], atol=1e-7)


def test_so3_can_use_existing_tracking_rollout():
    """Changing the tracker does not change Environment or Task."""
    controller = SO3Controller()
    env = Environment(duration=0.1, control_level=controller.output_level)
    episodes, _ = rollout(env, seed=0, method=PlannerController(controller))
    assert len(episodes) == 1 and episodes[0]["flight_seconds"] >= 0.1 - 1e-6


def test_ideal_tracking_advances_time_and_checks_task():
    """Ideal tracking is a declared response, not a hidden mutation in a controller."""
    env = Environment(duration=0.1, reference="hover", control_level="ideal")
    state = env.reset(jax.random.key(0))
    controller = IdealTracking()
    reference = env.task.target(state.time)
    command, _ = controller(state.physics, reference, controller.initialize_memory(1))
    after = env.step_setpoint(state, command)
    np.testing.assert_allclose(after.physics.states.pos[:, 0], reference[0], atol=1e-7)
    np.testing.assert_allclose(after.time, env.dt, atol=1e-7)


def test_sampling_mpc_executes_real_prediction():
    """A small candidate batch still uses the native predictive optimization."""
    from drone_playground.simulation.mpc import SamplingMPC

    controller = SamplingMPC(samples=32, horizon=5, prediction_seconds=0.2)
    env = Environment(duration=0.04)
    state = env.reset(jax.random.key(0))
    reference = env.task.target(jnp.asarray(controller.reference_offsets)[None])
    command, memory = controller(state.physics, reference, controller.initialize_memory(1, 0))
    assert np.isfinite(command.value).all() and command.value.shape == (1, 4)
    assert memory["mean"].shape == (5, 4)
    after = env.step_setpoint(state, command)
    assert float(after.time[0]) > 0
    controller.close()
