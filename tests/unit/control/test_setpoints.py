"""Typed physical setpoints preserve meaning across JAX and host boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_state_setpoint_keeps_velocity_and_acceleration_distinct():
    from drone_playground.control.setpoints import StateSetpoint, validate_setpoint

    velocity = StateSetpoint(velocity=jnp.array([1.0, 2.0, 3.0]), yaw=0.2)
    acceleration = StateSetpoint(acceleration=jnp.array([1.0, 2.0, 3.0]))
    validate_setpoint(velocity)
    validate_setpoint(acceleration)
    assert velocity.kind == acceleration.kind == "state"
    assert velocity.acceleration is None
    assert acceleration.velocity is None
    with pytest.raises(ValueError, match="field"):
        validate_setpoint(StateSetpoint())


def test_rate_and_attitude_setpoints_have_different_physical_fields():
    from drone_playground.control.setpoints import AttitudeSetpoint, RateSetpoint

    attitude = AttitudeSetpoint(rpy=jnp.array([0.1, 0.2, 0.3]), thrust=1.0)
    rates = RateSetpoint(thrust=1.0, body_rates=jnp.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(attitude.as_array(), [0.1, 0.2, 0.3, 1.0])
    np.testing.assert_allclose(rates.as_array(), [1.0, 0.1, 0.2, 0.3])


def test_setpoint_is_a_batched_differentiable_pytree():
    from drone_playground.control.setpoints import RateSetpoint

    def command(thrust):
        return RateSetpoint(thrust=thrust, body_rates=jnp.ones(3)).as_array().sum()

    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(command)))(jnp.ones(2)), 1.0)


def test_host_setpoint_validation_rejects_invalid_units_shape_or_values():
    from drone_playground.control.setpoints import RateSetpoint, StateSetpoint, validate_setpoint

    with pytest.raises(ValueError, match="three"):
        validate_setpoint(RateSetpoint(thrust=1.0, body_rates=[0.1, 0.2]))
    with pytest.raises(ValueError, match="finite"):
        validate_setpoint(StateSetpoint(acceleration=[0.0, np.nan, 0.0]))
