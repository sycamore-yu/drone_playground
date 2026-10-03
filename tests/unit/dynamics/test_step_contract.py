"""One Dynamics.step contract preserves native controls and integration."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.control.setpoints import AttitudeSetpoint, RateSetpoint, StateSetpoint


def test_all_dynamics_expose_the_same_state_control_dt_signature():
    from drone_playground.dynamics.crazyflow import CrazyflowModel
    from drone_playground.dynamics.lotf import LOTFModel
    from drone_playground.dynamics.point_mass import PointMassLag

    for cls in (CrazyflowModel, LOTFModel, PointMassLag):
        assert hasattr(cls, "step"), cls.__name__
        assert list(inspect.signature(cls.step).parameters) == ["self", "state", "control", "dt"]


def test_crazyflow_step_is_the_native_public_pipeline():
    from crazyflow.sim import Sim, functional

    from drone_playground.dynamics.crazyflow import CrazyflowModel

    sim = Sim(n_worlds=1, n_drones=1, dynamics="so_rpy", drone="cf2x_L250", device="cpu")
    try:
        model = CrazyflowModel().bind(sim)
        control = AttitudeSetpoint(rpy=jnp.array([0.03, -0.02, 0.01]), thrust=jnp.float32(0.3))
        expected = model.step_fn(
            functional.attitude_control(sim.data, control.as_array()[None, None]), 10
        )
        actual = jax.jit(lambda state: model.step(state, control, 0.02))(sim.data)
        for name in ("pos", "vel", "quat", "ang_vel", "rotor_vel"):
            np.testing.assert_allclose(
                getattr(actual.states, name), getattr(expected.states, name), atol=1e-7
            )
        np.testing.assert_array_equal(actual.core.steps, sim.data.core.steps + 10)
        with pytest.raises(ValueError, match="multiple"):
            model.step(sim.data, control, 0.003)
        with pytest.raises(TypeError, match="control"):
            model.step(sim.data, RateSetpoint(thrust=0.3, body_rates=jnp.zeros(3)), 0.02)
    finally:
        sim.close()


def test_point_mass_accepts_only_acceleration_and_preserves_lag_gradient():
    from drone_playground.dynamics.point_mass import PointMassLag, PointMassState

    model = PointMassLag(backward="direct")
    state = PointMassState.create(jnp.array([0.0, 0.0, 1.0]))
    control = StateSetpoint(acceleration=jnp.array([1.0, 0.0, 0.0]))
    result = jax.jit(lambda data: model.step(data, control, 0.02))(state)
    expected_acceleration = 1 - np.exp(-0.02 / model.time_constant)
    assert float(result.acc[0]) == pytest.approx(expected_acceleration, abs=1e-7)
    derivative = jax.grad(
        lambda value: model.step(
            state, StateSetpoint(acceleration=jnp.array([value, 0.0, 0.0])), 0.02
        ).vel[0]
    )(1.0)
    assert float(derivative) == pytest.approx(0.01 * expected_acceleration, abs=1e-7)
    with pytest.raises(TypeError, match="acceleration"):
        model.step(state, StateSetpoint(velocity=jnp.zeros(3)), 0.02)


def test_crazyflow_body_rates_use_native_order_and_no_attitude_controller():
    from crazyflow.sim import Sim, functional

    from drone_playground.dynamics.crazyflow import CrazyflowModel

    sim = Sim(n_worlds=1, n_drones=1, dynamics="first_principles", drone="cf2x_L250", device="cpu")
    try:
        model = CrazyflowModel(forward="first_principles").bind(sim, control_mode="body_rate")
        assert "attitude_controller" not in sim.step_pipeline
        control = RateSetpoint(thrust=0.35, body_rates=jnp.array([0.1, -0.2, 0.3]))
        native = jnp.array([0.1, -0.2, 0.3, 0.35])[None, None]
        expected = model.step_fn(functional.body_rate_control(sim.data, native), 10)
        actual = jax.jit(lambda data: model.step(data, control, 0.02))(sim.data)
        np.testing.assert_allclose(actual.states.pos, expected.states.pos, atol=1e-7)
        np.testing.assert_allclose(actual.controls.body_rate.staged_cmd, native, atol=1e-7)
    finally:
        sim.close()
