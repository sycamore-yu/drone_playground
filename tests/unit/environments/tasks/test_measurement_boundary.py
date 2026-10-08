"""Measurement corruption must not redefine the ground-truth training target."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.dynamics.point_mass import PointMassState
from drone_playground.environments.observations.flight_state import FlightStateObservation
from drone_playground.environments.tasks.navigation.acceleration import AccelerationNavigationTask


@pytest.mark.parametrize("channels", [1, 3], ids=["depth", "pointcloud"])
def test_measurement_noise_changes_actor_input_not_loss_target(channels):
    # Isolate measurement from geometry using one ideal sensor return, with
    # the real observation, corruption and training-target computations.
    values = jnp.ones((2,) if channels == 1 else (2, 3))
    task = object.__new__(AccelerationNavigationTask)
    task.sensor = SimpleNamespace(
        sample=lambda *_: (values, jnp.ones(2, dtype=bool)),
        policy_value_channels=channels,
    )
    task.observation = FlightStateObservation()
    task.body_radius = 0.07
    task.transition = SimpleNamespace(dt=0.1)
    task.observation_noise = {}
    bank = SimpleNamespace(num_instances=1, goal=jnp.array([[10.0, 3.0, 2.0]]))
    state = PointMassState.create(jnp.array([[0.0, 0.0, 1.0]])).replace(
        vel=jnp.array([[2.0, 0.0, 0.0]]),
        measurement_key=jax.random.split(jax.random.PRNGKey(17), 1),
    )
    original_state = jax.tree.map(lambda value: np.array(value), state)
    speeds = jnp.array([4.0])
    ideal = task.measure(bank, state, jnp.zeros(1), speeds)
    task.observation_noise = {"position_std_m": 1.0, "velocity_std_mps": 0.2}
    noisy = task.measure(bank, state, jnp.zeros(1), speeds)

    assert not np.array_equal(ideal[2], noisy[2])
    np.testing.assert_array_equal(noisy[3], ideal[3])
    ideal_loss = jnp.sum(jnp.square(state.vel - ideal[3]))
    noisy_loss = jnp.sum(jnp.square(state.vel - noisy[3]))
    np.testing.assert_array_equal(noisy_loss, ideal_loss)
    for before, after in zip(jax.tree.leaves(original_state), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(before, after)
