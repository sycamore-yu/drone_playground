"""Body velocity, target velocity, attitude and vehicle-size observation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drone_playground.numerics import euclidean_norm


@dataclass(frozen=True)
class FlightStateObservation:
    name: str = "flight_state"
    proprioception_size: int = 10

    def proprioception(self, state, goal, speed, radius):
        """Body velocity/target velocity, world-up row of attitude, vehicle radius."""
        delta = goal - jax.lax.stop_gradient(state.pos)
        distance = euclidean_norm(delta)
        target = delta * jnp.minimum(1.0, speed / jnp.maximum(distance, 1e-6))[..., None]
        body_velocity = jnp.einsum("...ij,...i->...j", state.rotation, state.vel)
        body_target = jnp.einsum("...ij,...i->...j", state.rotation, target)
        r = jnp.full((*state.pos.shape[:-1], 1), radius)
        return (
            jnp.concatenate([body_velocity, body_target, state.rotation[..., 2, :], r], -1),
            target,
        )
