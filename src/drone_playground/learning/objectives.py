"""Single-source task rewards; learner-specific losses remain in each algorithm."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class TrackingObjective:
    name: str = "tracking_exp"
    scale: float = 1.0
    distance_scale: float = 2.0

    def __call__(self, failed, position, goal):
        error = jnp.linalg.norm(position - goal)
        return self.scale * jnp.where(failed, -1.0, jnp.exp(-self.distance_scale * error))
