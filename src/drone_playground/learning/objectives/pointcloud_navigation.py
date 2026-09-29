"""Continuous loss for explicitly trained, delayed point-cloud navigation.

Task-domain adaptation adds instantaneous velocity and near-goal position
supervision. The original paper reconstruction and its moving-average loss are
unchanged. These coefficients are project choices, not undisclosed paper values.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.navigation import euclidean_norm
from drone_playground.learning.objectives.navigation import smooth_error


@dataclass(frozen=True)
class PointCloudNavigationObjective:
    velocity_weight: float = 1.0
    position_weight: float = 0.1
    altitude_weight: float = 0.2
    clearance_weight: float = 8.0
    clearance_margin: float = 1.0
    acceleration_weight: float = 0.005
    jerk_weight: float = 0.0005
    overspeed_weight: float = 1.0
    max_speed: float = 20.0

    def __call__(self, previous, current, goal, speeds, clearance, command, last, dt):
        direction = jax.lax.stop_gradient(goal - previous.pos)
        target = (
            direction
            * jnp.minimum(1.0, speeds / jnp.maximum(euclidean_norm(direction), 1e-6))[..., None]
        )
        parts = dict(
            velocity=jnp.mean(jnp.sum(smooth_error(current.vel - target), axis=-1)),
            position=jnp.mean(euclidean_norm(current.pos - goal)),
            altitude=jnp.mean(smooth_error(current.pos[..., 2] - goal[..., 2])),
            clearance=jnp.mean(
                jax.nn.relu(self.clearance_margin - clearance) ** 2
                + jax.nn.softplus(-8 * clearance) / 8
            ),
            acceleration=jnp.mean(jnp.sum(command**2, axis=-1)),
            jerk=jnp.mean(jnp.sum(((command - last) / dt) ** 2, axis=-1)),
            overspeed=jnp.mean(jax.nn.relu(euclidean_norm(current.vel) - self.max_speed) ** 2),
        )
        return sum(getattr(self, name + "_weight") * value for name, value in parts.items()), parts
