"""Motion-aware navigation training reward; physical evaluation events are unchanged.

This is a named platform recipe inspired by velocity, clearance and smoothness
objectives in differentiable-flight work, not a reproduction of their coefficients.
Continuous costs are integrated in seconds. Discrete success/failure remains in
the return for PPO and the SHAC critic; BPTT also receives continuous escape and
goal-velocity gradients rather than depending on those binary events.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.geometry import euclidean_norm


def smooth_error(value):
    """Unit pseudo-Huber: finite derivative at zero and linear growth at large errors."""
    return jnp.sqrt(1.0 + value * value) - 1.0


@dataclass(frozen=True)
class MotionNavigationObjective:
    name: str = "navigation"
    recipe: str = "motion-aware-v1"
    progress_scale: float = 1.0
    position_scale: float = 0.0
    time_cost: float = 0.1
    target_speed: float = 2.0
    max_speed: float = 20.0
    velocity_scale: float = 1.0
    altitude_scale: float = 2.0
    clearance_scale: float = 4.0
    clearance_radius: float = 1.2
    smoothness_scale: float = 0.0001
    overspeed_scale: float = 1.0
    arrival_bonus: float = 100.0
    failure_penalty: float = -1000.0
    reward_scale: float = 1.0

    def __post_init__(self):
        values = (
            self.target_speed,
            self.max_speed,
            self.clearance_radius,
            self.reward_scale,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(
                "Speeds and clearance radius must be finite and positive"
            )
        if self.target_speed > self.max_speed:
            raise ValueError(
                "Commanded target speed cannot exceed the nominal speed maximum"
            )

    def __call__(
        self,
        *,
        arrived,
        collided,
        out_of_bounds,
        numerical_failure,
        previous_distance,
        distance,
        clearance,
        action,
        previous_action,
        velocity,
        goal_delta,
        dt,
    ):
        target = jax.lax.stop_gradient(goal_delta)
        target = target * jnp.minimum(
            1.0, self.target_speed / jnp.maximum(euclidean_norm(target), 1e-6)
        )
        velocity_cost = jnp.sum(smooth_error(velocity - target))
        altitude_cost = smooth_error(goal_delta[2])
        # Unlike the legacy clipped proximity term, this retains an escape
        # derivative inside an obstacle and begins shaping before body contact.
        safety_cost = jnp.maximum(self.clearance_radius - clearance, 0.0) ** 2
        safety_cost += jax.nn.softplus(-8.0 * clearance) / 8.0
        speed_cost = (
            jnp.maximum(euclidean_norm(velocity) - self.max_speed, 0.0) ** 2
        )
        action_rate_cost = jnp.sum(((action - previous_action) / dt) ** 2)
        cost = (
            self.time_cost
            + self.position_scale * distance
            + self.velocity_scale * velocity_cost
            + self.altitude_scale * altitude_cost
            + self.clearance_scale * safety_cost
            + self.overspeed_scale * speed_cost
            + self.smoothness_scale * action_rate_cost
        )
        failed = collided | out_of_bounds | numerical_failure
        success = arrived & jnp.logical_not(failed)
        return self.reward_scale * (
            self.progress_scale * (previous_distance - distance)
            - dt * cost
            + jnp.where(success, self.arrival_bonus, 0.0)
            + jnp.where(failed, self.failure_penalty, 0.0)
        )
