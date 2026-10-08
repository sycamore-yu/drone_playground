"""Single-source task rewards; learner-specific losses remain in each algorithm."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drone_playground.numerics import euclidean_norm, pseudo_huber


@dataclass(frozen=True)
class TrackingObjective:
    name: str = "tracking_exp"
    scale: float = 1.0
    distance_scale: float = 2.0

    def __call__(self, failed, position, goal):
        error = jnp.linalg.norm(position - goal)
        return self.scale * jnp.where(failed, -1.0, jnp.exp(-self.distance_scale * error))


@dataclass(frozen=True)
class NavigationObjective:
    """The single navigation reward shared by every learning unit (spec 8.1).

    Progress towards the goal, a per-step time cost, a continuous clearance
    penalty that grows as the body approaches any obstacle, an action-smoothness
    term, and frozen terminal terms. Collision, out-of-bounds and numerical
    invalidity carry the same legacy terminal cost. Its old short-corridor
    calibration is not a guarantee against reward shortcuts in longer tasks.
    """

    name: str = "navigation"
    progress_scale: float = 5.0
    time_cost: float = 0.01
    clearance_scale: float = 0.1
    clearance_radius: float = 0.5
    smoothness_scale: float = 0.01
    arrival_bonus: float = 20.0
    failure_penalty: float = -150.0
    """Legacy 15 m calibration, retained for frozen checkpoint reproducibility.

    On a 96 m course the progress potential is 480, not 75. This constant alone
    therefore cannot establish a whole-episode safety ordering. New training
    recipes must audit the actual distance, time cost and continuous gradients.
    """

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
        velocity=None,
        goal_delta=None,
        dt=None,
    ):
        reward = self.progress_scale * (previous_distance - distance) - self.time_cost
        near = jnp.clip(self.clearance_radius - clearance, 0.0, self.clearance_radius)
        reward = reward - self.clearance_scale * near / self.clearance_radius
        reward = reward - self.smoothness_scale * jnp.sum((action - previous_action) ** 2)
        success = arrived & jnp.logical_not(collided)
        reward = reward + jnp.where(success, self.arrival_bonus, 0.0)
        failed = collided | out_of_bounds | numerical_failure
        reward = reward + jnp.where(failed, self.failure_penalty, 0.0)
        return reward


@dataclass(frozen=True)
class MotionNavigationReward:
    """Per-transition motion, clearance and terminal reward for navigation."""

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
        """Validate and prepare the MotionNavigationReward instance after initialization."""
        values = (
            self.target_speed,
            self.max_speed,
            self.clearance_radius,
            self.reward_scale,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Speeds and clearance radius must be finite and positive")
        if self.target_speed > self.max_speed:
            raise ValueError("Commanded target speed cannot exceed the nominal speed maximum")

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
        velocity_cost = jnp.sum(pseudo_huber(velocity - target))
        altitude_cost = pseudo_huber(goal_delta[2])
        # Unlike the legacy clipped proximity term, this retains an escape
        # derivative inside an obstacle and begins shaping before body contact.
        safety_cost = jnp.maximum(self.clearance_radius - clearance, 0.0) ** 2
        safety_cost += jax.nn.softplus(-8.0 * clearance) / 8.0
        speed_cost = jnp.maximum(euclidean_norm(velocity) - self.max_speed, 0.0) ** 2
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
