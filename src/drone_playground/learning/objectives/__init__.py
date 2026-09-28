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


@dataclass(frozen=True)
class NavigationObjective:
    """The single navigation reward shared by every learning unit (spec 8.1).

    Progress towards the goal, a per-step time cost, a continuous clearance
    penalty that grows as the body approaches any obstacle, an action-smoothness
    term, and frozen terminal terms. Collision, out-of-bounds and numerical
    invalidity carry the same large negative cost so that ending an episode
    early can never be a high-return shortcut.
    """

    name: str = "navigation"
    progress_scale: float = 5.0
    time_cost: float = 0.01
    clearance_scale: float = 0.1
    clearance_radius: float = 0.5
    smoothness_scale: float = 0.01
    arrival_bonus: float = 20.0
    failure_penalty: float = -150.0
    """Deliberately twice the maximum attainable progress reward.

    Progress reward is ``progress_scale * navigation_distance`` = 75 for the
    frozen 15 m corridor. A failure cost smaller than that makes flying most of
    the way and crashing worth more than hovering, which is exactly the shortcut
    the protocol forbids. At -150 a collided episode is worse than every safe
    outcome, including a full timeout, while a successful arrival (+95) stays the
    best outcome by a wide margin.
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
