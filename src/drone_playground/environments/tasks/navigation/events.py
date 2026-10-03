"""Navigation event priority shared by all physical backends and evaluators."""

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.geometry import clearance_and_collision, euclidean_norm


class NavigationEvents:
    """One task definition for all dynamics and observation/execution adapters."""

    def __init__(self, goal_radius, body_radius):
        if goal_radius <= 0 or body_radius <= 0:
            raise ValueError("Task radii must be positive")
        self.goal_radius, self.body_radius = goal_radius, body_radius

    def clearance(self, bank, scenario_id, centre, time):
        return clearance_and_collision(bank, scenario_id, time, centre, self.body_radius)

    def events(self, bank, position, goal, collided, numerical_failure):
        distance = euclidean_norm(position - goal)
        arrived = distance <= self.goal_radius
        outside = jnp.any((position < bank.world_low) | (position > bank.world_high), axis=-1)
        outcome = _outcome(collided, arrived, outside, numerical_failure)
        return distance, arrived, outside, outcome


OUTCOME_RUNNING = 0
OUTCOME_ARRIVED = 1
OUTCOME_COLLISION = 2
OUTCOME_OUT_OF_BOUNDS = 3
OUTCOME_NUMERICAL = 4
OUTCOME_TIMEOUT = 5

OUTCOME_NAMES = {
    OUTCOME_RUNNING: "running",
    OUTCOME_ARRIVED: "arrived",
    OUTCOME_COLLISION: "collision",
    OUTCOME_OUT_OF_BOUNDS: "out_of_bounds",
    OUTCOME_NUMERICAL: "numerical_failure",
    OUTCOME_TIMEOUT: "timeout",
}


def _outcome(collided, arrived, out_of_bounds, numerical_failure) -> jax.Array:
    """Collision wins over arrival in the same step, as the protocol requires."""
    return jnp.where(
        collided,
        OUTCOME_COLLISION,
        jnp.where(
            numerical_failure,
            OUTCOME_NUMERICAL,
            jnp.where(
                out_of_bounds,
                OUTCOME_OUT_OF_BOUNDS,
                jnp.where(arrived, OUTCOME_ARRIVED, OUTCOME_RUNNING),
            ),
        ),
    ).astype(jnp.int32)
