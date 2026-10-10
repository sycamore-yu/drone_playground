"""Opt-in motion-aware Navigation PPO reward used in the old Brax baseline.

This is a training-only alternative. The active Scenario, physics, action
execution, task terminal events and frozen benchmarking are unchanged.
Reference: older drone_playground MotionNavigationReward('motion-aware-v2').
"""

import jax
import jax.numpy as jnp

from drone_playground.simulation.tasks import Event


def motion_navigation_reward(env, before, after, action):
    """Compute old motion-aware-v2 reward using current per-world true geometry.

    Old physical velocity control is selected separately by the Action; this
    scalar is comparable to its original 0.01-scaled reward definition.
    Values are not normalized on-policy here; Critic/Actor optimizers must
    treat a reward-scale change as a new scratch training recipe.
    """
    previous = before.physics.states.pos[:, 0]
    position = after.physics.states.pos[:, 0]
    velocity = after.physics.states.vel[:, 0]
    target = env.goal_for(after)
    goal_delta = target - position
    distance = jnp.linalg.norm(goal_delta, axis=-1)
    previous_distance = jnp.linalg.norm(target - previous, axis=-1)
    desired_velocity = goal_delta * jnp.minimum(1.0, 2.0 / jnp.maximum(distance, 1e-6))[..., None]

    def pseudo_huber(error):
        return jnp.sqrt(1.0 + error**2) - 1.0

    velocity_cost = jnp.sum(pseudo_huber(velocity - desired_velocity), axis=-1)
    altitude_cost = pseudo_huber(goal_delta[:, 2])
    clearance = env.scene_clearance(position, after.time, after.scene_state) - env.task.radius
    safety_cost = jax.nn.relu(1.2 - clearance) ** 2 + jax.nn.softplus(-8.0 * clearance) / 8.0
    speed_cost = jax.nn.relu(jnp.linalg.norm(velocity, axis=-1) - 20.0) ** 2
    action_rate = jnp.sum(((action - before.previous_action) / env.dt) ** 2, axis=-1)
    continuous_cost = (
        0.1
        + velocity_cost
        + 2.0 * altitude_cost
        + 4.0 * safety_cost
        + speed_cost
        + 1e-4 * action_rate
    )
    event = after.task.event
    failed = (
        (event == Event.COLLISION)
        | (event == Event.OUT_OF_BOUNDS)
        | (event == Event.NUMERICAL_FAILURE)
    ) & ~before.done
    arrived = (event == Event.SUCCESS) & ~before.done
    progress = 0.2 * (previous_distance - distance)
    reward = 0.01 * (
        progress
        - env.dt * continuous_cost
        + 100.0 * arrived.astype(jnp.float32)
        - 1_000.0 * failed.astype(jnp.float32)
    )
    reward = jnp.where(before.done, 0.0, reward)
    return reward, {
        "task_cost": -reward,
        "progress_reward": jnp.where(before.done, 0.0, 0.01 * progress),
        "failure_cost": 10.0 * failed.astype(jnp.float32),
        "height_boundary_cost": jnp.zeros_like(reward),
        "perception_cost": jnp.zeros_like(reward),
        "motion_velocity_cost": 0.01 * env.dt * velocity_cost,
        "motion_altitude_cost": 0.02 * env.dt * altitude_cost,
        "motion_safety_cost": 0.04 * env.dt * safety_cost,
        "motion_arrival_reward": arrived.astype(jnp.float32),
        "motion_action_rate_cost": 1e-6 * env.dt * action_rate,
    }
