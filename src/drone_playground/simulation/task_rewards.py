"""Task-owned reward and shaping functions shared by all learning algorithms.

These calculations belong to the Task/Environment domain. They do not import
PPO, APG, SHAC or a neural network. Named Zhang/Liu perception prediction and
auxiliary losses remain under learning/.
"""

import jax
import jax.numpy as jnp

from drone_playground.simulation.tasks import Event


def motion_navigation_reward(env, before, after, action):
    """Compute the optional old-inspired Navigation task reward.

    The task's configured reward profile selects this function, independently
    of the learning algorithm. The physical action contract is selected by
    Action, and changing either contract requires a separately named run.
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
        "motion_velocity_cost": 0.01 * env.dt * velocity_cost,
        "motion_altitude_cost": 0.02 * env.dt * altitude_cost,
        "motion_safety_cost": 0.04 * env.dt * safety_cost,
        "motion_arrival_reward": arrived.astype(jnp.float32),
        "motion_action_rate_cost": 1e-6 * env.dt * action_rate,
    }


def shaped_task_cost(env, before, after, action, shaping):
    """Return Task cost and named Task terms; preserve old coefficient values.

    `shaping` is the resolved per-experiment task-reward parameter contract.
    Existing runs still supply that contract via the historical Learning
    options; new Task recipes can own it in their task configuration. This
    function deliberately knows nothing about Actor, Critic or perception.
    """
    shaping = {
        "task_weight": 1.0,
        "progress_reward_scale": 0.0,
        "failure_cost": 0.0,
        "altitude_weight": 0.0,
        "height_boundary_weight": 0.0,
        **shaping,
    }
    task_cost, task_terms = env._base_task_reward_components(before, after, action)
    progress_reward = jnp.zeros_like(task_cost)
    if shaping["progress_reward_scale"]:
        if env.task.name == "racing":
            order = env.task.gate_order
            active_gate = order[jnp.minimum(before.task.gates, len(order) - 1)]
            goal = env.task.gate_positions[active_gate]
            previous_distance = jnp.linalg.norm(goal - before.physics.states.pos[:, 0], axis=-1)
            distance = jnp.linalg.norm(goal - after.physics.states.pos[:, 0], axis=-1)
            progress = previous_distance - distance
            progress += 5.0 * (after.task.gates - before.task.gates)
        else:
            goal = env.goal_for(before)
            previous_distance = jnp.linalg.norm(goal - before.physics.states.pos[:, 0], axis=-1)
            distance = jnp.linalg.norm(goal - after.physics.states.pos[:, 0], axis=-1)
            progress = previous_distance - distance
        progress_reward = shaping["progress_reward_scale"] * progress
        progress_reward = jnp.where(before.done, 0.0, progress_reward)
        task_cost -= progress_reward

    failure = after.done & (after.task.event != Event.SUCCESS) & ~before.done
    extra_failure_cost = shaping["failure_cost"] * failure
    event_penalties = shaping.get("failure_cost_by_event")
    if event_penalties is not None:
        event = after.task.event
        extra_failure_cost = (
            jnp.select(
                [
                    event == Event.COLLISION,
                    event == Event.OUT_OF_BOUNDS,
                    event == Event.TIMEOUT,
                    event == Event.NUMERICAL_FAILURE,
                ],
                [
                    event_penalties["collision"],
                    event_penalties["out_of_bounds"],
                    event_penalties["timeout"],
                    event_penalties["numerical_failure"],
                ],
                default=shaping["failure_cost"],
            )
            * failure
        )
    task_cost += extra_failure_cost
    height_boundary_cost = jnp.zeros_like(task_cost)
    if shaping["altitude_weight"]:
        goal = env.goal_for(after)
        altitude_error = after.physics.states.pos[:, 0, 2] - goal[..., 2]
        task_cost += shaping["altitude_weight"] * env.dt * altitude_error**2
    if shaping["height_boundary_weight"]:
        height = after.physics.states.pos[:, 0, 2]
        edge_distance = jnp.minimum(height - env.task.low[2], env.task.high[2] - height)
        height_boundary_cost = (
            shaping["height_boundary_weight"] * env.dt * jax.nn.relu(1.0 - edge_distance) ** 2
        )
        height_boundary_cost = jnp.where(before.done, 0.0, height_boundary_cost)
        task_cost += height_boundary_cost

    additional_terms = {}
    boundary = shaping.get("boundary_proximity")
    if boundary is not None:
        position = after.physics.states.pos[:, 0]
        signed_distances = jnp.minimum(position - env.task.low, env.task.high - position)
        nearest = jnp.min(signed_distances, axis=-1)
        penalty = boundary["weight"] * env.dt * jax.nn.relu(boundary["margin"] - nearest) ** 2
        penalty = jnp.where(before.done, 0.0, penalty)
        task_cost += penalty
        additional_terms["boundary_proximity_cost"] = penalty

    risk = shaping.get("approach_risk")
    if risk is not None:
        previous = before.physics.states.pos[:, 0]
        current = after.physics.states.pos[:, 0]
        old = env.scene_clearance(previous, before.time, before.scene_state) - env.task.radius
        new = env.scene_clearance(current, after.time, after.scene_state) - env.task.radius
        finite = jnp.isfinite(old) & jnp.isfinite(new)
        old_safe = jnp.where(finite, old, risk["margin"] + 1.0)
        new_safe = jnp.where(finite, new, risk["margin"] + 1.0)
        closing = jax.lax.stop_gradient(
            jnp.minimum(jax.nn.relu((old_safe - new_safe) / env.dt), 10.0)
        )
        penalty = (
            risk["weight"]
            * env.dt
            * jax.nn.relu(risk["margin"] - new_safe) ** 2
            * (1.0 + closing / risk["speed_reference"])
        )
        penalty = jnp.where(before.done, 0.0, penalty)
        task_cost += penalty
        additional_terms["approach_risk_cost"] = penalty

    cost = shaping["task_weight"] * task_cost
    terms = {
        **task_terms,
        "task_cost": task_cost,
        "failure_cost": extra_failure_cost + task_terms.get("failure_cost", 0.0),
        "height_boundary_cost": height_boundary_cost + task_terms.get("height_boundary_cost", 0.0),
        "progress_reward": progress_reward + task_terms.get("progress_reward", 0.0),
        **additional_terms,
    }
    return cost, terms
