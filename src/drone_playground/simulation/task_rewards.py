"""Task-owned reward and shaping functions shared by all learning algorithms.

These calculations belong to the Task/Environment domain. They do not import
PPO, APG, SHAC or a neural network. Named Zhang/Liu perception prediction and
auxiliary losses remain under learning/.
"""

import jax
import jax.numpy as jnp

from drone_playground.simulation.scene import _distance, _norm
from drone_playground.simulation.tasks import Event


def visfly_reward_terms(
    *,
    position,
    target,
    velocity,
    quaternion,
    angular_velocity,
    direction,
    clearance,
    collision_vector,
    success,
    step_count,
    max_episode_steps,
):
    """Port VisFly NavigationEnv.get_reward at 6ff29bf, without extra shaping.

    Quaternion is wxyz. Inputs are metres, m/s and rad/s. This is a reward
    per decision, exactly as in the upstream source, not a dt-scaled cost.
    """
    delta = target - position
    distance, speed = _norm(delta), _norm(velocity)
    cosine = jnp.sum(direction * velocity, axis=-1) / (speed + 1e-6)
    angle = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
    terms = {
        "target_velocity": 0.01
        * jnp.minimum(jnp.sum(velocity * delta, axis=-1) / (distance + 1e-6), 10.0),
        "view": -0.01 * jax.nn.relu(angle - jnp.pi / 18),
        "orientation": -0.00001 * _norm(quaternion - jnp.array([1.0, 0.0, 0.0, 0.0])),
        "speed": -0.002 * speed,
        "angular_velocity": -0.002 * _norm(angular_velocity),
        "clearance": -0.01 / (clearance + 0.2),
        "approach": -0.005
        * jax.nn.relu(1.0 - clearance)
        * jax.nn.relu(jnp.sum(collision_vector * velocity, axis=-1) / (clearance + 1e-6)),
        "success": success
        * jnp.maximum(max_episode_steps - step_count, 0.0)
        * 0.1
        * (0.2 + 0.8 / (1.0 + speed)),
    }
    return sum(terms.values()), terms


def fan_reward_terms(
    *,
    previous_distance,
    distance,
    velocity,
    acceleration,
    previous_acceleration,
    static_clearance,
    dynamic_clearance,
    obstacle_to_drone,
    obstacle_velocity,
    dynamic_mask,
    collision,
    weights,
    min_speed,
    max_speed,
    safe_distance,
    hover_distance,
):
    """Evaluate Fan 2025 equations 5--16 with explicitly supplied weight hypotheses.

    Numeric reward weights are not published in the paper. Dynamic obstacle
    velocity and nearest surface distances are training-only geometry inputs.
    """
    speed = _norm(velocity)
    obstacle_speed = _norm(obstacle_velocity)
    displacement = _norm(obstacle_to_drone)
    direction = obstacle_velocity / jnp.maximum(obstacle_speed[..., None], 1e-6)
    along = jnp.sum(obstacle_to_drone * direction, axis=-1)
    cosine = along / jnp.maximum(displacement, 1e-6)
    # The value at parallel directions is exact; reward is not differentiated
    # through true obstacle velocity by PPO.
    theta = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
    line_clearance = _norm(obstacle_to_drone - along[..., None] * direction)
    dilation = 1.0 + obstacle_speed * jax.nn.relu(1.0 - 2.0 * theta / jnp.pi) * jnp.exp(
        1.0 / (1.0 + line_clearance)
    )
    effective_distance = jnp.where(dynamic_mask, dynamic_clearance / dilation, jnp.inf)
    dynamic_cost = jnp.sum(jnp.expm1(jax.nn.relu(safe_distance - effective_distance)), -1)
    static_cost = jnp.expm1(jax.nn.relu(safe_distance - static_clearance))
    # A disabled speed penalty must not evaluate an overflowing exponential.
    velocity_cost = (
        jnp.expm1(jax.nn.relu(jnp.where(weights["velocity"] > 0, speed - max_speed, 0.0)))
        + jnp.expm1(jax.nn.relu(jnp.where(weights["velocity"] > 0, min_speed - speed, 0.0)))
    )
    terms = {
        "alive": weights["alive"] * (~collision),
        "acceleration": -weights["acceleration"] * _norm(acceleration),
        "velocity": -weights["velocity"] * velocity_cost,
        "distance": -weights["distance"] * distance,
        "progress": -weights["progress"] * jnp.expm1(distance - previous_distance),
        "jerk": -weights["jerk"] * jnp.expm1(_norm(acceleration - previous_acceleration)),
        "obstacle": -weights["obstacle"] * (static_cost + dynamic_cost),
        "hover": weights["hover"] * jnp.expm1(jax.nn.relu(hover_distance - distance)),
        "collision": -weights["collision"] * collision,
    }
    return sum(terms.values()), terms


def obstacle_reward_geometry(env, after):
    """Obtain per-object distances and velocities from the same collision geometry."""
    position = after.physics.states.pos[:, 0]
    if after.scene_state is not None:
        state = after.scene_state
        centers, velocities = jax.jvp(
            lambda t: env.scene._centers_at(state, t), (after.time,), (jnp.ones_like(after.time),)
        )
        local = position[:, None] - centers
        distances = jnp.where(
            state.cylinder,
            _distance("cylinder", local, state.sizes),
            _distance("box", local, state.sizes),
        )
        return centers, velocities, distances, state.moving, state.active
    centers, velocities = jax.jvp(env.scene.positions, (after.time,), (jnp.ones_like(after.time),))
    distances = jnp.full(centers.shape[:-1], jnp.inf)
    for kind, ids in env.scene._groups:
        local = jnp.einsum(
            "bgi,gij->bgj", position[:, None] - centers[:, ids], env.scene.rotations[ids]
        )
        distances = distances.at[:, ids].set(_distance(kind, local, env.scene.sizes[ids]))
    active = jnp.broadcast_to(env.scene._contact, distances.shape)
    moving = jnp.broadcast_to(env.scene._motion[:, 0] != 0, distances.shape)
    return centers, velocities, distances, moving, active


def paper_navigation_reward(env, before, after, action):
    """Evaluate one paper reward profile with no legacy navigation surcharge."""
    physics = after.physics.states
    position, velocity = physics.pos[:, 0], physics.vel[:, 0]
    target = jnp.broadcast_to(env.goal_for(after), position.shape)
    if env.task.reward_profile in {"fan2025", "fan2025_arrival"}:
        centers, velocities, distances, moving, active = obstacle_reward_geometry(env, after)
        options = env.task.reward_options
        reward, terms = fan_reward_terms(
            previous_distance=_norm(target[:, :2] - before.physics.states.pos[:, 0, :2]),
            distance=_norm(target[:, :2] - position[:, :2]),
            velocity=velocity[:, :2],
            acceleration=env.action.decode(action),
            previous_acceleration=env.action.decode(before.previous_action),
            static_clearance=jnp.min(
                jnp.where(active & ~moving, distances, jnp.inf), -1, initial=jnp.inf
            ),
            dynamic_clearance=jnp.maximum(distances, 0.0),
            obstacle_to_drone=position[:, None, :2] - centers[..., :2],
            obstacle_velocity=velocities[..., :2],
            dynamic_mask=active & moving,
            collision=after.task.event == Event.COLLISION,
            **{key: value for key, value in options.items() if key != "success_reward"},
        )
        if env.task.reward_profile == "fan2025_arrival":
            # Arrival ends this task; positive occupancy rewards favor delaying it.
            terms["alive"] = -terms["alive"]
            terms["hover"] = jnp.zeros_like(reward)
            terms["arrival"] = options["success_reward"] * (after.task.event == Event.SUCCESS)
            reward = sum(terms.values())
    else:

        def clearance_at(p):
            return env.scene_clearance(p, after.time, after.scene_state)

        raw_clearance = clearance_at(position)
        normal = jax.grad(lambda p: jnp.sum(clearance_at(p)))(position)
        distance = jnp.maximum(raw_clearance, 0.0)
        collision_vector = -normal * jnp.where(jnp.isfinite(distance), distance, 0.0)[:, None]
        q = physics.quat[:, 0]
        # Crazyflow uses xyzw; the original VisFly state uses wxyz.
        direction = jnp.stack(
            [
                1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2),
                2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]),
                2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]),
            ],
            -1,
        )
        reward, terms = visfly_reward_terms(
            position=position,
            target=target,
            velocity=velocity,
            quaternion=q[:, jnp.array([3, 0, 1, 2])],
            angular_velocity=physics.ang_vel[:, 0],
            direction=direction,
            clearance=distance,
            collision_vector=collision_vector,
            success=after.task.event == Event.SUCCESS,
            step_count=jnp.ceil(after.time / env.dt - 1e-5),
            max_episode_steps=round(env.task.duration / env.dt),
        )
    terms = {
        f"paper_{name}_reward": jnp.where(before.done, 0.0, term) for name, term in terms.items()
    }
    reward = jnp.where(before.done, 0.0, reward)
    return reward, {**terms, "task_cost": -reward}


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
    if env.task.reward_profile in {"fan2025", "fan2025_arrival", "visfly2025"}:
        reward, terms = paper_navigation_reward(env, before, after, action)
        return -reward, terms
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
