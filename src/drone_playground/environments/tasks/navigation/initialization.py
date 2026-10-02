"""Collision-free training starts spanning departure, terminal region and free course.

This sampler queries geometry only to accept physical initial conditions. It
creates no route, action label or extra policy input. Nominal evaluation uses
the catalog's original starts and zero scene phase.
"""

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.geometry import clearance_and_collision, euclidean_norm


def sample_initial_state(
    bank, scenario_id, key, body_radius, target_speed, settings, category=None
):
    kind_key, position_key, phase_key, velocity_key = jax.random.split(key, 4)
    category = jax.random.uniform(kind_key) if category is None else category
    position = settings.get("position") or {}
    weights = position.get("mixture_weights", [1.0, 0.0, 0.0])
    if (
        len(weights) != 3
        or any(w < 0 for w in weights)
        or abs(sum(weights) - 1) > 1e-6
    ):
        raise ValueError(
            "Initial position mixture requires three nonnegative weights summing to one"
        )
    departure, near_goal = (
        category < weights[0],
        category < weights[0] + weights[1],
    )
    phase_bounds = settings.get("scene_phase_s", [0.0, 0.0])
    phase = jnp.where(
        departure,
        0.0,
        jax.random.uniform(
            phase_key, minval=phase_bounds[0], maxval=phase_bounds[1]
        ),
    )
    unit = jax.random.uniform(position_key, (position.get("candidates", 32), 3))
    low = bank.world_low + jnp.array([2.0, 1.0, 0.5])
    high = bank.world_high - jnp.array([2.0, 1.0, 0.5])
    start, goal = bank.start[scenario_id], bank.goal[scenario_id]
    candidates = low + (high - low) * unit
    terminal_candidates = jnp.clip(
        goal
        + (unit - 0.5)
        * jnp.asarray(position.get("goal_region_width_m", [4.0, 4.0, 1.0])),
        low,
        high,
    )
    candidates = jnp.where(near_goal, terminal_candidates, candidates)
    departure_candidates = start + (2 * unit - 1) * jnp.asarray(
        settings.get("position_half_width_m", [0, 0, 0])
    )
    if "position_std_m" in settings:
        departure_candidates = start + jax.random.normal(
            position_key, unit.shape
        ) * jnp.asarray(settings["position_std_m"])
    candidates = jnp.where(departure, departure_candidates, candidates)
    clearances = jax.vmap(
        lambda p: clearance_and_collision(
            bank,
            scenario_id,
            phase,
            p + jnp.array([0.0, 0.0, 0.005]),
            body_radius,
        )[0]
    )(candidates)
    valid = clearances >= position.get("minimum_clearance_m", 0.15)
    found = jnp.any(valid)
    position = jnp.where(found, candidates[jnp.argmax(valid)], start)
    phase = jnp.where(found, phase, 0.0)
    delta = goal - position
    velocity = delta * jnp.minimum(
        1.0, target_speed / jnp.maximum(euclidean_norm(delta), 1e-6)
    )
    velocity = jnp.where(departure, jnp.zeros(3), velocity)
    velocity += jnp.asarray(
        settings.get("velocity_std_mps", 0.0)
    ) * jax.random.normal(velocity_key, (3,))
    velocity = jnp.where(near_goal | ~found, jnp.zeros(3), velocity)
    return position, velocity, phase
