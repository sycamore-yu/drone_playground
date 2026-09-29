"""Collision-free training starts spanning departure, terminal region and free course.

This sampler queries geometry only to accept physical initial conditions. It
creates no route, action label or extra policy input. Nominal evaluation uses
the catalog's original starts and zero scene phase.
"""

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.navigation import clearance_and_collision, euclidean_norm


def sample_free_course(bank, scenario_id, key, body_radius, target_speed, duration):
    kind_key, position_key, phase_key, velocity_key = jax.random.split(key, 4)
    category = jax.random.uniform(kind_key)
    departure, near_goal = category < 0.25, category < 0.5
    phase = jnp.where(departure, 0.0, jax.random.uniform(phase_key, maxval=duration))
    unit = jax.random.uniform(position_key, (32, 3))
    low = bank.world_low + jnp.array([2.0, 1.0, 0.5])
    high = bank.world_high - jnp.array([2.0, 1.0, 0.5])
    start, goal = bank.start[scenario_id], bank.goal[scenario_id]
    candidates = low + (high - low) * unit
    terminal_candidates = jnp.clip(goal + (unit - 0.5) * jnp.array([4.0, 4.0, 1.0]), low, high)
    candidates = jnp.where(near_goal, terminal_candidates, candidates)
    candidates = jnp.where(departure, jnp.broadcast_to(start, candidates.shape), candidates)
    clearances = jax.vmap(
        lambda p: clearance_and_collision(
            bank, scenario_id, phase, p + jnp.array([0.0, 0.0, 0.005]), body_radius
        )[0]
    )(candidates)
    valid = clearances >= 0.15
    found = jnp.any(valid)
    position = jnp.where(found, candidates[jnp.argmax(valid)], start)
    phase = jnp.where(found, phase, 0.0)
    delta = goal - position
    velocity = delta * jnp.minimum(1.0, target_speed / jnp.maximum(euclidean_norm(delta), 1e-6))
    velocity += 0.2 * jax.random.normal(velocity_key, (3,))
    velocity = jnp.where(near_goal | ~found, jnp.zeros(3), velocity)
    return position, velocity, phase
