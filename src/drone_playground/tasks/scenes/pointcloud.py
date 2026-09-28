"""Explicit static primitive reconstruction recipe; Navigation8 is never sampled here.

The paper identifies primitive classes but not its exact generator. Counts and
ranges below are reconstruction settings, recorded in the experiment config.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drone_playground.tasks.scenes.navigation import (
    KIND_BOX,
    KIND_CYLINDER,
    KIND_SPHERE,
    SceneBank,
)


@dataclass(frozen=True)
class PaperPrimitiveScene:
    name: str = "paper_static_primitives_reconstruction"
    obstacles_per_kind: int = 30
    length_m: float = 48.0
    width_m: float = 18.0
    height_m: float = 6.0
    cylinder_radius_m: tuple[float, float] = (0.05, 0.4)
    box_half_m: tuple[float, float] = (0.2, 0.7)
    sphere_radius_m: tuple[float, float] = (0.2, 0.6)

    def __post_init__(self):
        if self.obstacles_per_kind < 1 or self.length_m <= 8:
            raise ValueError("A primitive scene needs obstacles and a corridor longer than 8 m")

    def sample(self, key, count):
        """Sample one independent static scene per environment using explicit PRNG state."""
        position_key, size_key, target_key = jax.random.split(key, 3)
        n = self.obstacles_per_kind
        capacity = 3 * n
        unit = jax.random.uniform(position_key, (count, capacity, 3))
        low = jnp.array([3.0, -self.width_m / 2, 0.8])
        high = jnp.array([self.length_m - 3, self.width_m / 2, self.height_m - 0.8])
        origin = low + unit * (high - low)
        origin = origin.at[:, :n, 2].set(self.height_m / 2)
        unit_size = jax.random.uniform(size_key, (count, capacity, 3))
        cr0, cr1 = self.cylinder_radius_m
        bh0, bh1 = self.box_half_m
        sr0, sr1 = self.sphere_radius_m
        size = bh0 + (bh1 - bh0) * unit_size
        radius = cr0 + (cr1 - cr0) * unit_size[:, :n, 0]
        size = size.at[:, :n, 0].set(radius).at[:, :n, 1].set(self.height_m).at[:, :n, 2].set(0.0)
        sphere_radius = sr0 + (sr1 - sr0) * unit_size[:, 2 * n :, :1]
        size = size.at[:, 2 * n :].set(jnp.broadcast_to(sphere_radius, (count, n, 3)))
        kind = jnp.repeat(jnp.array([KIND_CYLINDER, KIND_BOX, KIND_SPHERE], jnp.int32), n)
        lateral = jax.random.uniform(target_key, (count, 2), minval=-3.0, maxval=3.0)
        start = jnp.stack([jnp.ones(count), lateral[:, 0], jnp.full(count, 3.0)], -1)
        goal = jnp.stack(
            [jnp.full(count, self.length_m - 1), lateral[:, 1], jnp.full(count, 3.0)], -1
        )
        return SceneBank(
            kind=jnp.broadcast_to(kind, (count, capacity)),
            size=size,
            origin=origin,
            motion=jnp.zeros((count, capacity), jnp.int32),
            params=jnp.zeros((count, capacity, 5)),
            active=jnp.ones((count, capacity), bool),
            start=start,
            goal=goal,
            difficulty=jnp.zeros(count, jnp.int32),
            subtype=jnp.zeros(count, jnp.int32),
            subtype_names=(self.name,),
            world_low=jnp.array([0.0, -self.width_m / 2, 0.0]),
            world_high=jnp.array([self.length_m, self.width_m / 2, self.height_m]),
        )
