"""Per-world procedural Navigation geometry with shared collision and ray queries.

All arrays use immutable static JAX shapes. No benchmark MJCF is loaded during
training. The generator deliberately reserves a certified navigable outside
corridor; start/goal are placed outside the occupied x-range.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from drone_playground.simulation.scene import _distance, _intersection


@struct.dataclass
class SceneState:
    """Episode geometry; B worlds, N static-capacity obstacle slots."""

    centers: jax.Array  # [B,N,3] static anchor / oscillation center
    sizes: jax.Array  # [B,N,3] cylinder radius,half-height / box half-extents
    target: jax.Array  # [B,N,3] dynamic excursion target
    speed: jax.Array  # [B,N] motion speed in metres/s
    phase: jax.Array  # [B,N] independent movement phases
    active: jax.Array  # [B,N] capacity mask
    cylinder: jax.Array  # [B,N] false means axis-aligned box
    moving: jax.Array  # [B,N] false means static geometry
    start: jax.Array  # [B,3] safe start location
    goal: jax.Array  # [B,3] safe terminal sphere center
    episode: jax.Array  # [B] monotone count for deterministic reset streams


@dataclass(frozen=True)
class SceneDistribution:
    """Static-shape scenario distribution; no runtime mutations to benchmarks."""

    max_obstacles: int = 208
    corridor_y: float = -17.5
    training_seed: int = 0
    start_sampling: str = "endpoints"
    min_start_goal_distance_m: float = 0.0
    start_goal_distance_range_m: tuple[float, float] | None = None

    def __post_init__(self):
        """Validate the analytically reserved start/goal and outside corridor."""
        if type(self.max_obstacles) is not int or self.max_obstacles < 3:
            raise ValueError("max_obstacles must be at least 3")
        if not -19.0 < self.corridor_y < -16.0:
            raise ValueError("corridor_y must lie in the clear outside lane")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValueError("training_seed must be a nonnegative integer")
        if self.start_sampling not in ("endpoints", "collision_free", "stratified"):
            raise ValueError("start_sampling must be endpoints, collision_free or stratified")
        if not 0 <= self.min_start_goal_distance_m <= 80.0 or (
            self.min_start_goal_distance_m > 0 and self.start_sampling != "collision_free"
        ):
            raise ValueError("Long-flight distance is supported only for collision_free, up to 80m")
        interval = self.start_goal_distance_range_m
        if interval is not None and (
            self.start_sampling != "collision_free"
            or self.min_start_goal_distance_m != 0
            or not isinstance(interval, (tuple, list))
            or len(interval) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in interval)
            or not 5 <= interval[0] <= interval[1] <= 90
        ):
            raise ValueError(
                "Distance range requires collision_free starts, legacy minimum=0, "
                "and 5<=low<=high<=90m"
            )


class ProceduralScene:
    """Fixed-capacity geometry, independently resampled for each training world.

    Obstacle anchors are confined to x=[8,92], y=[-12,12] and excursions
    to ±2 metres, leaving the corridor at y=-17.5 unobstructed.
    At x<=5 or x>=95 it is possible to connect that corridor to the
    start/goal, independent of the obstacle sample. This is a constructive
    path-existence certificate, not a claim that arbitrary straight paths work.
    """

    name = "train/procedural"

    def __init__(self, config: SceneDistribution | dict | None = None):
        """Bind one explicit generator version and static obstacle capacity."""
        self.distribution = (
            config if isinstance(config, SceneDistribution) else SceneDistribution(**(config or {}))
        )
        generator = (
            "procedural-navigation-v3-distance-range"
            if self.distribution.start_goal_distance_range_m is not None
            else "procedural-navigation-v2-long-start"
            if self.distribution.min_start_goal_distance_m > 0
            else "procedural-navigation-v1"
        )
        # Keep v1's byte-for-byte identity compatible with historical checkpoints.
        distribution_identity = vars(self.distribution).copy()
        if self.distribution.start_goal_distance_range_m is None:
            distribution_identity.pop("start_goal_distance_range_m")
        if self.distribution.min_start_goal_distance_m == 0:
            distribution_identity.pop("min_start_goal_distance_m")
        self.geometry_hash = hashlib.sha256(
            json.dumps({"generator": generator, **distribution_identity}, sort_keys=True).encode()
        ).hexdigest()
        self.geometry_identity = {
            "name": self.name,
            "sha256": self.geometry_hash,
            "generator": generator,
            "distribution": distribution_identity,
        }
        self.geom_names = ()
        self.gate_positions = jnp.zeros((0, 3))
        self.gate_rotations = jnp.zeros((0, 3, 3))
        self.gate_order = ()
        self.numeric = {}

    def sample(self, key: jax.Array, worlds: int, prior: SceneState | None = None):
        """Sample episode state using training seed, world index and episode."""
        if not isinstance(worlds, int) or worlds < 1:
            raise ValueError("worlds must be positive")
        episodes = jnp.zeros((worlds,), jnp.int32) if prior is None else prior.episode + 1
        master = jax.random.fold_in(key, self.distribution.training_seed)
        indices = jnp.arange(worlds, dtype=jnp.int32)
        keys = jax.vmap(
            lambda index, episode: jax.random.fold_in(jax.random.fold_in(master, index), episode)
        )(indices, episodes)
        capacity = self.distribution.max_obstacles
        slot = jnp.arange(capacity)

        def sample_world(rng):
            a, b, c, d, e, f, g, h, k, reset_key = jax.random.split(rng, 10)
            difficulty = jax.random.randint(a, (), 0, 3)
            dynamic = jax.random.bernoulli(b, 0.5)
            # Same order of magnitude as the fixed SANDO Navigation8: static
            # easy/medium/hard 41/81/162; dynamic 50/100/200, 65% moving.
            static_count = jnp.array([41, 81, 162])[difficulty]
            dynamic_count = jnp.array([50, 100, 200])[difficulty]
            nominal = jnp.where(dynamic, dynamic_count, static_count)
            count = jnp.clip(nominal + jax.random.randint(c, (), -8, 9), 1, capacity)
            enabled = slot < count
            moving = dynamic & (slot < (count * 65 // 100))
            cylinder = ~moving & (slot % 2 == 0)
            xy = jax.random.uniform(d, (capacity, 2), minval=0, maxval=1)
            xy = xy * jnp.array([84.0, 24.0]) + jnp.array([8.0, -12.0])
            height = jax.random.uniform(e, (capacity,), minval=1.2, maxval=4.8)
            centers = jnp.concatenate([xy, height[:, None]], -1)
            units = jax.random.uniform(f, (capacity, 3))
            vertical = slot % 4 == 1
            vertical_boxes = units * jnp.array([0.2, 0.2, 0.25]) + jnp.array([0.2, 0.2, 1.8])
            horizontal_boxes = units * jnp.array([0.2, 0.3, 0.2]) + jnp.array([0.2, 1.7, 0.2])
            box_sizes = jnp.where(vertical[:, None], vertical_boxes, horizontal_boxes)
            cylinder_sizes = jnp.stack(
                [
                    units[:, 0] * 0.6 + 0.7,
                    jnp.full((capacity,), 3.0),
                    jnp.zeros((capacity,)),
                ],
                -1,
            )
            sizes = jnp.where(cylinder[:, None], cylinder_sizes, box_sizes)
            sizes = jnp.where(moving[:, None], units * 0.35 + 0.3, sizes)
            # Static cylinders remain six metres high; other shapes stay inside
            # the physical ceiling and the guaranteed corridor.
            centers = centers.at[:, 2].set(
                jnp.where(cylinder, 3.0, jnp.clip(centers[:, 2], sizes[:, 2], 6.0 - sizes[:, 2]))
            )
            excursions = jax.random.uniform(g, (capacity, 3), minval=-1.0, maxval=1.0)
            excursions = excursions * jnp.array([2.0, 2.0, 0.4])
            targets = centers + jnp.where(moving[:, None], excursions, 0.0)
            speeds = jax.random.uniform(h, (capacity,), minval=0.3, maxval=1.8)
            phase = jax.random.uniform(k, (capacity,), minval=0.0, maxval=2 * jnp.pi)
            start_offset = jax.random.uniform(reset_key, (6,), minval=-1.0, maxval=1.0)
            start = jnp.array([2.5, 0.0, 3.0]) + start_offset[:3] * jnp.array([0.7, 2.0, 0.5])
            goal = jnp.array([97.5, 0.0, 3.0]) + start_offset[3:] * jnp.array([0.7, 2.0, 0.5])
            return centers, sizes, targets, speeds, phase, enabled, cylinder, moving, start, goal

        centers, sizes, targets, speed, phase, active, cylinder, moving, start, goal = jax.vmap(
            sample_world
        )(keys)
        scenario = SceneState(
            centers,
            sizes,
            targets,
            speed,
            phase,
            active,
            cylinder,
            moving,
            start,
            goal,
            episodes,
        )
        if self.distribution.start_sampling in ("collision_free", "stratified"):
            # The old successful training design used broad reset positions.
            # Reserve some near-goal starts for terminal-reward exploration,
            # but keep most episodes distributed throughout the whole course.
            trial_keys = jax.vmap(lambda key: jax.random.fold_in(key, 731))(keys)

            def candidates(key, start_group, goal):
                near_key, coord_key = jax.random.split(key)
                coordinates = jax.random.uniform(coord_key, (16, 3))
                interval = self.distribution.start_goal_distance_range_m
                if interval is not None:
                    radius = interval[0] + (interval[1] - interval[0]) * coordinates[:, 0]
                    y_offset = (2 * coordinates[:, 1] - 1) * jnp.minimum(12.0, radius / 2)
                    height = 1.2 + 4.0 * coordinates[:, 2]
                    x_offset = jnp.sqrt(radius**2 - y_offset**2 - (height - goal[2]) ** 2)
                    return jnp.stack([goal[0] - x_offset, goal[1] + y_offset, height], -1)
                if self.distribution.start_sampling == "stratified":
                    # Rotate three equally sized cohorts across episodes:
                    # nominal benchmark starts, arbitrary starts, and near-goal starts.
                    xlow = jnp.where(start_group == 0, 1.8, jnp.where(start_group == 2, 65.0, 2.0))
                    xhigh = jnp.where(start_group == 0, 5.5, 93.0)
                    half_y = jnp.where(start_group == 0, 2.5, 15.0)
                else:
                    near = jax.random.bernoulli(near_key, 0.35)
                    long_flight = self.distribution.min_start_goal_distance_m > 0
                    xlow = jnp.where(long_flight, 2.0, jnp.where(near, 65.0, 2.0))
                    # Procedural goal_x >= 96.8, so start_x <= 16 yields >=80.8m.
                    xhigh = 16.0 if long_flight else 93.0
                    half_y = 15.0
                return jnp.stack(
                    [
                        xlow + (xhigh - xlow) * coordinates[:, 0],
                        -half_y + 2.0 * half_y * coordinates[:, 1],
                        1.2 + 4.0 * coordinates[:, 2],
                    ],
                    axis=-1,
                )

            trial_positions = jax.vmap(candidates)(
                trial_keys, (indices + episodes) % 3, scenario.goal
            )
            distances = self.clearance(trial_positions, scene_state=scenario)
            valid = (distances > 0.45) & (
                jnp.linalg.norm(trial_positions - scenario.goal[:, None, :], axis=-1) > 4.0
            )
            best = jnp.argmax(valid, axis=1)
            start = jnp.take_along_axis(trial_positions, best[:, None, None], axis=1)[:, 0, :]
            fallback = scenario.start
            if self.distribution.start_goal_distance_range_m is not None:
                radius = jnp.linalg.norm(trial_positions[:, 0] - scenario.goal, axis=-1)
                # The clear end zone joins the reserved outside lane at the same distance.
                y_offset = jnp.minimum(radius, scenario.goal[:, 1] - self.distribution.corridor_y)
                fallback = scenario.goal - jnp.stack(
                    [
                        jnp.sqrt(jnp.maximum(radius**2 - y_offset**2, 0)),
                        y_offset,
                        jnp.zeros_like(radius),
                    ],
                    -1,
                )
            scenario = scenario.replace(
                start=jnp.where(jnp.any(valid, axis=1)[:, None], start, fallback)
            )
        return scenario

    @staticmethod
    def _centers_at(state: SceneState, times: jax.Array):
        """Return per-world obstacle centers at batched/ray acquisition times."""
        extra = (None,) * (times.ndim - 1)
        index = (slice(None), *extra, slice(None))
        anchor = state.centers[index]
        target = state.target[index]
        speed = state.speed[index]
        phase = state.phase[index]
        moving = state.moving[index]
        delta = target - anchor
        distance = jnp.maximum(jnp.linalg.norm(delta, axis=-1), 0.2)
        omega = speed / distance
        displacement = delta * jnp.sin(times[..., None] * omega + phase)[..., None]
        return anchor + jnp.where(moving[..., None], displacement, 0.0)

    def clearance(self, position, t=0.0, *, scene_state: SceneState | None = None):
        """Minimum surface clearance against exactly the sampled geometry."""
        if scene_state is None:
            return jnp.full(jnp.asarray(position).shape[:-1], jnp.inf)
        point = jnp.asarray(position)
        if point.ndim < 2 or point.shape[0] != scene_state.active.shape[0]:
            raise ValueError("Procedural clearance needs leading per-world geometry")
        time = jnp.broadcast_to(jnp.asarray(t), point.shape[:-1])
        centers = self._centers_at(scene_state, time)
        local = point[..., None, :] - centers
        index = (slice(None), *(None,) * (point.ndim - 2), slice(None))
        sizes = scene_state.sizes[index]
        cylinder = scene_state.cylinder[index]
        mask = scene_state.active[index]
        distance = jnp.where(
            cylinder, _distance("cylinder", local, sizes), _distance("box", local, sizes)
        )
        return jnp.min(jnp.where(mask, distance, jnp.inf), axis=-1)

    def raycast(self, origin, directions, t=0.0, max_range=40.0, *, scene_state: SceneState):
        """Use the same moving geometry for depth, LiDAR and collision rays."""
        origin, directions = jnp.asarray(origin), jnp.asarray(directions)
        if origin.ndim < 2 or origin.shape[0] != scene_state.active.shape[0]:
            raise ValueError("Procedural raycast requires the batched world axis")
        single_ray = directions.ndim == 1
        if single_ray:
            directions = directions[None, :]
        batch = jnp.broadcast_shapes(origin.shape[:-1], directions.shape[:-2])
        count = directions.shape[-2]
        if batch[0] != scene_state.active.shape[0]:
            raise ValueError("Ray batch and procedural world count disagree")
        origin = jnp.broadcast_to(origin, (*batch, 3))
        directions = jnp.broadcast_to(directions, (*batch, count, 3))
        time = jnp.broadcast_to(jnp.asarray(t), batch)
        limits_per_origin = jnp.broadcast_to(jnp.asarray(max_range), batch)
        ranges = jnp.broadcast_to(limits_per_origin[..., None], (*batch, count))
        points = origin[..., None, :]
        points = jnp.broadcast_to(points, directions.shape)
        shape = (batch[0], -1)
        origins = points.reshape((*shape, 3))
        directions = directions.reshape((*shape, 3))
        times = jnp.broadcast_to(time[..., None], (*batch, count)).reshape(shape)
        limits = ranges.reshape(shape)
        count_flat = origins.shape[1]
        chunk_size = min(128, count_flat)
        padded = (-count_flat) % chunk_size
        origins = jnp.pad(origins, ((0, 0), (0, padded), (0, 0)))
        directions = jnp.pad(directions, ((0, 0), (0, padded), (0, 0)))
        times = jnp.pad(times, ((0, 0), (0, padded)))
        limits = jnp.pad(limits, ((0, 0), (0, padded)))
        index = (slice(None), None, slice(None))
        sizes = scene_state.sizes[index]
        active = scene_state.active[:, None, :]
        cylinder = scene_state.cylinder[:, None, :]

        def cast_block(i):
            start = i * chunk_size
            p = jax.lax.dynamic_slice_in_dim(origins, start, chunk_size, 1)
            d = jax.lax.dynamic_slice_in_dim(directions, start, chunk_size, 1)
            times_i = jax.lax.dynamic_slice_in_dim(times, start, chunk_size, 1)
            limit = jax.lax.dynamic_slice_in_dim(limits, start, chunk_size, 1)
            center = self._centers_at(scene_state, times_i)
            norm = jnp.linalg.norm(d, axis=-1)
            unit = d / jnp.where(norm > 0, norm, 1)[..., None]
            delta = p[..., None, :] - center
            direction = unit[..., None, :]
            cylinder_hit = _intersection("cylinder", delta, direction, sizes)
            box_hit = _intersection("box", delta, direction, sizes)
            hit = jnp.where(cylinder, cylinder_hit, box_hit)
            result = jnp.min(jnp.where(active, hit, jnp.inf), axis=-1)
            return jnp.where(norm > 0, jnp.minimum(limit, result), limit)

        hits = jax.lax.map(cast_block, jnp.arange(origins.shape[1] // chunk_size))
        hits = hits.transpose(1, 0, 2).reshape((batch[0], -1))[:, :count_flat]
        hits = hits.reshape((*batch, count))
        return hits[..., 0] if single_ray else hits
