"""Per-world procedural Navigation geometry with shared collision and ray queries.

All arrays use immutable static JAX shapes. No benchmark MJCF is loaded during
training. The matched preset covers long flights, variable interior routes,
forest/narrow-passage geometry and both benchmark motion families.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from drone_playground.simulation.scene import _distance, _intersection


def distance_sampling_weights(progress=None):
    """Fixed probe mixture, or three training phases that retain earlier distances."""
    if progress is None:
        return jnp.array([0.4, 0.3, 0.3])
    phase = jnp.clip(jnp.floor(progress * 3).astype(jnp.int32), 0, 2)
    return jnp.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])[phase]


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
    motion: jax.Array | None = None  # [B,N,6] prescribed motion, matched layouts
    route: jax.Array | None = None  # [B,5,3] static-clear construction route


@dataclass(frozen=True)
class SceneDistribution:
    """Static-shape scenario distribution; no runtime mutations to benchmarks."""

    obstacle_motion: str = "mixed"
    max_obstacles: int = 208
    corridor_y: float = -17.5
    training_seed: int = 0
    start_sampling: str = "endpoints"
    min_start_goal_distance_m: float = 0.0
    start_goal_distance_range_m: tuple[float, float] | None = None
    static_obstacle_scale: float = 1.0
    layout: str = "legacy"
    horizontal_beam_probability: float = 0.0

    def __post_init__(self):
        """Validate the analytically reserved start/goal and outside corridor."""
        if self.obstacle_motion not in {"static", "dynamic", "mixed"}:
            raise ValueError("obstacle_motion must be static, dynamic, or mixed")
        if self.layout not in {"legacy", "matched"}:
            raise ValueError("layout must be legacy or matched")
        if type(self.max_obstacles) is not int or self.max_obstacles < 3:
            raise ValueError("max_obstacles must be at least 3")
        if (
            type(self.static_obstacle_scale) not in (int, float)
            or not math.isfinite(self.static_obstacle_scale)
            or not 0 < self.static_obstacle_scale <= 1
        ):
            raise ValueError("static_obstacle_scale must be finite and in (0, 1]")
        if (
            type(self.horizontal_beam_probability) not in (int, float)
            or not math.isfinite(self.horizontal_beam_probability)
            or not 0 <= self.horizontal_beam_probability <= 1
            or (self.horizontal_beam_probability > 0 and self.layout != "matched")
        ):
            raise ValueError("Horizontal beam probability requires matched scenes in [0, 1]")
        if not -19.0 < self.corridor_y < -16.0:
            raise ValueError("corridor_y must lie in the clear outside lane")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValueError("training_seed must be a nonnegative integer")
        if self.start_sampling not in (
            "endpoints",
            "collision_free",
            "stratified",
            "distance_mixture",
            "distance_curriculum",
            "distance_stratified",
        ):
            raise ValueError("Unsupported procedural start_sampling")
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
            "procedural-navigation-v6-matched-horizontal-beams"
            if self.distribution.horizontal_beam_probability > 0
            else "procedural-navigation-v5-matched"
            if self.distribution.layout == "matched"
            else "procedural-navigation-v4-distance-schedule"
            if self.distribution.start_sampling in ("distance_mixture", "distance_curriculum")
            else "procedural-navigation-v3-distance-range"
            if self.distribution.start_goal_distance_range_m is not None
            else "procedural-navigation-v2-long-start"
            if self.distribution.min_start_goal_distance_m > 0
            else "procedural-navigation-v1"
        )
        # Keep v1's byte-for-byte identity compatible with historical checkpoints.
        distribution_identity = vars(self.distribution).copy()
        if self.distribution.layout == "legacy":
            distribution_identity.pop("layout")
        if self.distribution.obstacle_motion == "mixed":
            distribution_identity.pop("obstacle_motion")
        if self.distribution.static_obstacle_scale == 1:
            distribution_identity.pop("static_obstacle_scale")
        if self.distribution.horizontal_beam_probability == 0:
            distribution_identity.pop("horizontal_beam_probability")
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

    def sample(
        self,
        key: jax.Array,
        worlds: int,
        prior: SceneState | None = None,
        *,
        sampling_progress=None,
    ):
        """Sample episode state using training seed, world index and episode."""
        if not isinstance(worlds, int) or worlds < 1:
            raise ValueError("worlds must be positive")
        episodes = jnp.zeros((worlds,), jnp.int32) if prior is None else prior.episode + 1
        master = jax.random.fold_in(key, self.distribution.training_seed)
        indices = jnp.arange(worlds, dtype=jnp.int32)
        keys = jax.vmap(
            lambda index, episode: jax.random.fold_in(jax.random.fold_in(master, index), episode)
        )(indices, episodes)
        if self.distribution.layout == "matched":
            return self._sample_matched(keys, indices, episodes, sampling_progress)
        capacity = self.distribution.max_obstacles
        slot = jnp.arange(capacity)

        def sample_world(rng):
            a, b, c, d, e, f, g, h, k, reset_key = jax.random.split(rng, 10)
            difficulty = jax.random.randint(a, (), 0, 3)
            dynamic = (
                jax.random.bernoulli(b, 0.5)
                if self.distribution.obstacle_motion == "mixed"
                else jnp.asarray(self.distribution.obstacle_motion == "dynamic")
            )
            # Same order of magnitude as the fixed SANDO Navigation8: static
            # easy/medium/hard 41/81/162; dynamic 50/100/200, 65% moving.
            static_count = jnp.array([41, 81, 162])[difficulty]
            dynamic_count = jnp.array([50, 100, 200])[difficulty]
            nominal = jnp.where(dynamic, dynamic_count, static_count)
            count = jnp.clip(nominal + jax.random.randint(c, (), -8, 9), 1, capacity)
            if self.distribution.static_obstacle_scale != 1:
                count = jnp.where(
                    dynamic,
                    count,
                    jnp.maximum(
                        1, (count * self.distribution.static_obstacle_scale).astype(jnp.int32)
                    ),
                )
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
        distance_schedule = self.distribution.start_sampling in (
            "distance_mixture",
            "distance_curriculum",
        )
        if (
            self.distribution.start_sampling in ("collision_free", "stratified")
            or distance_schedule
        ):
            # The old successful training design used broad reset positions.
            # Reserve some near-goal starts for terminal-reward exploration,
            # but keep most episodes distributed throughout the whole course.
            trial_keys = jax.vmap(lambda key: jax.random.fold_in(key, 731))(keys)

            def candidates(key, start_group, goal):
                near_key, coord_key = jax.random.split(key)
                coordinates = jax.random.uniform(coord_key, (16, 3))
                interval = self.distribution.start_goal_distance_range_m
                if distance_schedule:
                    weights = distance_sampling_weights(
                        sampling_progress
                        if self.distribution.start_sampling == "distance_curriculum"
                        else None
                    )
                    group = jax.random.categorical(near_key, jnp.log(weights))
                    radius = jnp.array([20.0, 50.0, 80.0])[group]
                    interval = (radius, radius)
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
            if self.distribution.start_goal_distance_range_m is not None or distance_schedule:
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

    def _sample_matched(self, keys, indices, episodes, progress):
        """Generate routes and obstacles from parameters, never from benchmark layouts.

        Static geometry leaves a different bent interior route in each world.
        Dynamic obstacles may cross it, so motion timing still requires avoidance.
        Only the start and goal are excluded from swept dynamic envelopes.
        """
        capacity = self.distribution.max_obstacles
        count_candidates = 2 * capacity
        slots = jnp.arange(capacity)
        stratified = self.distribution.start_sampling == "distance_stratified"

        def world(key, index, episode):
            shape_key, route_key, distance_key, location_key, size_key, move_key, phase_key = (
                jax.random.split(key, 7)
            )
            family = (index // 3) % 4 if stratified else jax.random.randint(shape_key, (), 0, 4)
            dynamic = (
                jax.random.bernoulli(jax.random.fold_in(shape_key, 1))
                if self.distribution.obstacle_motion == "mixed"
                else jnp.asarray(self.distribution.obstacle_motion == "dynamic")
            )
            nominal = jnp.where(
                dynamic,
                jnp.array([50, 100, 200, 150])[family],
                jnp.array([41, 81, 162, 150])[family],
            )
            count = jnp.minimum(capacity, nominal)
            count = jnp.where(
                dynamic,
                count,
                jnp.maximum(1, (count * self.distribution.static_obstacle_scale).astype(jnp.int32)),
            )
            # Vary the interior topology; no permanently clear y=-17.5 lane.
            bends = jax.random.uniform(route_key, (3,), minval=-12.0, maxval=12.0)
            route = jnp.stack(
                [
                    jnp.array([2.0, 25.0, 50.0, 75.0, 98.0]),
                    jnp.concatenate([jnp.zeros(1), bends, jnp.zeros(1)]),
                    jnp.full(5, 3.0),
                ],
                -1,
            )
            goal = route[-1]
            progress_value = jnp.asarray(1.0 if progress is None else progress)
            alpha = jnp.clip(progress_value, 0, 1)
            weights = (1 - alpha) * jnp.array([0.4, 0.3, 0.3]) + alpha * jnp.array([0.1, 0.2, 0.7])
            distance_group = (
                index % 3
                if stratified
                else jax.random.categorical(jax.random.fold_in(distance_key, 1), jnp.log(weights))
            )
            distance_range = jnp.array([[10.0, 30.0], [30.0, 70.0], [80.0, 98.0]])[distance_group]
            radius = jax.random.uniform(
                distance_key, (), minval=distance_range[0], maxval=distance_range[1]
            )
            radius = jnp.minimum(radius, 96.0)
            if stratified:
                # Half of the long-distance check cases use the full nominal flight.
                radius = jnp.where((distance_group == 2) & ((index // 3) % 2 == 0), 96.0, radius)
            if self.distribution.start_sampling == "endpoints":
                radius = jnp.asarray(96.0)
            a = route[:-1] - goal
            segment = route[1:] - route[:-1]
            aa = jnp.sum(segment * segment, axis=-1)
            bb = 2 * jnp.sum(a * segment, axis=-1)
            cc = jnp.sum(a * a, axis=-1) - radius**2
            discriminant = bb**2 - 4 * aa * cc
            fraction = (-bb - jnp.sqrt(jnp.maximum(discriminant, 0))) / (2 * aa)
            valid_segment = (discriminant >= 0) & (fraction >= -1e-5) & (fraction <= 1 + 1e-5)
            selected_segment = jnp.argmax(valid_segment)
            start = (
                route[selected_segment]
                + jnp.clip(fraction[selected_segment], 0, 1) * segment[selected_segment]
            )

            candidate_slot = jnp.arange(count_candidates)
            units = jax.random.uniform(size_key, (count_candidates, 3))
            locations = jax.random.uniform(location_key, (count_candidates, 3))
            centers = locations * jnp.array([84.0, 36.0, 0.0]) + jnp.array([8.0, -18.0, 3.0])
            moving = dynamic & (candidate_slot % 20 < 13)
            cylinder = ~moving & ((family != 3) | (candidate_slot % 3 != 0))
            sizes_cylinder = jnp.stack(
                [
                    jnp.where(family == 3, 0.7 + units[:, 0] * 0.45, 1.0 + units[:, 0] * 0.5),
                    jnp.where(
                        family == 3,
                        jnp.full(count_candidates, 2.75),
                        jnp.full(count_candidates, 3.0),
                    ),
                    jnp.zeros(count_candidates),
                ],
                -1,
            )
            sizes_box = 0.2 + units * 0.2
            tall = candidate_slot % 3 == 0
            wide = candidate_slot % 3 == 1
            sizes_box = sizes_box.at[:, 2].set(
                jnp.where(tall, 1.8 + 0.2 * units[:, 2], sizes_box[:, 2])
            )
            sizes_box = sizes_box.at[:, 1].set(
                jnp.where(wide, 1.7 + 0.3 * units[:, 1], sizes_box[:, 1])
            )
            sizes = jnp.where(cylinder[:, None], sizes_cylinder, sizes_box)
            # Draw the same two motion equations as Scene.positions, with new parameters.
            motion_units = jax.random.uniform(move_key, (count_candidates, 4))
            phase = jax.random.uniform(phase_key, (count_candidates,), minval=0, maxval=1)
            kind = jnp.where(moving, jnp.where(candidate_slot % 3 == 0, 2, 1), 0)
            scales = 2 + 2 * motion_units[:, :3]
            scales = scales.at[:, 2].set(jnp.minimum(scales[:, 2], 2 * (2.45 - sizes[:, 2])))
            direction_x = candidate_slot % 2 == 0
            bounce = jnp.stack(
                [
                    jnp.where(direction_x, 4 + 3 * motion_units[:, 0], 0),
                    jnp.where(direction_x, 0, 4 + 3 * motion_units[:, 1]),
                    0.7 * motion_units[:, 2],
                ],
                -1,
            )
            amplitudes = jnp.where((kind == 2)[:, None], bounce, scales)
            motion = jnp.concatenate(
                [
                    kind[:, None],
                    amplitudes,
                    jnp.where(kind == 1, phase * 3, 8 + 4 * motion_units[:, 3])[:, None],
                    jnp.where(kind == 1, 31.5 + 16 * motion_units[:, 3], phase)[:, None],
                ],
                -1,
            )
            envelope = jnp.where(
                (kind == 2)[:, None], jnp.abs(bounce), scales * jnp.array([0.5, 0.6, 0.5])
            )
            envelope = jnp.where(moving[:, None], envelope, 0)
            half_extent = jnp.where(
                cylinder[:, None], jnp.stack([sizes[:, 0], sizes[:, 0], sizes[:, 1]], -1), sizes
            )
            centers = jnp.clip(
                centers,
                jnp.array([0.3, -19.7, 0.5]) + half_extent + envelope,
                jnp.array([99.7, 19.7, 6.0]) - half_extent - envelope,
            )
            centers = centers.at[:, 2].set(3.0)
            # Conservative 2-D distances keep tall shapes off the static route.
            delta = centers[:, None, :2] - route[None, :-1, :2]
            vector = segment[:, :2]
            t = jnp.clip(jnp.sum(delta * vector[None], -1) / jnp.sum(vector * vector, -1), 0, 1)
            closest = route[None, :-1, :2] + t[..., None] * vector[None]
            separation = jnp.min(jnp.linalg.norm(centers[:, None, :2] - closest, axis=-1), axis=-1)
            shape_radius = jnp.where(cylinder, sizes[:, 0], jnp.linalg.norm(sizes[:, :2], axis=-1))
            route_clear = moving | (separation > shape_radius + 0.65)
            swept_radius = shape_radius + jnp.linalg.norm(envelope[:, :2], axis=-1)
            ends_clear = (
                jnp.linalg.norm(centers[:, :2] - start[:2], axis=-1) > swept_radius + 1.1
            ) & (jnp.linalg.norm(centers[:, :2] - goal[:2], axis=-1) > swept_radius + 1.1)
            valid = route_clear & ends_clear
            take = jnp.nonzero(valid, size=capacity, fill_value=0)[0]
            enabled = (slots < count) & (slots < jnp.count_nonzero(valid))
            centers, sizes, cylinder, moving, motion = (
                array[take] for array in (centers, sizes, cylinder, moving, motion)
            )
            # Structured cases include two different interior openings, not a side bypass.
            if capacity >= 4:
                gap = 1.4 + 0.8 * jax.random.uniform(jax.random.fold_in(route_key, 8), (2,))
                opening = bends[jnp.array([0, 2])]
                wall_x = jnp.repeat(jnp.array([25.0, 75.0]), 2)
                lower = jnp.stack([jnp.full(2, -20.0), opening + gap], -1).reshape(-1)
                upper = jnp.stack([opening - gap, jnp.full(2, 20.0)], -1).reshape(-1)
                walls = jnp.stack([wall_x, (lower + upper) / 2, jnp.full(4, 3.0)], -1)
                wall_sizes = jnp.stack(
                    [jnp.full(4, 0.35), (upper - lower) / 2, jnp.full(4, 2.75)], -1
                )
                use_walls = family == 3
                centers = centers.at[:4].set(jnp.where(use_walls, walls, centers[:4]))
                sizes = sizes.at[:4].set(jnp.where(use_walls, wall_sizes, sizes[:4]))
                cylinder = cylinder.at[:4].set(jnp.where(use_walls, False, cylinder[:4]))
                moving = moving.at[:4].set(jnp.where(use_walls, False, moving[:4]))
                motion = motion.at[:4].set(jnp.where(use_walls, 0, motion[:4]))
            if self.distribution.horizontal_beam_probability > 0:
                # Static crossbeams exercise vertical bypasses absent from the
                # standard matched static/dynamic distributions. Their x/z/extent vary by world
                # and reset, so this does not sample the held-out S06 geometry.
                # The existing 2-D route is free of other static obstacles;
                # ascending to z=4.3 before a beam gives a 3-D escape route.
                beam_key = jax.random.fold_in(key, 841)
                shape_key, presence_key = jax.random.split(beam_key)
                units = jax.random.uniform(shape_key, (2, 4))
                beam_centers = jnp.stack(
                    [
                        jnp.array([12.0, 62.0]) + 26.0 * units[:, 0],
                        jnp.zeros((2,)),
                        2.35 + 0.7 * units[:, 1],
                    ],
                    -1,
                )
                beam_sizes = jnp.stack(
                    [
                        0.25 + 0.3 * units[:, 2],
                        15.5 + 1.5 * units[:, 3],
                        0.45 + 0.3 * units[:, 2],
                    ],
                    -1,
                )
                beam_enabled = jax.random.bernoulli(
                    presence_key, self.distribution.horizontal_beam_probability, (2,)
                )
                # Distance-based starts may fall near the second beam. Leave
                # four metres for depth acquisition and an altitude change.
                starts_clear = jnp.abs(start[0] - beam_centers[:, 0]) > beam_sizes[:, 0] + 4.0
                beam_enabled &= starts_clear
                mask = beam_enabled[:, None]
                centers = centers.at[:2].set(jnp.where(mask, beam_centers, centers[:2]))
                sizes = sizes.at[:2].set(jnp.where(mask, beam_sizes, sizes[:2]))
                cylinder = cylinder.at[:2].set(jnp.where(beam_enabled, False, cylinder[:2]))
                moving = moving.at[:2].set(jnp.where(beam_enabled, False, moving[:2]))
                motion = motion.at[:2].set(jnp.where(mask, 0, motion[:2]))
                enabled = enabled.at[:2].set(enabled[:2] | beam_enabled)
            return SceneState(
                centers,
                sizes,
                centers,
                jnp.zeros(capacity),
                jnp.zeros(capacity),
                enabled,
                cylinder,
                moving,
                start,
                goal,
                episode,
                motion,
                route,
            )

        return jax.vmap(world)(keys, indices, episodes)

    @staticmethod
    def _centers_at(state: SceneState, times: jax.Array):
        """Return per-world obstacle centers at batched/ray acquisition times."""
        extra = (None,) * (times.ndim - 1)
        index = (slice(None), *extra, slice(None))
        anchor = state.centers[index]
        if state.motion is not None:
            kind, sx, sy, sz, fourth, fifth = (state.motion[index][..., i] for i in range(6))
            t = times[..., None]
            u = 2 * t / jnp.where(kind == 1, fifth, 1) + fourth
            trefoil = jnp.stack(
                (
                    sx / 6 * (jnp.sin(u) + 2 * jnp.sin(2 * u)),
                    sy / 5 * (jnp.cos(u) - 2 * jnp.cos(2 * u)),
                    -sz / 2 * jnp.sin(3 * u),
                ),
                -1,
            )
            phase = t / jnp.where(kind == 2, fourth, 1) + fifth
            triangle = 2 * jnp.abs(2 * (phase - jnp.floor(phase + 0.5))) - 1
            bounce = triangle[..., None] * jnp.stack((sx, sy, sz), -1)
            return anchor + jnp.where(
                (kind == 1)[..., None], trefoil, jnp.where((kind == 2)[..., None], bounce, 0)
            )
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
        # Every ray from one origin shares its acquisition time. Calculate
        # moving obstacle positions once per origin, not once per ray block.
        centers_at_origin = self._centers_at(scene_state, time)
        limits_per_origin = jnp.broadcast_to(jnp.asarray(max_range), batch)
        ranges = jnp.broadcast_to(limits_per_origin[..., None], (*batch, count))
        points = origin[..., None, :]
        points = jnp.broadcast_to(points, directions.shape)
        shape = (batch[0], -1)
        origins = points.reshape((*shape, 3))
        directions = directions.reshape((*shape, 3))
        limits = ranges.reshape(shape)
        count_flat = origins.shape[1]
        chunk_size = min(128, count_flat)
        padded = (-count_flat) % chunk_size
        origins = jnp.pad(origins, ((0, 0), (0, padded), (0, 0)))
        directions = jnp.pad(directions, ((0, 0), (0, padded), (0, 0)))
        limits = jnp.pad(limits, ((0, 0), (0, padded)))
        index = (slice(None), *(None,) * (len(batch) - 1), slice(None))
        sizes = jnp.broadcast_to(scene_state.sizes[index], (*batch, scene_state.sizes.shape[1], 3))
        cylinder = jnp.broadcast_to(
            scene_state.cylinder[index], (*batch, scene_state.active.shape[1])
        )
        active = jnp.broadcast_to(scene_state.active[index], (*batch, scene_state.active.shape[1]))

        def trace_rays(centers, shape_sizes, shape_active, shape_cylinder):
            geometry_count = centers.shape[-2]

            def cast_block(i):
                start = i * chunk_size
                p = jax.lax.dynamic_slice_in_dim(origins, start, chunk_size, 1)
                d = jax.lax.dynamic_slice_in_dim(directions, start, chunk_size, 1)
                limit = jax.lax.dynamic_slice_in_dim(limits, start, chunk_size, 1)
                if len(batch) == 1:
                    center = centers[:, None, :, :]
                    size = shape_sizes[:, None, :, :]
                    enabled = shape_active[:, None, :]
                    kind = shape_cylinder[:, None, :]
                else:
                    # Flattening multiple origins per world interleaves their rays.
                    origin_index = (start + jnp.arange(chunk_size)) // count
                    center = jnp.take(
                        centers.reshape((batch[0], -1, geometry_count, 3)), origin_index, axis=1
                    )
                    size = jnp.take(
                        shape_sizes.reshape((batch[0], -1, geometry_count, 3)),
                        origin_index,
                        axis=1,
                    )
                    enabled = jnp.take(
                        shape_active.reshape((batch[0], -1, geometry_count)),
                        origin_index,
                        axis=1,
                    )
                    kind = jnp.take(
                        shape_cylinder.reshape((batch[0], -1, geometry_count)),
                        origin_index,
                        axis=1,
                    )
                norm = jnp.linalg.norm(d, axis=-1)
                unit = d / jnp.where(norm > 0, norm, 1)[..., None]
                delta = p[..., None, :] - center
                direction = unit[..., None, :]
                cylinder_hit = _intersection("cylinder", delta, direction, size)
                box_hit = _intersection("box", delta, direction, size)
                hit = jnp.where(kind, cylinder_hit, box_hit)
                result = jnp.min(jnp.where(enabled, hit, jnp.inf), axis=-1)
                return jnp.where(norm > 0, jnp.minimum(limit, result), limit)

            return jax.lax.map(cast_block, jnp.arange(origins.shape[1] // chunk_size))

        capacity = scene_state.active.shape[1]
        if capacity <= 64:
            hits = trace_rays(centers_at_origin, sizes, active, cylinder)
        else:
            # Conservative sphere test, evaluated once per origin. Sorting only
            # the nearby obstacles avoids ray/geometry work in dense forests.
            radius = jnp.where(
                cylinder,
                jnp.hypot(sizes[..., 0], sizes[..., 1]),
                jnp.linalg.norm(sizes, axis=-1),
            )
            offset = centers_at_origin - origin[..., None, :]
            distance_squared = jnp.sum(offset * offset, axis=-1)
            nearby = active & (distance_squared <= (limits_per_origin[..., None] + radius) ** 2)

            def trace_nearby():
                _, closest = jax.lax.top_k(jnp.where(nearby, -distance_squared, -jnp.inf), 64)

                def gather(array):
                    return jnp.take_along_axis(array, closest[..., None], axis=-2)

                return trace_rays(
                    gather(centers_at_origin),
                    gather(sizes),
                    jnp.take_along_axis(nearby, closest, axis=-1),
                    jnp.take_along_axis(cylinder, closest, axis=-1),
                )

            hits = jax.lax.cond(
                jnp.any(jnp.sum(nearby, axis=-1) > 64),
                lambda: trace_rays(centers_at_origin, sizes, active, cylinder),
                trace_nearby,
            )
        hits = hits.transpose(1, 0, 2).reshape((batch[0], -1))[:, :count_flat]
        hits = hits.reshape((*batch, count))
        return hits[..., 0] if single_ray else hits
