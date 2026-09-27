"""SANDO-style analytic navigation scenes: geometry, motion and instances.

Geometry and obstacle-motion semantics are reproduced from the pinned SANDO
reference (``research_dev/sando`` at ``3a4450dcc5a8ed5ca825c9da7966e2642be091de``,
BSD-3-Clause, read-only): the trefoil equations, the 0.65 dynamic fraction, the
0.8 m dynamic cube, the 1.0-1.5 m radius forest cylinder and the ``<= 0.5 m/s``
speed bound all come from ``scripts/paper_dynamic_scene.py``; the collision and
clearance rules follow ``scripts/integer_forest_geometry.py``; the start/goal
anchors and the extra-clearance constant follow ``scripts/integer_scene_protocol.py``.

The P5 corridor (20 x 10 x 5 m, 15 m navigation distance, 2 m start/goal height),
the per-difficulty obstacle counts and the ceiling-constrained vertical motion
amplitude are this project's own frozen recipe. Every departure from the source
is recorded in the instance manifest, so source identity and project recipe stay
separable.

Obstacles are analytic primitives. One description therefore serves training
geometry, reward clearance, collision and replay; display, sensor and collision
definitions cannot drift apart because there is only one of them.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

# --------------------------------------------------------------------------------------
# Obstacle taxonomy
# --------------------------------------------------------------------------------------

KIND_EMPTY = 0
KIND_CYLINDER = 1
KIND_BOX = 2

MOTION_STATIC = 0
MOTION_TREFOIL = 1
MOTION_BOUNCE = 2

MOTION_NAMES = {MOTION_STATIC: "static", MOTION_TREFOIL: "trefoil", MOTION_BOUNCE: "linear_bounce"}
KIND_NAMES = {KIND_EMPTY: "empty", KIND_CYLINDER: "cylinder", KIND_BOX: "box"}

# Instances are padded to this capacity so JAX shapes stay static.
MAX_OBSTACLES = 64
MOTION_PARAMS = 5

# --------------------------------------------------------------------------------------
# SANDO source constants (paper_dynamic_scene.py, integer_scene_protocol.py)
# --------------------------------------------------------------------------------------

SANDO_GLOBAL_TIME_SCALE = 7.9442501919
SANDO_DYNAMIC_FRACTION = 0.65
SANDO_CUBE_M = 0.8
SANDO_CYLINDER_RADIUS_M = (1.0, 1.5)
SANDO_CYLINDER_HEIGHT_M = 6.0
SANDO_TREFOIL_SCALES_M = (2.0, 4.0)
SANDO_TREFOIL_OFFSET_S = (0.0, 3.0)
SANDO_TREFOIL_SLOWER_RAW = (4.0, 6.0)
SANDO_TREFOIL_SPEED_BOUND_M_PER_S = 0.5
SANDO_EXTRA_CLEARANCE_M = 0.5
SANDO_ROBOT_RADIUS_M = 0.1
SANDO_NOMINAL_START_M = (0.0, 0.0, 2.0)
SANDO_MAX_PLACEMENT_ATTEMPTS = 10000

# Element size classes. ``column`` is SANDO's forest cylinder; the smaller
# classes are this project's mixed-obstacle elements, sized so a 20 x 10 m
# corridor can carry a meaningful density gradient.
COLUMN_RADIUS_M = SANDO_CYLINDER_RADIUS_M
PILLAR_RADIUS_M = (0.2, 0.35)
CUBE_HALF_M = (0.4, 0.9)
CROSSBAR_HALF_M = (0.12, 0.2)
CROSSER_HALF_M = 0.4

# Body collision sphere of the pinned Crazyflow model
# (crazyflow/drones/cf2x_L250.xml:67, name="col_sphere", size="0.07").
BODY_RADIUS_M = 0.07
BODY_SPHERE_OFFSET_M = (0.0, 0.0, 0.005)

DIFFICULTIES = ("easy", "medium", "hard")
STATIC_FAMILIES = ("cylinder_forest", "mixed")
DYNAMIC_FAMILIES = ("dynamic_forest", "crossers")
FAMILIES = STATIC_FAMILIES + DYNAMIC_FAMILIES


def trefoil_speed_bound(sx: float, sy: float, sz: float, slower: float) -> float:
    """SANDO's analytical upper bound on trefoil speed (paper_dynamic_scene.py:41)."""
    return 2.0 * math.sqrt((5 * sx / 6) ** 2 + sy**2 + (3 * sz / 2) ** 2) / slower


def _axis_gap(low_a, high_a, low_b, high_b) -> float:
    """Separating gap between two intervals; negative means overlap."""
    return max(low_a - high_b, low_b - high_a)


# --------------------------------------------------------------------------------------
# Instance description
# --------------------------------------------------------------------------------------


@dataclass
class Obstacle:
    """One analytic obstacle in the project world frame."""

    shape: str
    kind: int
    origin: tuple[float, float, float]
    """Static centre, or the trefoil/oscillation centre."""

    size: tuple[float, float, float]
    """Cylinder ``(radius, height, 0)``; box half extents ``(hx, hy, hz)``."""

    motion: int = MOTION_STATIC
    params: tuple[float, float, float, float, float] = (0.0,) * MOTION_PARAMS

    def half_extents(self) -> tuple[float, float, float]:
        if self.kind == KIND_CYLINDER:
            return (self.size[0], self.size[0], self.size[1] / 2.0)
        return (self.size[0], self.size[1], self.size[2])

    def position(self, time: float) -> tuple[float, float, float]:
        """Analytic obstacle centre at one simulation time (host-side reference)."""
        x0, y0, z0 = self.origin
        if self.motion == MOTION_STATIC:
            return (x0, y0, z0)
        if self.motion == MOTION_TREFOIL:
            sx, sy, sz, offset, slower = self.params
            tt = 2.0 * time / slower + offset
            return (
                sx / 6.0 * (math.sin(tt) + 2.0 * math.sin(2.0 * tt)) + x0,
                sy / 5.0 * (math.cos(tt) - 2.0 * math.cos(2.0 * tt)) + y0,
                -sz / 2.0 * math.sin(3.0 * tt) + z0,
            )
        if self.motion == MOTION_BOUNCE:
            ax, ay, az, period, phase = self.params
            u = time / period + phase
            tri = 2.0 * abs(2.0 * (u - math.floor(u + 0.5))) - 1.0
            return (x0 + ax * tri, y0 + ay * tri, z0 + az * tri)
        raise ValueError(f"Unknown motion code: {self.motion}")

    def excursion(self) -> tuple[float, float, float]:
        """Half-extent of the reachable centre displacement over all phases."""
        if self.motion == MOTION_STATIC:
            return (0.0, 0.0, 0.0)
        if self.motion == MOTION_TREFOIL:
            sx, sy, sz, _, _ = self.params
            # |sin(t)+2sin(2t)| <= 3, |cos(t)-2cos(2t)| <= 3, |sin(3t)| <= 1
            return (sx / 2.0, sy * 3.0 / 5.0, sz / 2.0)
        ax, ay, az, _, _ = self.params
        return (abs(ax), abs(ay), abs(az))

    def swept_aabb(self):
        """Axis-aligned centre bounds of the body over one full motion period."""
        dx, dy, dz = self.excursion()
        hx, hy, hz = self.half_extents()
        x0, y0, z0 = self.origin
        return (
            (x0 - dx - hx, y0 - dy - hy, z0 - dz - hz),
            (x0 + dx + hx, y0 + dy + hy, z0 + dz + hz),
        )

    def footprint(self, time: float = 0.0):
        """Axis-aligned body bounds at one instant."""
        hx, hy, hz = self.half_extents()
        x, y, z = self.position(time)
        return ((x - hx, y - hy, z - hz), (x + hx, y + hy, z + hz))


# --------------------------------------------------------------------------------------
# Frozen instance bank
# --------------------------------------------------------------------------------------


@struct.dataclass
class SceneBank:
    """Static per-instance scene description shared by every environment.

    The first axis indexes scene instances; ``scenario_id`` in the task state
    selects one. Instances are generated on the host once per split, which keeps
    JAX shapes static and makes train/dev/heldout isolation an explicit,
    inspectable list rather than an implicit draw.
    """

    kind: jax.Array
    size: jax.Array
    origin: jax.Array
    motion: jax.Array
    params: jax.Array
    active: jax.Array
    start: jax.Array
    goal: jax.Array
    difficulty: jax.Array
    subtype: jax.Array
    world_low: jax.Array
    world_high: jax.Array
    subtype_names: tuple = struct.field(pytree_node=False, default=())

    @property
    def num_instances(self) -> int:
        return int(self.kind.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.kind.shape[1])

    def digest(self) -> str:
        """Content hash used to freeze scenes before heldout evaluation."""
        digest = hashlib.sha256()
        for name in ("kind", "size", "origin", "motion", "params", "active", "start", "goal"):
            array = np.asarray(getattr(self, name))
            digest.update(str((name, array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def labels(self, index: int) -> dict[str, str]:
        """Human-readable instance identity used in reports and manifests."""
        return {
            "difficulty": DIFFICULTIES[int(self.difficulty[index])],
            "subtype": str(self.subtype_names[int(self.subtype[index])]),
        }

    def active_count(self, index: int) -> int:
        return int(np.asarray(self.active[index]).sum())


# --------------------------------------------------------------------------------------
# Analytic geometry evaluated on device
# --------------------------------------------------------------------------------------


def obstacle_positions(bank: SceneBank, scenario_id: jax.Array, time: jax.Array) -> jax.Array:
    """Obstacle centres ``[capacity, 3]`` for one instance at one simulation time."""
    origin = bank.origin[scenario_id]
    motion = bank.motion[scenario_id]
    params = bank.params[scenario_id]
    sx, sy, sz, offset, slower = (params[:, index] for index in range(MOTION_PARAMS))

    tt = 2.0 * time / slower + offset
    trefoil = jnp.stack(
        [
            sx / 6.0 * (jnp.sin(tt) + 2.0 * jnp.sin(2.0 * tt)) + origin[:, 0],
            sy / 5.0 * (jnp.cos(tt) - 2.0 * jnp.cos(2.0 * tt)) + origin[:, 1],
            -sz / 2.0 * jnp.sin(3.0 * tt) + origin[:, 2],
        ],
        axis=-1,
    )

    ax, ay, az, period, phase = (params[:, index] for index in range(MOTION_PARAMS))
    u = time / period + phase
    tri = 2.0 * jnp.abs(2.0 * (u - jnp.floor(u + 0.5))) - 1.0
    bounce = origin + jnp.stack([ax, ay, az], axis=-1) * tri[:, None]

    moving = jnp.where((motion == MOTION_TREFOIL)[:, None], trefoil, bounce)
    return jnp.where((motion == MOTION_STATIC)[:, None], origin, moving)


def signed_distance(kind: jax.Array, size: jax.Array, centre: jax.Array, point: jax.Array):
    """Exact signed distance from a point to an analytic primitive.

    Cylinder and box forms follow SANDO's ``signed_distance_point_cylinder``
    and ``signed_distance_point_box``. Negative inside, zero on the surface.
    """
    delta = point - centre
    radial = jnp.hypot(delta[..., 0], delta[..., 1]) - size[..., 0]
    vertical = jnp.abs(delta[..., 2]) - size[..., 1] / 2.0
    cylinder = jnp.hypot(jnp.maximum(radial, 0.0), jnp.maximum(vertical, 0.0)) + jnp.minimum(
        jnp.maximum(radial, vertical), 0.0
    )

    q = jnp.abs(delta) - size
    box = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) + jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return jnp.where(kind == KIND_CYLINDER, cylinder, box)


def clearance_and_collision(
    bank: SceneBank,
    scenario_id: jax.Array,
    time: jax.Array,
    body_centre: jax.Array,
    body_radius: float = BODY_RADIUS_M,
):
    """Signed clearance to the nearest obstacle and the strict contact flag.

    Clearance is negative inside an obstacle, so a body centre inside geometry
    never clamps to a false positive. Contact is ``< radius`` for every
    primitive: SANDO's strict cylinder rule extended to the body sphere. The
    point-mass benchmark rule's box-inclusive boundary is recorded in the P5
    source inventory as the difference.
    """
    centre = obstacle_positions(bank, scenario_id, time)
    distance = signed_distance(bank.kind[scenario_id], bank.size[scenario_id], centre, body_centre)
    distance = jnp.where(bank.active[scenario_id], distance, jnp.inf)
    clearance = jnp.min(distance)
    return clearance - body_radius, clearance < body_radius


def rotate_body_offset(quat: jax.Array, offset: jax.Array) -> jax.Array:
    """Rotate a fixed body-frame offset into world coordinates.

    Crazyflow stores drone orientation as an **xyzw** quaternion
    (``crazyflow/dynamics/symbols.py``); MuJoCo's ``mocap_quat`` is wxyz and the
    replay exporter converts when it writes the trace. Using the wrong layout
    here would silently rotate the collision sphere by 180 degrees.
    """
    x, y, z, w = quat
    rotation = jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            jnp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            jnp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ]
    )
    return rotation @ offset


def body_centre_from_state(pos: jax.Array, quat: jax.Array) -> jax.Array:
    """Collision sphere centre in world coordinates from the body pose."""
    offset = jnp.asarray(BODY_SPHERE_OFFSET_M, jnp.float32)
    return pos + rotate_body_offset(quat, offset)


# --------------------------------------------------------------------------------------
# Corridor and generation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Corridor:
    """P5 navigation corridor recipe."""

    length_m: float = 20.0
    width_m: float = 10.0
    height_m: float = 5.0
    start: tuple[float, float, float] = (0.5, 0.0, 2.0)
    goal: tuple[float, float, float] = (15.5, 0.0, 2.0)
    safety_radius_m: float = SANDO_EXTRA_CLEARANCE_M
    wall_margin_m: float = 0.0

    @property
    def x(self) -> tuple[float, float]:
        return (0.0, self.length_m)

    @property
    def y(self) -> tuple[float, float]:
        return (-self.width_m / 2.0, self.width_m / 2.0)

    @property
    def z(self) -> tuple[float, float]:
        return (0.0, self.height_m)

    @property
    def area_m2(self) -> float:
        return self.length_m * self.width_m


# SANDO's occupied-footprint density targets (generate_random_forest.py:255-263).
SANDO_DENSITY = {"easy": 0.05, "medium": 0.1, "hard": 0.2}
SANDO_MIN_CLEARANCE_M = 1.5
SANDO_MAX_PLACE_TRIES = 2000
SANDO_SHRINK_AFTER_RATIO = 0.5
SANDO_SHRINK_RATE = 0.5

# SANDO element footprints (full sizes) used by its own dynamic-obstacle scenes.
PILLAR_FULL_M = (0.4, 0.4, 4.0)
CROSSBAR_FULL_M = (0.4, 4.0, 0.4)

# Element mix per family: SANDO sizes, this project's weights.
FAMILY_MIX = {
    "cylinder_forest": (("column", 1.0),),
    "mixed": (("column", 0.3), ("cube", 0.4), ("crossbar", 0.2), ("pillar", 0.1)),
    "dynamic_forest": (("dynamic_cube", 0.65), ("column", 0.35)),
    "crossers": (("crosser", 1.0),),
}


@dataclass
class Element:
    """One sampled element: geometry, motion and the area it occupies."""

    shape: str
    kind: int
    half: tuple[float, float, float]
    """Body half extents in world axes."""

    excursion: tuple[float, float, float]
    """Half-extent of the reachable centre displacement over all phases."""

    area_m2: float
    motion: int = MOTION_STATIC
    params: tuple[float, float, float, float, float] = (0.0,) * MOTION_PARAMS

    @property
    def reach(self) -> tuple[float, float, float]:
        """Half extents of the swept body, used for bounds and spacing."""
        return tuple(
            self.half[axis] + self.excursion[axis] for axis in range(3)
        )

    def obstacle(self, centre: tuple[float, float, float]) -> Obstacle:
        size = (
            (self.half[0], self.half[2] * 2.0, 0.0)
            if self.kind == KIND_CYLINDER
            else self.half
        )
        return Obstacle(self.shape, self.kind, centre, size, self.motion, self.params)


@dataclass(frozen=True)
class NavigationScene:
    """Scene presets own SANDO-style geometry and motion, independently of reward.

    Difficulty follows SANDO's density definition: the fraction of the corridor
    area covered by obstacle footprints, at the source values 0.05/0.10/0.20.
    Obstacle count is therefore an outcome of the placement search, exactly as
    in ``generate_random_forest.py``, and is reported per instance.
    """

    name: str = "navigation"
    families: tuple[str, ...] = STATIC_FAMILIES
    dynamic: bool = False
    capacity: int = MAX_OBSTACLES
    length_m: float = 20.0
    width_m: float = 10.0
    height_m: float = 5.0
    start: tuple[float, float, float] = (0.5, 0.0, 2.0)
    goal: tuple[float, float, float] = (15.5, 0.0, 2.0)
    safety_radius_m: float = SANDO_EXTRA_CLEARANCE_M
    density: tuple[float, float, float] = (0.05, 0.1, 0.2)
    min_clearance_m: float = SANDO_MIN_CLEARANCE_M
    max_place_tries: int = SANDO_MAX_PLACE_TRIES
    crosser_count: int = 3
    crosser_period_s: float = 8.0
    vertical_scale_m: tuple[float, float] = (0.6, 1.2)
    """Trefoil vertical amplitude; SANDO uses (2, 4) m in a ceiling-free world."""
    max_generation_attempts: int = 16
    reachability_cell_m: float = 0.2
    source_commit: str = "3a4450dcc5a8ed5ca825c9da7966e2642be091de"

    def __post_init__(self) -> None:
        unknown = [family for family in self.families if family not in FAMILIES]
        if unknown:
            raise ValueError(f"Unknown scene families: {unknown}")
        if not self.families:
            raise ValueError("At least one scene family is required")
        moving = {family in DYNAMIC_FAMILIES for family in self.families}
        if self.dynamic and moving != {True}:
            raise ValueError("A dynamic navigation task requires only moving scene families")
        if not self.dynamic and moving != {False}:
            raise ValueError("A static navigation task requires only static scene families")
        if len(self.density) != 3 or not all(0.0 < value < 0.9 for value in self.density):
            raise ValueError("density must declare one fraction per difficulty")
        if self.vertical_scale_m[1] > self.height_m / 2.0 - SANDO_CUBE_M / 2.0:
            raise ValueError("vertical trefoil amplitude does not fit inside the corridor")

    @property
    def corridor(self) -> Corridor:
        return Corridor(
            length_m=self.length_m,
            width_m=self.width_m,
            height_m=self.height_m,
            start=self.start,
            goal=self.goal,
            safety_radius_m=self.safety_radius_m,
        )

    @property
    def cylinder_height_m(self) -> float:
        """Source cylinder height, clipped to the corridor ceiling (recorded deviation)."""
        return min(SANDO_CYLINDER_HEIGHT_M, self.height_m)

    # -- generation -------------------------------------------------------------------

    def build(self, seed: int, per_difficulty: int) -> tuple[SceneBank, dict[str, Any]]:
        """Generate ``per_difficulty`` instances for each difficulty level.

        Instances are laid out in difficulty blocks, so an evaluation cell
        ``(unit, difficulty)`` owns the contiguous slice
        ``difficulty_index * per_difficulty`` and the bank itself is the
        frozen heldout list the protocol requires.
        """
        if per_difficulty < 1:
            raise ValueError("per_difficulty must be positive")
        count = per_difficulty * len(DIFFICULTIES)
        instances: list[dict[str, np.ndarray]] = []
        rows: list[dict[str, Any]] = []
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            for offset in range(per_difficulty):
                index = difficulty_index * per_difficulty + offset
                family = self.families[offset % len(self.families)]
                instance, record = self._generate_one(seed, index, family, difficulty)
                instances.append(instance)
                rows.append(record)
        corridor = self.corridor
        bank = _stack_instances(instances).replace(
            start=jnp.asarray(np.tile(np.asarray(self.start, np.float32), (count, 1))),
            goal=jnp.asarray(np.tile(np.asarray(self.goal, np.float32), (count, 1))),
            difficulty=jnp.asarray(
                [DIFFICULTIES.index(row["difficulty"]) for row in rows], np.int32
            ),
            subtype=jnp.asarray(
                [list(self.families).index(row["family"]) for row in rows], np.int32
            ),
            subtype_names=tuple(self.families),
            world_low=jnp.asarray(
                [corridor.x[0], corridor.y[0], corridor.z[0]], jnp.float32
            ),
            world_high=jnp.asarray(
                [corridor.x[1], corridor.y[1], corridor.z[1]], jnp.float32
            ),
        )
        manifest = {
            "source": "SANDO generate_random_forest.py + paper_dynamic_scene.py",
            "source_commit": self.source_commit,
            "corridor": asdict(self.corridor),
            "families": list(self.families),
            "dynamic": self.dynamic,
            "density_by_difficulty": dict(zip(DIFFICULTIES, self.density, strict=True)),
            "min_clearance_m": self.min_clearance_m,
            "start": list(self.start),
            "goal": list(self.goal),
            "nav_distance_m": float(math.dist(self.start, self.goal)),
            "body_radius_m": BODY_RADIUS_M,
            "cylinder_height_m": self.cylinder_height_m,
            "source_cylinder_height_m": SANDO_CYLINDER_HEIGHT_M,
            "vertical_scale_m": list(self.vertical_scale_m),
            "generator_seed": int(seed),
            "instances_per_difficulty": per_difficulty,
            "instance_count": count,
            "bank_digest": bank.digest(),
            "instances": rows,
        }
        return bank, manifest

    def _generate_one(
        self, seed: int, index: int, family: str, difficulty: str
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        rejections: list[dict[str, Any]] = []
        for attempt in range(self.max_generation_attempts):
            instance_seed = seed * 1000003 + index * 7919 + attempt
            obstacles, sampling = self._sample_obstacles(instance_seed, family, difficulty)
            verdict = self._accept(obstacles)
            if verdict["accepted"]:
                return _pack_instance(obstacles, self.capacity), {
                    "instance": index,
                    "family": family,
                    "difficulty": difficulty,
                    "instance_seed": instance_seed,
                    "attempt": attempt,
                    "rejections_before_accept": rejections,
                    **sampling,
                    **verdict,
                    "obstacle_table": [
                        {
                            "shape": obstacle.shape,
                            "origin": [round(value, 6) for value in obstacle.origin],
                            "size": [round(value, 6) for value in obstacle.size],
                            "motion": MOTION_NAMES[obstacle.motion],
                            "params": [round(value, 6) for value in obstacle.params],
                        }
                        for obstacle in obstacles
                    ],
                }
            rejections.append({"instance_seed": instance_seed, "attempt": attempt, **verdict})
        raise RuntimeError(
            f"scene generation failed after {self.max_generation_attempts} attempts "
            f"for instance {index} ({family}/{difficulty}); "
            f"last rejection rule: {rejections[-1]['rule']}"
        )

    def _sample_obstacles(
        self, seed: int, family: str, difficulty: str
    ) -> tuple[list[Obstacle], dict[str, Any]]:
        """SANDO's occupied-area loop with rejection sampling and a capacity stop."""
        rng = random.Random(seed)
        corridor = self.corridor
        target_area = SANDO_DENSITY[difficulty] * corridor.area_m2
        mix = FAMILY_MIX[family]
        budget = {shape: weight * target_area for shape, weight in mix}
        placed_area = {shape: 0.0 for shape, _ in mix}
        obstacles: list[Obstacle] = []
        aabbs: list[tuple] = []
        failures = 0
        shrink = 1.0
        shrink_threshold = max(1, int(self.max_place_tries * SANDO_SHRINK_AFTER_RATIO))
        shrink_events = 0
        capacity_limited = False
        while any(value > 1e-9 for value in budget.values()):
            if len(obstacles) >= self.capacity:
                capacity_limited = True
                break
            if failures >= self.max_place_tries:
                break
            shape = self._pick_shape(rng, budget)
            element = self._sample_element(rng, shape, shrink)
            centre = self._sample_centre(rng, element)
            if self._placement_ok(centre, element, aabbs):
                obstacle = element.obstacle(centre)
                obstacles.append(obstacle)
                aabbs.append(obstacle.swept_aabb())
                placed_area[shape] += element.area_m2
                budget[shape] -= element.area_m2
                failures = 0
                shrink = 1.0
            else:
                failures += 1
                if failures > shrink_threshold:
                    shrink = max(self._shrink_floor(shape), shrink * SANDO_SHRINK_RATE)
                    shrink_events += 1
        occupied = sum(placed_area.values())
        return obstacles, {
            "target_area_m2": round(target_area, 4),
            "occupied_area_m2": round(occupied, 4),
            "occupied_fraction": round(occupied / corridor.area_m2, 4),
            "obstacles": len(obstacles),
            "dynamic_obstacles": sum(
                1 for obstacle in obstacles if obstacle.motion != MOTION_STATIC
            ),
            "obstacles_by_shape": {
                shape: sum(1 for obstacle in obstacles if obstacle.shape == shape)
                for shape in placed_area
            },
            "placement_failures": failures,
            "shrink_events": shrink_events,
            "capacity_limited": capacity_limited,
        }

    @staticmethod
    def _pick_shape(rng: random.Random, budget: dict[str, float]) -> str:
        shapes = [shape for shape, value in budget.items() if value > 1e-9]
        return rng.choices(shapes, weights=[budget[shape] for shape in shapes], k=1)[0]

    @staticmethod
    def _shrink_floor(shape: str) -> float:
        """SANDO floors the shrunk radius at its minimum source radius (1.0 m)."""
        if shape == "column":
            return COLUMN_RADIUS_M[0] / COLUMN_RADIUS_M[1]
        return SANDO_SHRINK_RATE

    def _sample_element(self, rng: random.Random, shape: str, shrink: float) -> Element:
        """Sample geometry and motion; ``shrink`` is SANDO's size-reduction fallback."""
        if shape in ("column", "pillar"):
            radius = (
                rng.uniform(*COLUMN_RADIUS_M) * shrink
                if shape == "column"
                else max(PILLAR_FULL_M[0], PILLAR_FULL_M[1]) / 2.0 * shrink
            )
            height = self.cylinder_height_m if shape == "column" else PILLAR_FULL_M[2]
            area = (
                math.pi * radius * radius
                if shape == "column"
                else PILLAR_FULL_M[0] * PILLAR_FULL_M[1] * shrink * shrink
            )
            return Element(shape, KIND_CYLINDER, (radius, radius, height / 2.0), (0.0,) * 3, area)
        if shape == "cube":
            half = SANDO_CUBE_M / 2.0 * shrink
            return Element(shape, KIND_BOX, (half,) * 3, (0.0,) * 3, (2 * half) ** 2)
        if shape == "crossbar":
            half = tuple(value / 2.0 * shrink for value in CROSSBAR_FULL_M)
            area = CROSSBAR_FULL_M[0] * shrink * CROSSBAR_FULL_M[1] * shrink
            return Element(shape, KIND_BOX, half, (0.0,) * 3, area)
        if shape == "dynamic_cube":
            half = (SANDO_CUBE_M / 2.0,) * 3
            sx, sy = (rng.uniform(*SANDO_TREFOIL_SCALES_M) for _ in range(2))
            sz = rng.uniform(*self.vertical_scale_m)
            offset = rng.uniform(*SANDO_TREFOIL_OFFSET_S)
            slower = rng.uniform(*SANDO_TREFOIL_SLOWER_RAW) * SANDO_GLOBAL_TIME_SCALE
            excursion = (sx / 2.0, sy * 3.0 / 5.0, sz / 2.0)
            return Element(
                shape,
                KIND_BOX,
                half,
                excursion,
                SANDO_CUBE_M**2,
                MOTION_TREFOIL,
                (sx, sy, sz, offset, slower),
            )
        if shape == "crosser":
            half = (CROSSER_HALF_M,) * 3
            # 2*amplitude/period <= SANDO's 0.5 m/s dynamic speed bound.
            amplitude = rng.uniform(1.2, SANDO_TREFOIL_SPEED_BOUND_M_PER_S * self.crosser_period_s / 2.0)
            phase = rng.uniform(0.0, 1.0)
            return Element(
                shape,
                KIND_BOX,
                half,
                (0.0, amplitude, 0.0),
                (2 * CROSSER_HALF_M) ** 2,
                MOTION_BOUNCE,
                (0.0, amplitude, 0.0, self.crosser_period_s, phase),
            )
        raise ValueError(f"Unknown element shape: {shape}")

    def _sample_centre(
        self, rng: random.Random, element: Element
    ) -> tuple[float, float, float]:
        """Uniform centre sample inside the bounds shrunk by the swept reach."""
        corridor = self.corridor
        margin = corridor.wall_margin_m
        reach = list(element.reach)
        if element.shape == "crosser":
            return (rng.uniform(5.0, 12.0), rng.uniform(1.0, 1.8) * rng.choice([-1.0, 1.0]), 2.0)
        return (
            rng.uniform(corridor.x[0] + margin + reach[0], corridor.x[1] - margin - reach[0]),
            rng.uniform(corridor.y[0] + margin + reach[1], corridor.y[1] - margin - reach[1]),
            rng.uniform(corridor.z[0] + margin + reach[2], corridor.z[1] - margin - reach[2]),
        )

    def _placement_ok(
        self, centre: tuple[float, float, float], element: Element, placed: list[tuple]
    ) -> bool:
        """Reject the start/goal safety box and too-tight pairwise swept spacing.

        SANDO rejects only the start point (robot radius 0.1 m) and tests the XY
        centre distance of the planned obstacle. This project additionally clears
        a safety box around the goal and compares swept bounding boxes, so a
        moving element can never overlap static geometry. Both strengthenings are
        recorded in the manifest.
        """
        corridor = self.corridor
        reach = element.reach
        for anchor in (corridor.start, corridor.goal):
            gaps = (
                max(
                    anchor[axis] - (centre[axis] + reach[axis]),
                    (centre[axis] - reach[axis]) - anchor[axis],
                )
                for axis in range(3)
            )
            if max(gaps) < corridor.safety_radius_m:
                return False
        low = tuple(centre[axis] - reach[axis] for axis in range(3))
        high = tuple(centre[axis] + reach[axis] for axis in range(3))
        for other_low, other_high in placed:
            gaps = [
                max(
                    low[axis] - other_high[axis],
                    other_low[axis] - high[axis],
                )
                for axis in range(3)
            ]
            if math.hypot(gaps[0], gaps[1]) < self.min_clearance_m and gaps[2] <= 0.0:
                return False
        return True

    # -- acceptance -------------------------------------------------------------------

    def _accept(self, obstacles: list[Obstacle]) -> dict[str, Any]:
        corridor = self.corridor
        bounds = (corridor.x, corridor.y, corridor.z)
        for obstacle in obstacles:
            low, high = obstacle.swept_aabb()
            if any(
                low[axis] < bounds[axis][0] - 1e-9 or high[axis] > bounds[axis][1] + 1e-9
                for axis in range(3)
            ):
                return {
                    "accepted": False,
                    "rule": "obstacle-swept-aabb-outside-corridor",
                    "obstacle": obstacle.shape,
                }
        search = self._occupancy_search(obstacles)
        if search is None:
            return {
                "accepted": False,
                "rule": "start-or-goal-cell-occupied-under-conservative-occupancy",
            }
        cells, connected = search
        if not connected:
            return {
                "accepted": False,
                "rule": "start-goal-not-connected-under-conservative-occupancy",
                "reachable_cells": int(len(cells)),
            }
        return {
            "accepted": True,
            "rule": "accepted",
            "reachable_cells": int(len(cells)),
            "straight_line_blocked": self._straight_line_blocked(obstacles),
            "min_path_clearance_m": self._min_path_clearance(obstacles),
            "max_speed_m_per_s": self._max_speed(obstacles),
        }

    def _occupancy_search(self, obstacles: list[Obstacle]) -> tuple[set, bool] | None:
        """Breadth-first connectivity check on a conservatively inflated grid.

        Cells are tested against obstacle footprints at ``t = 0``. For dynamic
        scenes this is the static skeleton at the episode start; the valid
        instance rule additionally requires that no element ever sweeps across
        the start or goal safety box, which the placement rule enforces.
        """
        corridor = self.corridor
        cell = self.reachability_cell_m
        nx = int(round(corridor.length_m / cell))
        ny = int(round(corridor.width_m / cell))
        inflation = BODY_RADIUS_M + 0.5 * cell * math.sqrt(2.0)
        xs = corridor.x[0] + (np.arange(nx) + 0.5) * cell
        ys = corridor.y[0] + (np.arange(ny) + 0.5) * cell
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        blocked = np.zeros((nx, ny), dtype=bool)
        for obstacle in obstacles:
            low, high = obstacle.footprint(0.0)
            if (
                low[0] - inflation > xs[-1]
                or high[0] + inflation < xs[0]
                or low[1] - inflation > ys[-1]
                or high[1] + inflation < ys[0]
            ):
                continue
            half = obstacle.half_extents()
            dx = np.abs(grid_x - obstacle.origin[0]) - half[0] - inflation
            dy = np.abs(grid_y - obstacle.origin[1]) - half[1] - inflation
            horizontal = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)) + np.minimum(
                np.maximum(dx, dy), 0.0
            )
            blocked |= horizontal < 0.0
        start_cell = self._cell_of(corridor.start, cell)
        goal_cell = self._cell_of(corridor.goal, cell)
        if blocked[start_cell] or blocked[goal_cell]:
            return None
        seen = {start_cell}
        frontier = [start_cell]
        while frontier:
            current = frontier.pop()
            for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (current[0] + delta[0], current[1] + delta[1])
                if not (0 <= neighbour[0] < nx and 0 <= neighbour[1] < ny):
                    continue
                if neighbour in seen or blocked[neighbour]:
                    continue
                seen.add(neighbour)
                frontier.append(neighbour)
        return seen, goal_cell in seen

    def _cell_of(self, point: tuple[float, float, float], cell: float) -> tuple[int, int]:
        corridor = self.corridor
        return (
            int((point[0] - corridor.x[0]) / cell),
            int((point[1] - corridor.y[0]) / cell),
        )

    def _segments(self, obstacles: list[Obstacle]) -> list[tuple[float, float, float]]:
        corridor = self.corridor
        start, goal = corridor.start, corridor.goal
        return [
            tuple(start[axis] + (step / 300.0) * (goal[axis] - start[axis]) for axis in range(3))
            for step in range(301)
        ]

    def _straight_line_blocked(self, obstacles: list[Obstacle]) -> bool:
        """Does any obstacle cross the direct start-goal segment at ``t = 0``?"""
        points = self._segments(obstacles)
        for obstacle in obstacles:
            centre = obstacle.position(0.0)
            half = obstacle.half_extents()
            for point in points:
                if obstacle.kind == KIND_CYLINDER:
                    if (
                        math.hypot(point[0] - centre[0], point[1] - centre[1]) < half[0]
                        and abs(point[2] - centre[2]) < half[2]
                    ):
                        return True
                elif all(abs(point[axis] - centre[axis]) < half[axis] for axis in range(3)):
                    return True
        return False

    def _min_path_clearance(self, obstacles: list[Obstacle]) -> float:
        """Body-centre clearance to the nearest obstacle along the direct segment."""
        points = self._segments(obstacles)
        best = math.inf
        for point in points:
            for obstacle in obstacles:
                centre = obstacle.position(0.0)
                half = obstacle.half_extents()
                if obstacle.kind == KIND_CYLINDER:
                    radial = math.hypot(point[0] - centre[0], point[1] - centre[1]) - half[0]
                    vertical = abs(point[2] - centre[2]) - half[2]
                    distance = math.hypot(
                        max(radial, 0.0), max(vertical, 0.0)
                    ) + min(max(radial, vertical), 0.0)
                else:
                    q = [abs(point[axis] - centre[axis]) - half[axis] for axis in range(3)]
                    distance = math.sqrt(sum(max(value, 0.0) ** 2 for value in q)) + min(
                        max(q), 0.0
                    )
                best = min(best, distance)
        return round(best - BODY_RADIUS_M, 4)

    def _max_speed(self, obstacles: list[Obstacle]) -> float:
        speeds = []
        for obstacle in obstacles:
            if obstacle.motion == MOTION_TREFOIL:
                sx, sy, sz, _, slower = obstacle.params
                speeds.append(trefoil_speed_bound(sx, sy, sz, slower))
            elif obstacle.motion == MOTION_BOUNCE:
                ax, ay, az, period, _ = obstacle.params
                speeds.append(2.0 * math.sqrt(ax**2 + ay**2 + az**2) / period)
        return float(max(speeds)) if speeds else 0.0


def _pack_instance(obstacles: list[Obstacle], capacity: int) -> dict[str, np.ndarray]:
    if len(obstacles) > capacity:
        raise ValueError(f"instance has {len(obstacles)} obstacles, capacity is {capacity}")
    instance = {
        "kind": np.zeros(capacity, np.int32),
        "size": np.zeros((capacity, 3), np.float32),
        "origin": np.zeros((capacity, 3), np.float32),
        "motion": np.zeros(capacity, np.int32),
        "params": np.zeros((capacity, MOTION_PARAMS), np.float32),
        "active": np.zeros(capacity, bool),
    }
    for index, obstacle in enumerate(obstacles):
        instance["kind"][index] = obstacle.kind
        instance["size"][index] = obstacle.size
        instance["origin"][index] = obstacle.origin
        instance["motion"][index] = obstacle.motion
        instance["params"][index] = obstacle.params
        instance["active"][index] = True
    return instance


def _stack_instances(instances: list[dict[str, np.ndarray]]) -> SceneBank:
    return SceneBank(
        kind=jnp.asarray(np.stack([item["kind"] for item in instances])),
        size=jnp.asarray(np.stack([item["size"] for item in instances])),
        origin=jnp.asarray(np.stack([item["origin"] for item in instances])),
        motion=jnp.asarray(np.stack([item["motion"] for item in instances])),
        params=jnp.asarray(np.stack([item["params"] for item in instances])),
        active=jnp.asarray(np.stack([item["active"] for item in instances])),
        start=jnp.zeros((len(instances), 3)),
        goal=jnp.zeros((len(instances), 3)),
        difficulty=jnp.zeros((len(instances),), jnp.int32),
        subtype=jnp.zeros((len(instances),), jnp.int32),
        world_low=jnp.zeros(3),
        world_high=jnp.zeros(3),
        subtype_names=(),
    )


def make_bank(
    scene: NavigationScene, seed: int, per_difficulty: int
) -> tuple[SceneBank, dict[str, Any]]:
    """Build a scene bank plus its manifest, ``per_difficulty`` instances each."""
    return scene.build(seed, per_difficulty)


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text)
