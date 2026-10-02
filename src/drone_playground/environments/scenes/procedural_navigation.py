"""SANDO-style procedural navigation generation and connectivity validation.

Sampling order, source constants, placement and reachability rules are preserved
from the original geometry module; collision queries remain in geometry.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from drone_playground.environments.scenes.geometry import (
    BODY_RADIUS_M,
    COLUMN_RADIUS_M,
    CROSSER_HALF_M,
    DIFFICULTIES,
    DYNAMIC_FAMILIES,
    FAMILIES,
    KIND_BOX,
    KIND_CYLINDER,
    MAX_OBSTACLES,
    MOTION_BOUNCE,
    MOTION_NAMES,
    MOTION_PARAMS,
    MOTION_STATIC,
    MOTION_TREFOIL,
    Obstacle,
    SANDO_CUBE_M,
    SANDO_CYLINDER_HEIGHT_M,
    SANDO_EXTRA_CLEARANCE_M,
    SANDO_GLOBAL_TIME_SCALE,
    SANDO_TREFOIL_OFFSET_S,
    SANDO_TREFOIL_SCALES_M,
    SANDO_TREFOIL_SLOWER_RAW,
    SANDO_TREFOIL_SPEED_BOUND_M_PER_S,
    STATIC_FAMILIES,
    SceneBank,
    _pack_instance,
    _stack_instances,
    trefoil_speed_bound,
)


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
    "mixed": (
        ("column", 0.3),
        ("cube", 0.4),
        ("crossbar", 0.2),
        ("pillar", 0.1),
    ),
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
        return Obstacle(
            self.shape, self.kind, centre, size, self.motion, self.params
        )


@dataclass(frozen=True)
class ProceduralNavigationScene:
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
            raise ValueError(
                "A dynamic navigation task requires only moving scene families"
            )
        if not self.dynamic and moving != {False}:
            raise ValueError(
                "A static navigation task requires only static scene families"
            )
        if len(self.density) != 3 or not all(
            0.0 < value < 0.9 for value in self.density
        ):
            raise ValueError("density must declare one fraction per difficulty")
        if self.vertical_scale_m[1] > self.height_m / 2.0 - SANDO_CUBE_M / 2.0:
            raise ValueError(
                "vertical trefoil amplitude does not fit inside the corridor"
            )

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

    def build(
        self, seed: int, per_difficulty: int
    ) -> tuple[SceneBank, dict[str, Any]]:
        """Generate ``per_difficulty`` instances for each difficulty level.

        Instances are laid out in difficulty blocks, so an evaluation cell
        ``(unit, difficulty)`` owns the contiguous slice
        ``difficulty_index * per_difficulty`` and the bank itself is the
        frozen benchmark list the protocol requires.
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
                instance, record = self._generate_one(
                    seed, index, family, difficulty
                )
                instances.append(instance)
                rows.append(record)
        corridor = self.corridor
        bank = _stack_instances(instances).replace(
            start=jnp.asarray(
                np.tile(np.asarray(self.start, np.float32), (count, 1))
            ),
            goal=jnp.asarray(
                np.tile(np.asarray(self.goal, np.float32), (count, 1))
            ),
            difficulty=jnp.asarray(
                [DIFFICULTIES.index(row["difficulty"]) for row in rows],
                np.int32,
            ),
            subtype=jnp.asarray(
                [list(self.families).index(row["family"]) for row in rows],
                np.int32,
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
            "density_by_difficulty": dict(
                zip(DIFFICULTIES, self.density, strict=True)
            ),
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
            obstacles, sampling = self._sample_obstacles(
                instance_seed, family, difficulty
            )
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
                            "origin": [
                                round(value, 6) for value in obstacle.origin
                            ],
                            "size": [
                                round(value, 6) for value in obstacle.size
                            ],
                            "motion": MOTION_NAMES[obstacle.motion],
                            "params": [
                                round(value, 6) for value in obstacle.params
                            ],
                        }
                        for obstacle in obstacles
                    ],
                }
            rejections.append(
                {"instance_seed": instance_seed, "attempt": attempt, **verdict}
            )
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
        shrink_threshold = max(
            1, int(self.max_place_tries * SANDO_SHRINK_AFTER_RATIO)
        )
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
                    shrink = max(
                        self._shrink_floor(shape), shrink * SANDO_SHRINK_RATE
                    )
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
                shape: sum(
                    1 for obstacle in obstacles if obstacle.shape == shape
                )
                for shape in placed_area
            },
            "placement_failures": failures,
            "shrink_events": shrink_events,
            "capacity_limited": capacity_limited,
        }

    @staticmethod
    def _pick_shape(rng: random.Random, budget: dict[str, float]) -> str:
        shapes = [shape for shape, value in budget.items() if value > 1e-9]
        return rng.choices(
            shapes, weights=[budget[shape] for shape in shapes], k=1
        )[0]

    @staticmethod
    def _shrink_floor(shape: str) -> float:
        """SANDO floors the shrunk radius at its minimum source radius (1.0 m)."""
        if shape == "column":
            return COLUMN_RADIUS_M[0] / COLUMN_RADIUS_M[1]
        return SANDO_SHRINK_RATE

    def _sample_element(
        self, rng: random.Random, shape: str, shrink: float
    ) -> Element:
        """Sample geometry and motion; ``shrink`` is SANDO's size-reduction fallback."""
        if shape in ("column", "pillar"):
            radius = (
                rng.uniform(*COLUMN_RADIUS_M) * shrink
                if shape == "column"
                else max(PILLAR_FULL_M[0], PILLAR_FULL_M[1]) / 2.0 * shrink
            )
            height = (
                self.cylinder_height_m
                if shape == "column"
                else PILLAR_FULL_M[2]
            )
            area = (
                math.pi * radius * radius
                if shape == "column"
                else PILLAR_FULL_M[0] * PILLAR_FULL_M[1] * shrink * shrink
            )
            return Element(
                shape,
                KIND_CYLINDER,
                (radius, radius, height / 2.0),
                (0.0,) * 3,
                area,
            )
        if shape == "cube":
            half = SANDO_CUBE_M / 2.0 * shrink
            return Element(
                shape, KIND_BOX, (half,) * 3, (0.0,) * 3, (2 * half) ** 2
            )
        if shape == "crossbar":
            half = tuple(value / 2.0 * shrink for value in CROSSBAR_FULL_M)
            area = CROSSBAR_FULL_M[0] * shrink * CROSSBAR_FULL_M[1] * shrink
            return Element(shape, KIND_BOX, half, (0.0,) * 3, area)
        if shape == "dynamic_cube":
            half = (SANDO_CUBE_M / 2.0,) * 3
            sx, sy = (rng.uniform(*SANDO_TREFOIL_SCALES_M) for _ in range(2))
            sz = rng.uniform(*self.vertical_scale_m)
            offset = rng.uniform(*SANDO_TREFOIL_OFFSET_S)
            slower = (
                rng.uniform(*SANDO_TREFOIL_SLOWER_RAW) * SANDO_GLOBAL_TIME_SCALE
            )
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
            amplitude = rng.uniform(
                1.2,
                SANDO_TREFOIL_SPEED_BOUND_M_PER_S * self.crosser_period_s / 2.0,
            )
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
            return (
                rng.uniform(5.0, 12.0),
                rng.uniform(1.0, 1.8) * rng.choice([-1.0, 1.0]),
                2.0,
            )
        return (
            rng.uniform(
                corridor.x[0] + margin + reach[0],
                corridor.x[1] - margin - reach[0],
            ),
            rng.uniform(
                corridor.y[0] + margin + reach[1],
                corridor.y[1] - margin - reach[1],
            ),
            rng.uniform(
                corridor.z[0] + margin + reach[2],
                corridor.z[1] - margin - reach[2],
            ),
        )

    def _placement_ok(
        self,
        centre: tuple[float, float, float],
        element: Element,
        placed: list[tuple],
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
            if (
                math.hypot(gaps[0], gaps[1]) < self.min_clearance_m
                and gaps[2] <= 0.0
            ):
                return False
        return True

    # -- acceptance -------------------------------------------------------------------

    def _accept(self, obstacles: list[Obstacle]) -> dict[str, Any]:
        corridor = self.corridor
        bounds = (corridor.x, corridor.y, corridor.z)
        for obstacle in obstacles:
            low, high = obstacle.swept_aabb()
            if any(
                low[axis] < bounds[axis][0] - 1e-9
                or high[axis] > bounds[axis][1] + 1e-9
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

    def _occupancy_search(
        self, obstacles: list[Obstacle]
    ) -> tuple[set, bool] | None:
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
            horizontal = np.hypot(
                np.maximum(dx, 0.0), np.maximum(dy, 0.0)
            ) + np.minimum(np.maximum(dx, dy), 0.0)
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

    def _cell_of(
        self, point: tuple[float, float, float], cell: float
    ) -> tuple[int, int]:
        corridor = self.corridor
        return (
            int((point[0] - corridor.x[0]) / cell),
            int((point[1] - corridor.y[0]) / cell),
        )

    def _segments(
        self, obstacles: list[Obstacle]
    ) -> list[tuple[float, float, float]]:
        corridor = self.corridor
        start, goal = corridor.start, corridor.goal
        return [
            tuple(
                start[axis] + (step / 300.0) * (goal[axis] - start[axis])
                for axis in range(3)
            )
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
                        math.hypot(point[0] - centre[0], point[1] - centre[1])
                        < half[0]
                        and abs(point[2] - centre[2]) < half[2]
                    ):
                        return True
                elif all(
                    abs(point[axis] - centre[axis]) < half[axis]
                    for axis in range(3)
                ):
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
                    radial = (
                        math.hypot(point[0] - centre[0], point[1] - centre[1])
                        - half[0]
                    )
                    vertical = abs(point[2] - centre[2]) - half[2]
                    distance = math.hypot(
                        max(radial, 0.0), max(vertical, 0.0)
                    ) + min(max(radial, vertical), 0.0)
                else:
                    q = [
                        abs(point[axis] - centre[axis]) - half[axis]
                        for axis in range(3)
                    ]
                    distance = math.sqrt(
                        sum(max(value, 0.0) ** 2 for value in q)
                    ) + min(max(q), 0.0)
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




def make_bank(
    scene: Any, seed: int, per_difficulty: int
) -> tuple[SceneBank, dict[str, Any]]:
    """Build a scene bank through the selected scene implementation's contract."""
    import inspect

    return (
        scene.build(seed, per_difficulty)
        if inspect.signature(scene.build).parameters
        else scene.build()
    )
