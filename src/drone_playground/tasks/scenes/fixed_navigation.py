"""Fixed, numbered P5 navigation scenes for manual review and later freezing.

The random density generator remains readable for historical experiment
reconstruction, but current benchmark candidates are explicit committed scene
catalogs. Legacy v1-v3 catalogs contain an ``inspection_path`` used only for
review. Route-free v4 deliberately stores no oracle/reference trajectory:
connectivity is checked with offline 3-D occupancy A* and the resulting path
coordinates are discarded.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .navigation import (
    BODY_RADIUS_M,
    DIFFICULTIES,
    KIND_BOX,
    KIND_CYLINDER,
    MOTION_BOUNCE,
    MOTION_STATIC,
    MOTION_TREFOIL,
    Obstacle,
    SceneBank,
    _pack_instance,
    _stack_instances,
)

DEFAULT_CATALOG = Path(__file__).resolve().parents[4] / "configs/scene/p5_fixed_catalog.json"

MOTION_BY_NAME = {
    "static": MOTION_STATIC,
    "trefoil": MOTION_TREFOIL,
    "linear_bounce": MOTION_BOUNCE,
}


def load_fixed_catalog(path: Path | str = DEFAULT_CATALOG) -> dict[str, Any]:
    """Load and structurally validate the fixed-scene catalog."""

    path = Path(path)
    catalog = json.loads(path.read_text())
    scenes = catalog.get("scenes", [])
    if not scenes:
        raise ValueError("fixed scene catalog is empty")
    ids = [scene["id"] for scene in scenes]
    if len(ids) != len(set(ids)):
        raise ValueError("fixed scene ids must be unique")
    for scene in scenes:
        if scene["difficulty"] not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty in {scene['id']}: {scene['difficulty']}")
        if bool(scene["dynamic"]) != scene["id"].startswith("D"):
            raise ValueError(f"scene id/dynamic flag mismatch: {scene['id']}")
        if scene["dynamic"] and not any(
            obstacle.get("motion", "static") != "static" for obstacle in scene["obstacles"]
        ):
            raise ValueError(f"dynamic scene has no moving obstacle: {scene['id']}")
        if not scene["dynamic"] and any(
            obstacle.get("motion", "static") != "static" for obstacle in scene["obstacles"]
        ):
            raise ValueError(f"static scene contains moving obstacle: {scene['id']}")
    return catalog


def scene_obstacles(catalog: dict[str, Any], scene: dict[str, Any]) -> list[dict[str, Any]]:
    """Return shared physical boundaries followed by scene-specific geometry."""

    return [*catalog.get("boundary_obstacles", []), *scene["obstacles"]]


def catalog_obstacle(payload: dict[str, Any]) -> Obstacle:
    """Convert one explicit JSON obstacle to the shared analytic primitive."""

    shape = payload["shape"]
    if shape == "cylinder":
        kind = KIND_CYLINDER
    elif shape == "box":
        kind = KIND_BOX
    else:
        raise ValueError(f"unknown fixed-scene primitive: {shape}")
    motion_name = payload.get("motion", "static")
    if motion_name not in MOTION_BY_NAME:
        raise ValueError(f"unknown fixed-scene motion: {motion_name}")
    params = tuple(float(value) for value in payload.get("params", [0.0] * 5))
    if len(params) != 5:
        raise ValueError("fixed-scene motion params must contain five values")
    return Obstacle(
        shape=shape,
        kind=kind,
        origin=tuple(float(value) for value in payload["origin"]),
        size=tuple(float(value) for value in payload["size"]),
        motion=MOTION_BY_NAME[motion_name],
        params=params,
    )


def scene_by_id(catalog: dict[str, Any], scene_id: str) -> dict[str, Any]:
    """Return one numbered scene or fail rather than silently substituting."""

    matches = [scene for scene in catalog["scenes"] if scene["id"] == scene_id]
    if len(matches) != 1:
        raise KeyError(f"fixed scene not found: {scene_id}")
    return matches[0]


def _piecewise_path(points: list[list[float]], samples: int) -> np.ndarray:
    points_np = np.asarray(points, np.float64)
    lengths = np.linalg.norm(np.diff(points_np, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 0:
        raise ValueError("inspection path has zero length")
    distances = np.linspace(0.0, cumulative[-1], samples)
    result = []
    for distance in distances:
        index = min(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
        fraction = (distance - cumulative[index]) / lengths[index] if lengths[index] > 0 else 0.0
        result.append(points_np[index] + fraction * (points_np[index + 1] - points_np[index]))
    return np.asarray(result, np.float64)


def _signed_clearance(obstacle: Obstacle, point: np.ndarray, time: float) -> float:
    centre = np.asarray(obstacle.position(float(time)), np.float64)
    delta = point - centre
    if obstacle.kind == KIND_CYLINDER:
        radius, height, _ = obstacle.size
        radial = np.linalg.norm(delta[:2]) - radius
        vertical = abs(delta[2]) - height / 2.0
        distance = np.linalg.norm(np.maximum([radial, vertical], 0.0)) + min(
            max(radial, vertical), 0.0
        )
    else:
        q = np.abs(delta) - np.asarray(obstacle.size, np.float64)
        distance = np.linalg.norm(np.maximum(q, 0.0)) + min(max(q), 0.0)
    return float(distance - BODY_RADIUS_M)


def _occupancy_grid(
    catalog: dict[str, Any],
    scene: dict[str, Any],
    *,
    time_s: float,
    xy_resolution_m: float,
    z_resolution_m: float,
    clearance_m: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rasterize one scene snapshot without constructing or storing a route."""

    world = catalog["world"]
    z_min = float(world.get("z_min_m", 0.0))
    z_max = float(world.get("z_max_m", world.get("height_m", 5.0)))
    xs = np.arange(
        xy_resolution_m,
        float(world["length_m"]) - xy_resolution_m / 2.0,
        xy_resolution_m,
        dtype=np.float64,
    )
    ys = np.arange(
        -float(world["width_m"]) / 2.0 + xy_resolution_m,
        float(world["width_m"]) / 2.0 - xy_resolution_m / 2.0,
        xy_resolution_m,
        dtype=np.float64,
    )
    zs = np.arange(
        z_min + z_resolution_m,
        z_max - z_resolution_m / 2.0,
        z_resolution_m,
        dtype=np.float64,
    )
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    occupied = np.zeros(len(points), dtype=bool)
    for payload in scene_obstacles(catalog, scene):
        obstacle = catalog_obstacle(payload)
        centre = np.asarray(obstacle.position(float(time_s)), np.float64)
        delta = points - centre
        if obstacle.kind == KIND_CYLINDER:
            radius, height, _ = obstacle.size
            radial = np.linalg.norm(delta[:, :2], axis=1) - radius
            vertical = np.abs(delta[:, 2]) - height / 2.0
            outside = np.linalg.norm(
                np.stack([np.maximum(radial, 0.0), np.maximum(vertical, 0.0)], axis=-1),
                axis=1,
            )
            signed = outside + np.minimum(np.maximum(radial, vertical), 0.0) - BODY_RADIUS_M
        else:
            q = np.abs(delta) - np.asarray(obstacle.size, np.float64)
            signed = (
                np.linalg.norm(np.maximum(q, 0.0), axis=1)
                + np.minimum(np.max(q, axis=1), 0.0)
                - BODY_RADIUS_M
            )
        occupied |= signed < clearance_m
    return occupied.reshape(xx.shape), (xs, ys, zs)


def _nearest_grid_index(
    point: np.ndarray, axes: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[int, int, int]:
    return tuple(int(np.argmin(np.abs(axis - point[i]))) for i, axis in enumerate(axes))


def _astar_length(
    occupied: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
) -> tuple[bool, float, int]:
    """Return only route existence/length; route coordinates are intentionally discarded."""

    start_index = _nearest_grid_index(start, axes)
    goal_index = _nearest_grid_index(goal, axes)
    if occupied[start_index] or occupied[goal_index]:
        return False, math.inf, 0
    steps = [delta for delta in itertools.product((-1, 0, 1), repeat=3) if delta != (0, 0, 0)]
    scales = np.array(
        [float(np.median(np.diff(axis))) if len(axis) > 1 else 1.0 for axis in axes],
        np.float64,
    )
    costs = {delta: float(np.linalg.norm(np.asarray(delta) * scales)) for delta in steps}

    def heuristic(index):
        return float(np.linalg.norm((np.asarray(index) - np.asarray(goal_index)) * scales))

    queue = [(heuristic(start_index), 0.0, start_index)]
    distance = {start_index: 0.0}
    expanded = 0
    shape = occupied.shape
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost != distance.get(current):
            continue
        expanded += 1
        if current == goal_index:
            return True, float(current_cost), expanded
        for delta in steps:
            nxt = tuple(current[i] + delta[i] for i in range(3))
            if any(nxt[i] < 0 or nxt[i] >= shape[i] for i in range(3)) or occupied[nxt]:
                continue
            candidate = current_cost + costs[delta]
            if candidate + 1e-12 < distance.get(nxt, math.inf):
                distance[nxt] = candidate
                heapq.heappush(queue, (candidate + heuristic(nxt), candidate, nxt))
    return False, math.inf, expanded


def _straight_lane_report(
    catalog: dict[str, Any],
    scene: dict[str, Any],
    *,
    time_s: float,
    clearance_m: float,
    sample_step_m: float = 0.5,
) -> tuple[int, float]:
    """Find long constant-y/z shortcuts without saving any candidate route."""

    world = catalog["world"]
    start_x = float(world["start"][0])
    goal_x = float(world["goal"][0])
    ys = np.arange(-float(world["width_m"]) / 2.0 + 1.0, float(world["width_m"]) / 2.0, 1.0)
    z_min = float(world.get("z_min_m", 0.0))
    z_max = float(world.get("z_max_m", world.get("height_m", 5.0)))
    zs = np.arange(z_min + 0.5, z_max, 0.5)
    xs = np.arange(start_x, goal_x + sample_step_m / 2.0, sample_step_m)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    occupied = np.zeros(len(points), dtype=bool)
    for payload in scene_obstacles(catalog, scene):
        obstacle = catalog_obstacle(payload)
        centre = np.asarray(obstacle.position(float(time_s)), np.float64)
        delta = points - centre
        if obstacle.kind == KIND_CYLINDER:
            radius, height, _ = obstacle.size
            radial = np.linalg.norm(delta[:, :2], axis=1) - radius
            vertical = np.abs(delta[:, 2]) - height / 2.0
            outside = np.linalg.norm(
                np.stack([np.maximum(radial, 0.0), np.maximum(vertical, 0.0)], axis=-1),
                axis=1,
            )
            signed = outside + np.minimum(np.maximum(radial, vertical), 0.0) - BODY_RADIUS_M
        else:
            q = np.abs(delta) - np.asarray(obstacle.size, np.float64)
            signed = (
                np.linalg.norm(np.maximum(q, 0.0), axis=1)
                + np.minimum(np.max(q, axis=1), 0.0)
                - BODY_RADIUS_M
            )
        occupied |= signed < clearance_m
    clear_grid = (~occupied).reshape(xx.shape)
    full_lanes = int(np.sum(np.all(clear_grid, axis=0)))
    longest = 0.0
    for yi in range(clear_grid.shape[1]):
        for zi in range(clear_grid.shape[2]):
            run = 0
            best = 0
            for value in clear_grid[:, yi, zi]:
                run = run + 1 if value else 0
                best = max(best, run)
            longest = max(longest, best * sample_step_m)
    return full_lanes, float(longest)


def topology_report(
    catalog: dict[str, Any],
    scene: dict[str, Any],
    *,
    times_s: tuple[float, ...] | None = None,
    xy_resolution_m: float = 1.0,
    z_resolution_m: float = 0.5,
    clearance_m: float = 0.35,
) -> dict[str, Any]:
    """Validate connectivity without preserving an oracle/reference trajectory."""

    if times_s is None:
        times_s = (0.0, 10.0, 20.0, 30.0, 40.0) if scene["dynamic"] else (0.0,)
    world = catalog["world"]
    start = np.asarray(world["start"], np.float64)
    goal = np.asarray(world["goal"], np.float64)
    direct = float(np.linalg.norm(goal - start))
    snapshots = []
    for time_s in times_s:
        occupied, axes = _occupancy_grid(
            catalog,
            scene,
            time_s=float(time_s),
            xy_resolution_m=xy_resolution_m,
            z_resolution_m=z_resolution_m,
            clearance_m=clearance_m,
        )
        reachable, path_length, expanded = _astar_length(occupied, axes, start, goal)
        full_lanes, longest = _straight_lane_report(
            catalog,
            scene,
            time_s=float(time_s),
            clearance_m=clearance_m,
        )
        snapshots.append(
            {
                "time_s": float(time_s),
                "reachable": bool(reachable),
                "astar_path_length_m": float(path_length),
                "detour_ratio": float(path_length / direct) if reachable else math.inf,
                "astar_expanded": int(expanded),
                "full_length_straight_lanes": int(full_lanes),
                "longest_clear_straight_run_m": float(longest),
            }
        )
    return {
        "validation": "3D occupancy A* metrics only; path coordinates are discarded",
        "xy_resolution_m": float(xy_resolution_m),
        "z_resolution_m": float(z_resolution_m),
        "clearance_m": float(clearance_m),
        "snapshots": snapshots,
        "all_snapshots_reachable": all(row["reachable"] for row in snapshots),
        "max_full_length_straight_lanes": max(
            row["full_length_straight_lanes"] for row in snapshots
        ),
        "max_clear_straight_run_m": max(row["longest_clear_straight_run_m"] for row in snapshots),
    }


def inspect_fixed_scene(
    catalog: dict[str, Any], scene: dict[str, Any], samples: int = 601
) -> dict[str, Any]:
    """Compute review-only route and direct-line clearance diagnostics."""

    world = catalog["world"]
    start = np.asarray(world["start"], np.float64)
    goal = np.asarray(world["goal"], np.float64)
    obstacles = [catalog_obstacle(item) for item in scene_obstacles(catalog, scene)]
    route = None
    route_clearance = None
    duration = float(scene.get("inspection_duration_s", scene.get("review_duration_s", 40.0)))
    if "inspection_path" in scene:
        route = _piecewise_path(scene["inspection_path"], samples)
        times = np.linspace(0.0, duration, samples)
        route_clearance = min(
            _signed_clearance(obstacle, point, time)
            for point, time in zip(route, times, strict=True)
            for obstacle in obstacles
        )
    direct = np.linspace(start, goal, samples)
    direct_clearance = min(
        _signed_clearance(obstacle, point, 0.0) for point in direct for obstacle in obstacles
    )
    z_min = float(world.get("z_min_m", 0.0))
    z_max = float(world.get("z_max_m", world.get("height_m", 5.0)))
    low = np.array([0.0, -world["width_m"] / 2.0, z_min])
    high = np.array([world["length_m"], world["width_m"] / 2.0, z_max])
    if route is not None:
        if np.any(route < low - 1e-9) or np.any(route > high + 1e-9):
            raise ValueError(f"inspection path exits world bounds: {scene['id']}")
        if not np.allclose(route[0], start) or not np.allclose(route[-1], goal):
            raise ValueError(f"inspection path endpoints differ from start/goal: {scene['id']}")
    result = {
        "scene_id": scene["id"],
        "difficulty": scene["difficulty"],
        "dynamic": bool(scene["dynamic"]),
        "obstacles": len(obstacles),
        "field_obstacles": len(scene["obstacles"]),
        "boundary_obstacles": len(catalog.get("boundary_obstacles", [])),
        "direct_clearance_m": float(direct_clearance),
        "straight_line_blocked": bool(direct_clearance < 0.0),
        "review_duration_s": duration,
    }
    if route is not None:
        result.update(
            inspection_route_min_clearance_m=float(route_clearance),
            inspection_path_length_m=float(
                np.linalg.norm(
                    np.diff(np.asarray(scene["inspection_path"], float), axis=0), axis=1
                ).sum()
            ),
            inspection_duration_s=duration,
        )
    else:
        result["topology"] = topology_report(catalog, scene)
    return result


def validate_fixed_catalog(
    catalog: dict[str, Any],
    *,
    minimum_route_clearance_m: float = 0.35,
    maximum_clear_straight_run_m: float = 35.0,
) -> list[dict[str, Any]]:
    """Require blocked direct flight plus route-free topological feasibility.

    Legacy catalogs are still accepted through their historical
    ``inspection_path`` evidence so old review artifacts remain reconstructable.
    New route-free catalogs must instead pass A* connectivity and straight-lane
    shortcut checks without storing an oracle path.
    """

    reports = []
    for scene in catalog["scenes"]:
        report = inspect_fixed_scene(catalog, scene)
        if not report["straight_line_blocked"]:
            raise ValueError(f"fixed scene does not block the direct route: {scene['id']}")
        if "inspection_route_min_clearance_m" in report:
            if report["inspection_route_min_clearance_m"] < minimum_route_clearance_m:
                raise ValueError(
                    f"fixed scene inspection route is too tight: {scene['id']} "
                    f"({report['inspection_route_min_clearance_m']:.3f} m)"
                )
        else:
            topology = report["topology"]
            if not topology["all_snapshots_reachable"]:
                raise ValueError(
                    f"fixed scene is disconnected in a validation snapshot: {scene['id']}"
                )
            if topology["max_full_length_straight_lanes"]:
                raise ValueError(
                    f"fixed scene contains a full-length straight shortcut: {scene['id']}"
                )
            if topology["max_clear_straight_run_m"] > maximum_clear_straight_run_m:
                raise ValueError(
                    f"fixed scene contains an overlong straight shortcut: {scene['id']} "
                    f"({topology['max_clear_straight_run_m']:.1f} m)"
                )
        reports.append(report)
    return reports


def build_fixed_bank(
    catalog: dict[str, Any],
    scene_ids: list[str] | tuple[str, ...],
    *,
    validated_reports: dict[str, dict[str, Any]] | None = None,
) -> tuple[SceneBank, dict[str, Any]]:
    """Build a deterministic SceneBank from explicit numbered scenes."""

    if not scene_ids:
        raise ValueError("scene_ids must not be empty")
    selected = [scene_by_id(catalog, scene_id) for scene_id in scene_ids]
    capacity = max(len(scene_obstacles(catalog, scene)) for scene in selected)
    instances = []
    for scene in selected:
        obstacles = [catalog_obstacle(item) for item in scene_obstacles(catalog, scene)]
        instances.append(_pack_instance(obstacles, capacity))
    bank = _stack_instances(instances)
    world = catalog["world"]
    bank = bank.replace(
        start=jnp.asarray(np.tile(np.asarray(world["start"], np.float32), (len(selected), 1))),
        goal=jnp.asarray(np.tile(np.asarray(world["goal"], np.float32), (len(selected), 1))),
        difficulty=jnp.asarray(
            [DIFFICULTIES.index(scene["difficulty"]) for scene in selected], np.int32
        ),
        subtype=jnp.arange(len(selected), dtype=jnp.int32),
        subtype_names=tuple(scene["id"] for scene in selected),
        world_low=jnp.asarray(
            [0.0, -world["width_m"] / 2.0, float(world.get("z_min_m", 0.0))],
            jnp.float32,
        ),
        world_high=jnp.asarray(
            [
                world["length_m"],
                world["width_m"] / 2.0,
                float(world.get("z_max_m", world.get("height_m", 5.0))),
            ],
            jnp.float32,
        ),
    )
    reports = validated_reports or {
        report["scene_id"]: report for report in validate_fixed_catalog(catalog)
    }
    missing_reports = [scene_id for scene_id in scene_ids if scene_id not in reports]
    if missing_reports:
        raise ValueError(f"missing fixed-scene validation reports: {missing_reports}")
    manifest = {
        "version": catalog["version"],
        "status": catalog["status"],
        "source": "explicit numbered fixed-scene catalog; no random generation",
        "references": catalog["references"],
        "design_rules": catalog["design_rules"],
        "scene_ids": list(scene_ids),
        "bank_digest": bank.digest(),
        "scenes": [{**scene, "review": reports[scene["id"]]} for scene in selected],
    }
    return bank, manifest
