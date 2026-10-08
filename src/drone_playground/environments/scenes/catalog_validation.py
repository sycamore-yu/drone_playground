"""Offline navigation connectivity diagnostics; never run during scene loading."""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any

import numpy as np

from drone_playground.environments.scenes.catalog import catalog_obstacle, scene_obstacles
from drone_playground.environments.scenes.geometry import BODY_RADIUS_M, KIND_CYLINDER, Obstacle


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
                np.stack(
                    [np.maximum(radial, 0.0), np.maximum(vertical, 0.0)],
                    axis=-1,
                ),
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
                np.stack(
                    [np.maximum(radial, 0.0), np.maximum(vertical, 0.0)],
                    axis=-1,
                ),
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
    """Compute direct-line clearance and route-free connectivity diagnostics."""
    world = catalog["world"]
    start = np.asarray(world["start"], np.float64)
    goal = np.asarray(world["goal"], np.float64)
    obstacles = [catalog_obstacle(item) for item in scene_obstacles(catalog, scene)]
    duration = float(scene.get("review_duration_s", 40.0))
    direct = np.linspace(start, goal, samples)
    direct_clearance = min(
        _signed_clearance(obstacle, point, 0.0) for point in direct for obstacle in obstacles
    )
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
    result["topology"] = topology_report(catalog, scene)
    return result


def validate_fixed_catalog(
    catalog: dict[str, Any],
    *,
    maximum_clear_straight_run_m: float = 35.0,
) -> list[dict[str, Any]]:
    """Require blocked direct flight plus route-free topological feasibility."""
    reports = []
    for scene in catalog["scenes"]:
        report = inspect_fixed_scene(catalog, scene)
        acceptance = scene.get("topology_acceptance", {})
        require_direct_blocked = acceptance.get("require_direct_route_blocked", True)
        require_reachable = acceptance.get("require_reachable", True)
        max_full_lanes = acceptance.get("max_full_length_straight_lanes", 0)
        max_straight_run = acceptance.get("max_clear_straight_run_m", maximum_clear_straight_run_m)
        if require_direct_blocked and not report["straight_line_blocked"]:
            raise ValueError(f"fixed scene does not block the direct route: {scene['id']}")
        topology = report["topology"]
        if require_reachable and not topology["all_snapshots_reachable"]:
            raise ValueError(f"fixed scene is disconnected in a validation snapshot: {scene['id']}")
        if (
            max_full_lanes is not None
            and topology["max_full_length_straight_lanes"] > max_full_lanes
        ):
            raise ValueError(f"fixed scene contains a full-length straight shortcut: {scene['id']}")
        if max_straight_run is not None and topology["max_clear_straight_run_m"] > max_straight_run:
            raise ValueError(
                f"fixed scene contains an overlong straight shortcut: {scene['id']} "
                f"({topology['max_clear_straight_run_m']:.1f} m)"
            )
        reports.append(report)
    return reports
