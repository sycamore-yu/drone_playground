"""Fixed, numbered P5 navigation scenes for manual review and later freezing.

The random density generator remains readable for historical v2 experiment
reconstruction, but new benchmark scene design is authored here from the
explicit catalog in configs/scene/p5_fixed_catalog.json.

Every catalog scene contains a human inspection route. That route is only an
offline validation/replay aid: it is never included in the environment
observation and is not available to PPO, D.VA, EGO-Planner or SUPER.
"""

from __future__ import annotations

import json
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


def inspect_fixed_scene(
    catalog: dict[str, Any], scene: dict[str, Any], samples: int = 601
) -> dict[str, Any]:
    """Compute review-only route and direct-line clearance diagnostics."""

    world = catalog["world"]
    start = np.asarray(world["start"], np.float64)
    goal = np.asarray(world["goal"], np.float64)
    obstacles = [catalog_obstacle(item) for item in scene["obstacles"]]
    route = _piecewise_path(scene["inspection_path"], samples)
    duration = float(scene["inspection_duration_s"])
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
    low = np.array([0.0, -world["width_m"] / 2.0, 0.0])
    high = np.array([world["length_m"], world["width_m"] / 2.0, world["height_m"]])
    if np.any(route < low - 1e-9) or np.any(route > high + 1e-9):
        raise ValueError(f"inspection path exits world bounds: {scene['id']}")
    if not np.allclose(route[0], start) or not np.allclose(route[-1], goal):
        raise ValueError(f"inspection path endpoints differ from start/goal: {scene['id']}")
    return {
        "scene_id": scene["id"],
        "difficulty": scene["difficulty"],
        "dynamic": bool(scene["dynamic"]),
        "obstacles": len(obstacles),
        "direct_clearance_m": float(direct_clearance),
        "straight_line_blocked": bool(direct_clearance < 0.0),
        "inspection_route_min_clearance_m": float(route_clearance),
        "inspection_path_length_m": float(
            np.linalg.norm(
                np.diff(np.asarray(scene["inspection_path"], float), axis=0), axis=1
            ).sum()
        ),
        "inspection_duration_s": duration,
    }


def validate_fixed_catalog(
    catalog: dict[str, Any], *, minimum_route_clearance_m: float = 0.35
) -> list[dict[str, Any]]:
    """Require each candidate to block direct flight while retaining a known route."""

    reports = []
    for scene in catalog["scenes"]:
        report = inspect_fixed_scene(catalog, scene)
        if not report["straight_line_blocked"]:
            raise ValueError(f"fixed scene does not block the direct route: {scene['id']}")
        if report["inspection_route_min_clearance_m"] < minimum_route_clearance_m:
            raise ValueError(
                f"fixed scene inspection route is too tight: {scene['id']} "
                f"({report['inspection_route_min_clearance_m']:.3f} m)"
            )
        reports.append(report)
    return reports


def build_fixed_bank(
    catalog: dict[str, Any], scene_ids: list[str] | tuple[str, ...]
) -> tuple[SceneBank, dict[str, Any]]:
    """Build a deterministic SceneBank from explicit numbered scenes."""

    if not scene_ids:
        raise ValueError("scene_ids must not be empty")
    selected = [scene_by_id(catalog, scene_id) for scene_id in scene_ids]
    capacity = max(len(scene["obstacles"]) for scene in selected)
    instances = []
    for scene in selected:
        obstacles = [catalog_obstacle(item) for item in scene["obstacles"]]
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
        world_low=jnp.asarray([0.0, -world["width_m"] / 2.0, 0.0], jnp.float32),
        world_high=jnp.asarray(
            [world["length_m"], world["width_m"] / 2.0, world["height_m"]], jnp.float32
        ),
    )
    reports = {report["scene_id"]: report for report in validate_fixed_catalog(catalog)}
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
