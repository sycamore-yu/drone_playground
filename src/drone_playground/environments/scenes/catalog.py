"""Navigation8 fixed scenes, validated without an oracle/reference route.

Connectivity uses offline 3-D occupancy A*; resulting path coordinates are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from drone_playground.environments.scenes.geometry import (
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
from drone_playground.resources import resource_path

DEFAULT_CATALOG = resource_path("assets/scenes/navigation/catalog.xml")


@dataclass(frozen=True)
class NavigationCatalogScene:
    """Unique fixed cases loaded from the selected MJCF assets."""

    name: str = "navigation"
    scene_ids: tuple[str, ...] = (
        "S01",
        "S02",
        "S03",
        "S06",
        "D01",
        "D02",
        "D03",
        "D06",
    )
    catalog_path: str | None = None

    def build(self):
        path = Path(self.catalog_path or DEFAULT_CATALOG)
        if len(self.scene_ids) != len(set(self.scene_ids)):
            raise ValueError("Nominal Navigation8 cases must be unique")
        catalog = load_fixed_catalog(path)
        bank, manifest = build_fixed_bank(catalog, self.scene_ids)
        manifest.update(
            catalog_path=str(path),
            geometry_seed_role="none; unique accepted nominal cases",
        )
        return bank, manifest


MOTION_BY_NAME = {
    "static": MOTION_STATIC,
    "trefoil": MOTION_TREFOIL,
    "linear_bounce": MOTION_BOUNCE,
}


def load_fixed_catalog(path: Path | str = DEFAULT_CATALOG) -> dict[str, Any]:
    """Load and structurally validate the fixed-scene catalog."""
    path = Path(path)
    from drone_playground.environments.scenes.mjcf import load_catalog

    catalog = load_catalog(path)
    if catalog.get("version") != "navigation-v1":
        raise ValueError("Only the current navigation-v1 catalog format is supported")
    scenes = catalog.get("scenes", [])
    if not scenes:
        raise ValueError("fixed scene catalog is empty")
    ids = [scene["id"] for scene in scenes]
    if len(ids) != len(set(ids)):
        raise ValueError("fixed scene ids must be unique")
    for scene in scenes:
        if "inspection_path" in scene or "inspection_duration_s" in scene:
            raise ValueError("Historical inspection-path catalogs are unsupported")
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
    """Convert one compiled MJCF primitive to the shared runtime arrays."""
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
        asset_paths=tuple(scene.get("asset_path", "") for scene in selected),
        start=jnp.asarray(np.tile(np.asarray(world["start"], np.float32), (len(selected), 1))),
        goal=jnp.asarray(np.tile(np.asarray(world["goal"], np.float32), (len(selected), 1))),
        difficulty=jnp.asarray(
            [DIFFICULTIES.index(scene["difficulty"]) for scene in selected],
            np.int32,
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
    manifest = {
        "version": catalog["version"],
        "status": catalog["status"],
        "source": "explicit numbered fixed-scene catalog; no random generation",
        "references": catalog["references"],
        "design_rules": catalog["design_rules"],
        "scene_ids": list(scene_ids),
        "bank_digest": bank.digest(),
        "scenes": list(selected),
    }
    if validated_reports is not None:
        manifest["scenes"] = [
            {**scene, "review": validated_reports[scene["id"]]} for scene in selected
        ]
    return bank, manifest


@dataclass(frozen=True)
class CatalogNavigationScene:
    """Hydra scene adapter for the accepted Navigation8 fixed catalog.

    The catalog has four static and four dynamic scenes. Existing P5 training
    and evaluation code expects an equal number of instances per difficulty,
    so each view expands its fixed IDs deterministically: Easy and Medium repeat
    their one scene, while Hard alternates the primary hard scene and S06/D06.
    Geometry never depends on the run-role seed.
    """

    dynamic: bool
    scene_ids: tuple[str, ...]
    name: str = "navigation"
    families: tuple[str, ...] = ("navigation",)
    catalog_path: str | None = None

    def build(self, seed: int, per_difficulty: int) -> tuple[SceneBank, dict[str, Any]]:
        if per_difficulty < 1:
            raise ValueError("Navigation8 requires at least one instance per difficulty")
        catalog = load_fixed_catalog(self.catalog_path or DEFAULT_CATALOG)
        selected = [scene_by_id(catalog, scene_id) for scene_id in self.scene_ids]
        if any(bool(scene["dynamic"]) != self.dynamic for scene in selected):
            raise ValueError("Navigation8 view mixes static and dynamic scene IDs")

        grouped: dict[str, list[str]] = {difficulty: [] for difficulty in DIFFICULTIES}
        for scene in selected:
            grouped[scene["difficulty"]].append(scene["id"])
        missing = [difficulty for difficulty, ids in grouped.items() if not ids]
        if missing:
            raise ValueError(f"Navigation8 view has no scenes for difficulties: {missing}")

        expanded: list[str] = []
        for difficulty in DIFFICULTIES:
            ids = grouped[difficulty]
            expanded.extend(ids[index % len(ids)] for index in range(per_difficulty))

        bank, manifest = build_fixed_bank(catalog, expanded)
        manifest.update(
            name=self.name,
            catalog="navigation",
            view="dynamic" if self.dynamic else "static",
            accepted_scene_ids=list(self.scene_ids),
            instance_expansion=(
                "repeat fixed IDs within each difficulty; hard alternates primary and 3-D extension"
            ),
            requested_seed=int(seed),
            geometry_seed_role="none; Navigation8 geometry is fixed across run roles",
        )
        return bank, manifest
