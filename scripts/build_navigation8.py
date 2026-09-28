#!/usr/bin/env python3
"""Build the accepted Navigation8 fixed-scene catalog.

The six primary scenes are static/dynamic × easy/medium/hard. Static geometry is
copied from SANDO's pinned easy/medium/hard forest worlds. Dynamic scenes follow
SANDO's published 50/100/200 obstacle counts, 0.65 dynamic fraction, 0.8 m cubes,
1.0--1.5 m static cylinders and trefoil motion. Sampling is conditioned only on
the P5 physical bounds and start/goal safety so moving bodies stay inside the
finite MuJoCo world.

S06 and D06 are retained as explicit 3-D extension scenes from v4. D06's long
crossbars are upgraded to vertically moving gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[2]
SANDO_SNAPSHOT = WORKSPACE / "research_dev/sando/docker/dev-workspace/upstream-93b2eed"
SANDO_WORLDS = SANDO_SNAPSHOT / "worlds"
V4_PATH = REPO / "configs/scene/p5_fixed_catalog_v4.json"
OUTPUT = REPO / "configs/scene/navigation8.json"

WORLD = {
    "length_m": 100.0,
    "width_m": 40.0,
    "z_min_m": 0.5,
    "z_max_m": 6.0,
    "start": [2.0, 0.0, 3.0],
    "goal": [98.0, 0.0, 3.0],
    "body_radius_m": 0.07,
}

BOUNDARY = [
    {
        "shape": "box",
        "origin": [50.0, -19.75, 3.25],
        "size": [50.0, 0.25, 2.75],
        "motion": "static",
        "role": "boundary",
        "name": "wall_y_min",
    },
    {
        "shape": "box",
        "origin": [50.0, 19.75, 3.25],
        "size": [50.0, 0.25, 2.75],
        "motion": "static",
        "role": "boundary",
        "name": "wall_y_max",
    },
    {
        "shape": "box",
        "origin": [0.25, 0.0, 3.25],
        "size": [0.25, 19.5, 2.75],
        "motion": "static",
        "role": "boundary",
        "name": "wall_x_min",
    },
    {
        "shape": "box",
        "origin": [99.75, 0.0, 3.25],
        "size": [0.25, 19.5, 2.75],
        "motion": "static",
        "role": "boundary",
        "name": "wall_x_max",
    },
]

SANDO_GLOBAL_TIME_SCALE = 7.9442501919
SANDO_DYNAMIC_FRACTION = 0.65
PROJECT_STATIC_BOX_FRACTION = 0.5
SANDO_STATIC_VERTICAL_FRACTION = 0.35
SANDO_VERTICAL_BOX_HALF = (0.2, 0.2, 2.0)
SANDO_HORIZONTAL_BOX_HALF = (0.2, 2.0, 0.2)
SANDO_COUNTS = {"easy": 50, "medium": 100, "hard": 200}
SANDO_STATIC_DENSITY = {"easy": 0.05, "medium": 0.10, "hard": 0.20}
SANDO_STATIC_WORLD = {
    "easy": "easy_forest.world",
    "medium": "medium_forest.world",
    "hard": "hard_forest.world",
}
PRIMARY_IDS = {"easy": "01", "medium": "02", "hard": "03"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_pose(model: ET.Element) -> list[float]:
    return [float(value) for value in (model.findtext("pose") or "0 0 0 0 0 0").split()]


def load_sando_static(difficulty: str) -> tuple[list[dict], dict]:
    """Copy one pinned SANDO forest, translating x by -3 m into P5 coordinates."""

    path = SANDO_WORLDS / SANDO_STATIC_WORLD[difficulty]
    root = ET.parse(path).getroot()
    obstacles = []
    for model in root.findall(".//world/model"):
        if model.attrib.get("name") == "ground_plane_map":
            continue
        cylinder = model.find(".//collision/geometry/cylinder")
        if cylinder is None:
            continue
        pose = _model_pose(model)
        radius = float(cylinder.findtext("radius"))
        height = float(cylinder.findtext("length"))
        obstacles.append(
            {
                "shape": "cylinder",
                "origin": [pose[0] - 3.0, pose[1], pose[2]],
                "size": [radius, height, 0.0],
                "motion": "static",
                "role": "sando_static_forest",
                "source_name": model.attrib.get("name", ""),
            }
        )
    return obstacles, {
        "world_file": str(path.relative_to(WORKSPACE)),
        "world_sha256": sha256(path),
        "x_translation_m": -3.0,
        "published_density": SANDO_STATIC_DENSITY[difficulty],
        "cylinder_count": len(obstacles),
    }


def _endpoint_safe_box(x: float, y: float, z: float, ex: float, ey: float, ez: float) -> bool:
    # Conservative swept-AABB check around both endpoint safety spheres.
    margin = 0.42
    for px, py, pz in (WORLD["start"], WORLD["goal"]):
        if abs(px - x) <= ex + margin and abs(py - y) <= ey + margin and abs(pz - z) <= ez + margin:
            return False
    return True


def _endpoint_safe_cylinder(x: float, y: float, radius: float) -> bool:
    margin = 0.42
    return all(math.hypot(x - px, y - py) > radius + margin for px, py, _ in (WORLD["start"], WORLD["goal"]))


def generate_sando_dynamic(difficulty: str, seed: int = 0) -> tuple[list[dict], dict]:
    """Generate one bounded realization using SANDO's dynamic-scene semantics.

    The moving fraction remains SANDO's 0.8 m trefoil cubes.  To preserve some
    cylindrical clutter while adding the elongated geometry visible in SANDO's
    current dynamic launcher, half of the nominal static-cylinder share is
    replaced by SANDO-sized boxes.  Those replacement boxes are static, matching
    ``launch/dyn_obstacles.launch.py``: 35% vertical pillars and 65% horizontal
    walls within the replacement subset.
    """

    total = SANDO_COUNTS[difficulty]
    dynamic_count = int(total * SANDO_DYNAMIC_FRACTION)
    static_count = total - dynamic_count
    static_box_count = (static_count + 1) // 2
    static_cylinder_count = static_count - static_box_count
    vertical_box_count = int(static_box_count * SANDO_STATIC_VERTICAL_FRACTION)
    horizontal_box_count = static_box_count - vertical_box_count
    rng = random.Random(seed)
    obstacles = []
    rejected = 0
    for index in range(total):
        for _ in range(100_000):
            x = rng.uniform(0.0, 100.0)
            y = rng.uniform(-20.0, 20.0)
            if index < dynamic_count:
                z = rng.uniform(0.5, 4.5)
                sx, sy, sz = (rng.uniform(2.0, 4.0) for _ in range(3))
                offset = rng.uniform(0.0, 3.0)
                slower_raw = rng.uniform(4.0, 6.0)
                slower = slower_raw * SANDO_GLOBAL_TIME_SCALE
                ex, ey, ez = sx / 2.0 + 0.4, sy * 3.0 / 5.0 + 0.4, sz / 2.0 + 0.4
                bounded = (
                    x - ex >= 0.5
                    and x + ex <= 99.5
                    and y - ey >= -19.5
                    and y + ey <= 19.5
                    and z - ez >= WORLD["z_min_m"]
                    and z + ez <= WORLD["z_max_m"]
                )
                if not bounded or not _endpoint_safe_box(x, y, z, ex, ey, ez):
                    rejected += 1
                    continue
                obstacles.append(
                    {
                        "shape": "box",
                        "origin": [x, y, z],
                        "size": [0.4, 0.4, 0.4],
                        "motion": "trefoil",
                        "params": [sx, sy, sz, offset, slower],
                        "role": "sando_dynamic_cube",
                        "source_index": index,
                    }
                )
            else:
                static_index = index - dynamic_count
                if static_index < static_box_count:
                    is_vertical = static_index < vertical_box_count
                    half = SANDO_VERTICAL_BOX_HALF if is_vertical else SANDO_HORIZONTAL_BOX_HALF
                    z = 2.0 if is_vertical else rng.uniform(0.7, 4.3)
                    bounded = (
                        x - half[0] >= 0.5
                        and x + half[0] <= 99.5
                        and y - half[1] >= -19.5
                        and y + half[1] <= 19.5
                        and z - half[2] >= 0.0
                        and z + half[2] <= WORLD["z_max_m"]
                    )
                    if not bounded or not _endpoint_safe_box(x, y, z, *half):
                        rejected += 1
                        continue
                    obstacles.append(
                        {
                            "shape": "box",
                            "origin": [x, y, z],
                            "size": list(half),
                            "motion": "static",
                            "role": (
                                "sando_static_vertical_box"
                                if is_vertical
                                else "sando_static_horizontal_box"
                            ),
                            "source_index": index,
                        }
                    )
                else:
                    radius = rng.uniform(1.0, 1.5)
                    bounded = (
                        x - radius >= 0.5
                        and x + radius <= 99.5
                        and y - radius >= -19.5
                        and y + radius <= 19.5
                    )
                    if not bounded or not _endpoint_safe_cylinder(x, y, radius):
                        rejected += 1
                        continue
                    obstacles.append(
                        {
                            "shape": "cylinder",
                            "origin": [x, y, 3.0],
                            "size": [radius, 6.0, 0.0],
                            "motion": "static",
                            "role": "sando_dynamic_static_cylinder",
                            "source_index": index,
                        }
                    )
            break
        else:
            raise RuntimeError(f"unable to place dynamic obstacle {index} for {difficulty}")
    return obstacles, {
        "seed": seed,
        "total_obstacles": total,
        "dynamic_obstacles": dynamic_count,
        "static_obstacles": static_count,
        "static_cylinders": static_cylinder_count,
        "static_boxes": static_box_count,
        "static_vertical_boxes": vertical_box_count,
        "static_horizontal_boxes": horizontal_box_count,
        "dynamic_fraction": SANDO_DYNAMIC_FRACTION,
        "static_box_fraction_of_static_share": PROJECT_STATIC_BOX_FRACTION,
        "static_box_vertical_fraction": SANDO_STATIC_VERTICAL_FRACTION,
        "bounded_sampling_rejections": rejected,
        "sampling_note": (
            "SANDO dynamic laws conditioned on P5 physical bounds and endpoint safety; "
            "half of the static-cylinder share replaced by SANDO-sized static boxes"
        ),
    }


def primary_scene(kind: str, difficulty: str) -> dict:
    suffix = PRIMARY_IDS[difficulty]
    if kind == "static":
        obstacles, provenance = load_sando_static(difficulty)
        return {
            "id": f"S{suffix}",
            "difficulty": difficulty,
            "title": f"SANDO {difficulty} static forest",
            "source": "SANDO",
            "dynamic": False,
            "benchmark_role": "primary",
            "obstacles": obstacles,
            "review_duration_s": 3.0,
            "layout_policy": "direct conversion of pinned SANDO forest geometry; no route reservation",
            "source_provenance": provenance,
            "topology_acceptance": {
                "require_direct_route_blocked": True,
                "require_reachable": True,
                "max_full_length_straight_lanes": 0,
                "max_clear_straight_run_m": None,
            },
        }
    obstacles, provenance = generate_sando_dynamic(difficulty, seed=0)
    return {
        "id": f"D{suffix}",
        "difficulty": difficulty,
        "title": f"SANDO {difficulty} dynamic benchmark",
        "source": "SANDO",
        "dynamic": True,
        "benchmark_role": "primary",
        "obstacles": obstacles,
        "review_duration_s": 45.0,
        "layout_policy": "SANDO dynamic benchmark laws; fixed bounded realization; no route reservation",
        "source_provenance": provenance,
        "topology_acceptance": {
            "require_direct_route_blocked": True,
            "require_reachable": True,
            "max_full_length_straight_lanes": None,
            "max_clear_straight_run_m": None,
        },
    }


def retained_extensions(v4: dict) -> list[dict]:
    retained = []
    for scene_id in ("S06", "D06"):
        scene = copy.deepcopy(next(scene for scene in v4["scenes"] if scene["id"] == scene_id))
        scene["benchmark_role"] = "3d-extension"
        scene["topology_acceptance"] = {
            "require_direct_route_blocked": True,
            "require_reachable": True,
            "max_full_length_straight_lanes": 0,
            "max_clear_straight_run_m": 35.0,
        }
        if scene_id == "D06":
            moving_index = 0
            for obstacle in scene["obstacles"]:
                if (
                    obstacle["shape"] == "box"
                    and obstacle.get("role") == "layout_core"
                    and float(obstacle["size"][1]) >= 10.0
                ):
                    low_gate = float(obstacle["origin"][2]) < 3.5
                    amplitude = 0.7 if low_gate else 0.5
                    period = 8.0 + moving_index
                    phase = 0.25 if moving_index % 2 == 0 else 0.75
                    obstacle["motion"] = "linear_bounce"
                    obstacle["params"] = [0.0, 0.0, amplitude, period, phase]
                    obstacle["role"] = "moving_crossbar"
                    moving_index += 1
            scene["title"] = "Hybrid long 3D moving gates with moving crossbars"
            scene["layout_policy"] = (
                "retained v4 3D extension; long crossbars vertically oscillate; no reserved route"
            )
            scene["moving_crossbars"] = moving_index
        retained.append(scene)
    return retained


def main() -> None:
    if not SANDO_WORLDS.is_dir():
        raise FileNotFoundError(SANDO_WORLDS)
    v4 = json.loads(V4_PATH.read_text())
    scenes = []
    for difficulty in ("easy", "medium", "hard"):
        scenes.append(primary_scene("static", difficulty))
    for difficulty in ("easy", "medium", "hard"):
        scenes.append(primary_scene("dynamic", difficulty))
    scenes.extend(retained_extensions(v4))

    output = {
        "name": "navigation8",
        "version": "navigation8-v1",
        "status": "accepted",
        "world": WORLD,
        "design_rules": [
            "Primary benchmark is exactly six scenes: static/dynamic x easy/medium/hard.",
            "Static difficulty follows SANDO 5/10/20 percent forest density using pinned world geometry.",
            "Dynamic difficulty follows SANDO 50/100/200 total obstacles with 65 percent dynamic cubes.",
            "SANDO dynamic cubes are 0.8 m and use the inherited trefoil law; they are the moving population.",
            "Half of the remaining static-cylinder share is replaced by static SANDO-style rectangular obstacles: 0.4x0.4x4.0 m vertical pillars and 0.4x4.0x0.4 m horizontal walls, with a 35/65 vertical/horizontal split inside the replacement subset.",
            "The other half of the static share remains 1.0-1.5 m radius, 6 m high cylinders to retain cylindrical clutter requested for this benchmark.",
            "Dynamic randomization is conditioned only on finite P5 world bounds and endpoint safety, then frozen at seed 0.",
            "No reference, oracle, inspection or demonstration trajectory is stored or reserved.",
            "Four physical boundary walls are visible to depth/LiDAR and participate in collision.",
            "S06 and D06 are retained as additional 3-D extension scenes; D06 long crossbars move vertically.",
            "Reachability is checked only through offline 3-D occupancy metrics; path coordinates are discarded.",
        ],
        "references": {
            **v4["references"],
            "SANDO_navigation8_alignment": {
                "repository": "research_dev/sando/docker/dev-workspace/upstream-93b2eed",
                "snapshot": "93b2eed",
                "static_worlds": SANDO_STATIC_WORLD,
                "dynamic_protocol": (
                    "50/100/200 obstacles; 0.65 moving 0.8 m trefoil cubes; "
                    "SANDO current launcher uses static 0.4x0.4x4 vertical pillars and "
                    "0.4x4x0.4 horizontal walls for the non-moving population"
                ),
                "github_main_dynamic_launcher_blob": "21bb832ca2d63406d1e99e552decdea4676a3ef0",
            },
        },
        "boundary_obstacles": BOUNDARY,
        "scenes": scenes,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT)
    for scene in scenes:
        moving = sum(obstacle.get("motion", "static") != "static" for obstacle in scene["obstacles"])
        print(
            scene["id"],
            scene["benchmark_role"],
            scene["difficulty"],
            len(scene["obstacles"]),
            f"moving={moving}",
            scene["title"],
        )


if __name__ == "__main__":
    main()
