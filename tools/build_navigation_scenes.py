"""Build deterministic MuJoCo-native Forest, Maze, and Dense flight benchmarks.

The layouts reproduce FlightBench-style task families and difficulty progression,
not FlightBench's original geometry or benchmark measurements. The generated
MJCF assets are checked in; benchmarking never resamples their obstacle geometry.
"""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

FAMILIES = ("forest", "maze", "dense", "dynamic_dense")
LEVELS = ("easy", "medium", "hard")
OUTPUT = Path(__file__).resolve().parents[1] / "assets/scenes/navigation"


def _numbers(*values):
    return " ".join(f"{value:.5f}" for value in values)


def _geom(world, name, *, pos, kind, size, rgba, motion=None):
    """Add fixed geometry or a native MuJoCo mocap obstacle."""
    attributes = {"name": name, "pos": _numbers(*pos)}
    if motion is not None:
        attributes.update(mocap="true", user=_numbers(*motion))
    body = ET.SubElement(world, "body", attributes)
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_geom",
        type=kind,
        size=_numbers(*size),
        rgba=rgba,
    )


def _forest(world, rng, difficulty):
    """Increase trunk density and branch occlusion while keeping open end zones."""
    count = (40, 76, 125)[difficulty]
    for index in range(count):
        x = rng.uniform(9.0, 91.0)
        y = rng.uniform(-15.0, 15.0)
        radius = rng.uniform(0.32, 0.75)
        kind = "cylinder" if index % 5 else "capsule"
        _geom(
            world,
            f"tree_{index:03d}",
            pos=(x, y, 3.0),
            kind=kind,
            size=(radius, 3.0 if kind == "cylinder" else 2.65),
            rgba="0.35 0.34 0.26 1",
        )
    return count


def _maze(world, rng, difficulty):
    """Alternate full-height gates to force a traversable serpentine route."""
    walls = (3, 5, 7)[difficulty]
    opening = (5.0, 3.4, 2.6)[difficulty]
    count = 0
    for index in range(walls):
        x = 15.0 + (70.0 * index / max(1, walls - 1))
        center = (9.0 if index % 2 == 0 else -9.0) + rng.uniform(-0.4, 0.4)
        bounds = ((-19.8, center - opening / 2), (center + opening / 2, 19.8))
        for side, (low, high) in enumerate(bounds):
            half = (high - low) / 2
            _geom(
                world,
                f"wall_{index:02d}_{side}",
                pos=(x, (low + high) / 2, 3.0),
                kind="box",
                size=(0.45, half, 3.0),
                rgba="0.42 0.48 0.58 1",
            )
            count += 1
    return count


def _dense(world, rng, difficulty, *, dynamic=False):
    """Place many native collision shapes; dynamic variants use body.user motion."""
    count = (50, 95, 160)[difficulty] if dynamic else (65, 115, 180)[difficulty]
    for index in range(count):
        x = rng.uniform(9.0, 91.0)
        y = rng.uniform(-14.0, 14.0)
        moving = dynamic and index < (count * 3 // 5)
        kind = "cylinder" if index % 3 == 0 and not moving else "box"
        if kind == "cylinder":
            size = (rng.uniform(0.4, 0.9), 3.0)
            z = 3.0
        elif moving:
            size = (rng.uniform(0.3, 0.5), rng.uniform(0.3, 0.5), rng.uniform(0.3, 0.5))
            z = rng.uniform(1.8, 4.2)
        else:
            size = (
                rng.uniform(0.25, 0.7),
                rng.uniform(0.35, 1.25),
                rng.uniform(0.5, 2.0),
            )
            z = 3.0
        motion = (
            (
                2.0,
                rng.uniform(-0.9, 0.9),
                rng.uniform(-0.9, 0.9),
                rng.uniform(-0.2, 0.2),
                rng.uniform(3.5, 7.5),
                rng.uniform(0.0, 2.0),
            )
            if moving
            else None
        )
        _geom(
            world,
            f"obstacle_{index:03d}",
            pos=(x, y, z),
            kind=kind,
            size=size,
            rgba="0.91 0.43 0.24 1" if moving else "0.37 0.5 0.58 1",
            motion=motion,
        )
    return count


def build(family, level):
    """Generate a single fixed suite case without importing a simulator."""
    if family not in FAMILIES or level not in LEVELS:
        raise ValueError("Unknown navigation benchmark family or difficulty")
    difficulty = LEVELS.index(level)
    rng = random.Random(3901 + FAMILIES.index(family) * 100 + difficulty)
    name = f"{family}_{level}"
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "include", file="boundary.xml")
    world = ET.SubElement(root, "worldbody")
    if family == "forest":
        count = _forest(world, rng, difficulty)
    elif family == "maze":
        count = _maze(world, rng, difficulty)
    else:
        count = _dense(world, rng, difficulty, dynamic=family == "dynamic_dense")
    metadata = ET.SubElement(root, "custom")
    ET.SubElement(
        metadata,
        "text",
        name="benchmark_metadata",
        data=json.dumps(
            {
                "family": family,
                "difficulty": level,
                "geometries": count,
                "generator": "flight-task-mjcf-v1",
                "source": "FlightBench-inspired geometry and task family; independently generated",
                "goal": [98.0, 0.0, 3.0],
                "start": [2.0, 0.0, 3.0],
            },
            separators=(",", ":"),
        ),
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for scene_family in FAMILIES:
        for scene_level in LEVELS:
            (OUTPUT / f"{scene_family}_{scene_level}.xml").write_text(
                build(scene_family, scene_level)
            )
