"""Load and verify the exact external conditions of a named benchmark."""

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def protocol_identity(config):
    selected = config.get("evaluation", {}).get("protocol")
    if not selected:
        return None
    path = _path(selected)
    raw = path.read_bytes()
    specification = yaml.safe_load(raw)
    if specification.get("name") != "navigation" or specification.get("version") not in (1, 2):
        raise ValueError("Unsupported benchmark protocol version")
    task, scene = config["env"]["task"], config["env"]["scene"]
    if task["name"] not in ("navigation", "pointcloud_avoidance", "pointcloud_navigation", "depth_navigation"):
        raise ValueError("Navigation protocol requires a navigation task")
    catalog = _path(scene.get("catalog_path") or specification["catalog"])
    if hashlib.sha256(catalog.read_bytes()).hexdigest() != specification["catalog_sha256"]:
        raise ValueError("Benchmark catalog geometry differs from the locked identity")
    for field, expected in (("duration", "duration_s"), ("goal_radius", "goal_radius_m")):
        if float(task[field]) != float(specification[expected]):
            raise ValueError(f"Benchmark protocol differs on task.{field}")
    if "body_radius" in task and float(task["body_radius"]) != specification["body_radius_m"]:
        raise ValueError("Benchmark protocol differs on body collision radius")
    limits = config["method"].get("limits")
    if specification["version"] == 2 and limits is not None:
        if limits["max_velocity_mps"] != specification["nominal_max_velocity_mps"]:
            raise ValueError("Benchmark protocol differs on nominal planner velocity limit")
    chosen = scene.get("scene_ids", specification["scene_ids"])
    if not chosen or not set(chosen) <= set(specification["scene_ids"]):
        raise ValueError("Benchmark contains an undeclared scene identity")
    return dict(
        name=specification["name"],
        version=specification["version"],
        path=str(selected),
        sha256=hashlib.sha256(raw).hexdigest(),
        catalog_sha256=specification["catalog_sha256"],
    )
