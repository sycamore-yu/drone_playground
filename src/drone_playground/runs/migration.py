"""Explicit, copy-only migration of trusted historical experiment configurations.

Current execution accepts schema 3. This module is used by the migration tool
and numerical regression fixtures, never as an implicit configuration alias.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

MODULE_MOVES = {
    "drone_playground.policies.neural": "drone_playground.methods.neural",
    "drone_playground.policies.planning": "drone_playground.methods.planners.reference",
    "drone_playground.policies.native_planner": "drone_playground.integrations.native_planner",
    "drone_playground.controllers.factory": "drone_playground.methods.optimal_control.factory",
    "drone_playground.controllers.lsy_mpc": "drone_playground.methods.optimal_control.lsy_mpc",
    "drone_playground.controllers.sampling": "drone_playground.methods.optimal_control.sampling",
    "drone_playground.controllers.lsy_upstream": "drone_playground.methods.optimal_control.lsy_upstream",
    "drone_playground.controllers": "drone_playground.execution.controllers",
    "drone_playground.dynamics": "drone_playground.models",
    "drone_playground.learning.networks": "drone_playground.networks.policies",
    "drone_playground.learning.perception": "drone_playground.networks.encoders",
    "drone_playground.learning.pointcloud_network": "drone_playground.networks.pointcloud",
    "drone_playground.learning.pointcloud_objective": "drone_playground.learning.objectives.pointcloud",
    "drone_playground.learning.shac": "drone_playground.learning.algorithms.shac",
    "drone_playground.learning.dva": "drone_playground.learning.algorithms.dva",
    "drone_playground.learning.lotf_bptt": "drone_playground.learning.algorithms.lotf_bptt",
    "drone_playground.learning.pointcloud_bptt": "drone_playground.learning.algorithms.pointcloud_bptt",
    "drone_playground.tasks.scenes.fixed_navigation": "drone_playground.environments.scenes.catalog",
    "drone_playground.tasks.scenes": "drone_playground.environments.scenes",
    "drone_playground.tasks.sensors": "drone_playground.environments.sensors",
    "drone_playground.tasks.observations": "drone_playground.environments.observations",
    "drone_playground.tasks": "drone_playground.environments.tasks",
    "drone_playground.runs.rscope_io": "drone_playground.visualization.rscope_io",
    "drone_playground.runs.navigation_scene": "drone_playground.visualization.navigation_scene",
    "drone_playground.runs.lotf_scene": "drone_playground.visualization.lotf_scene",
}


def relocated_module(name: str) -> str:
    for old, new in sorted(MODULE_MOVES.items(), key=lambda item: -len(item[0])):
        if name == old or name.startswith(old + "."):
            return (
                (new + name[len(old) :])
                .replace("Navigation8CatalogScene", "NavigationCatalogScene")
                .replace("Navigation8Scene", "NavigationScene")
            )
    return name


def relocate_targets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: relocate_targets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_targets(item) for item in value]
    if isinstance(value, str) and value.startswith("drone_playground."):
        return relocated_module(value)
    return value


def _environment_name(task: dict, scene: dict) -> str:
    name = task["name"]
    if name == "navigation":
        return "navigation/dynamic" if task.get("dynamic", False) else "navigation/static"
    if name in ("figure8", "random"):
        return "tracking" if name == "figure8" else "tracking/random"
    if name in ("lotf_hover", "lotf_tracking"):
        return "paper/" + name
    if name == "pointcloud_avoidance":
        return (
            "paper/pointcloud_navigation"
            if scene.get("name") in ("navigation8", "navigation")
            else "paper/pointcloud_flight"
        )
    return name


def migrate_v2(config: dict) -> dict:
    """Convert exactly the supplied resolved values, without applying new defaults."""
    value = copy.deepcopy(config.get("components", config))
    if value.get("config_version") == 3:
        return value
    if isinstance(value.get("dynamics"), str):
        return migrate_flat_checkpoint(value)
    if not isinstance(value.get("dynamics"), dict):
        raise ValueError("Version 2 migration requires the full resolved component configuration")
    original_digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    value = relocate_targets(value)
    policy = value.pop("policy")
    controller = value.pop("controller")
    dynamics = value.pop("dynamics")
    if dynamics.get("forward", "").startswith("lotf_"):
        dynamics.setdefault("_target_", "drone_playground.models.lotf.LOTFModel")
    task = value.pop("task")
    scene = value.pop("scene")
    observation = value.pop("observation")
    sensor = observation.pop("sensor", {"name": "none"})
    algorithm = value["algorithm"]
    gradient = {"transition": dynamics.pop("backward", "direct")}
    if "decay_rate" in dynamics:
        gradient["decay_rate"] = dynamics.pop("decay_rate")
    algorithm["gradient"] = gradient
    generator = policy.pop("planning", None)
    if generator is not None:
        task["reference_generator"] = generator
    execution = {"controller": controller, "dynamics": dynamics, "tracker": None}
    implementation = policy["name"]
    trainable = implementation not in ("native_ego", "native_super", "fixed_trajectory")
    method_name = {
        "neural": algorithm["name"],
        "lotf_mlp": "lotf",
        "pointcloud_recurrent": "pointcloud_flight",
        "native_ego": "ego_planner",
        "native_super": "super",
    }.get(implementation, implementation)
    if controller["name"] in ("attitude_mpc", "sampling_mpc"):
        method_name = implementation = controller["name"]
        trainable = False
        decision = controller
        execution["controller"] = {"name": "crazyflow_attitude"}
        policy["output"] = "attitude_thrust"
        policy["decision"] = decision
    elif controller["name"] == "trajectory_tracking":
        execution["tracker"] = controller
        execution["controller"] = {"name": "crazyflow_attitude"}
    policy.update(name=method_name, implementation=implementation, trainable=trainable)
    execution["command"] = policy["output"]
    env_name = _environment_name(task, scene)
    if scene.get("name") == "navigation8":
        scene["name"] = "navigation"
    for key in ("catalog_path", "verification_path"):
        if scene.get(key):
            scene[key] = scene[key].replace(
                "configs/scene/navigation8.json", "assets/scenes/navigation/catalog.json"
            )
    value["method"] = policy
    value["env"] = dict(
        name=env_name,
        task=task,
        scene=scene,
        sensor=sensor,
        observation=observation,
        execution=execution,
    )
    value["runtime"] = dict(
        device=value["training"].pop("device", "cpu"),
        backend="jax" if trainable else "host",
        timing="synchronous",
        action_delay_steps=0,
    )
    value["training"].setdefault("development_episodes", 32)
    value["mode"] = {"evaluate": "eval", "simulate": "play"}.get(value["mode"], value["mode"])
    value["config_version"] = 3
    value.setdefault("visualization", {"mode": "rscope"})
    value.setdefault("evaluation", {})
    if value["evaluation"].get("environment") == "experiment":
        value["evaluation"]["environment"] = "config"
    value["migration"] = dict(source_schema=2, source_config_sha256=original_digest)
    return value


def migrate_flat_checkpoint(config):
    """Explicit P1--P4 flat-configuration migration, using the audited legacy map.

    Only the historical state-control tasks are admitted. Numerical hyperparameters
    override present defaults; the saved run had no additional transport queue.
    This is called by the copy-only artifact migration tool, never at runtime.
    """
    from drone_playground.composition import compose_method

    environments = {"figure8": "tracking", "random": "tracking/random", "racing": "racing"}
    task, algorithm = config.get("task"), config.get("algorithm")
    if task not in environments or algorithm not in ("ppo", "apg", "shac"):
        raise ValueError(
            "Flat checkpoint migration requires an audited P1--P4 state-control recipe"
        )
    result = compose_method("learning/" + algorithm, environments[task])
    result["env"]["execution"]["dynamics"].update(forward=config["dynamics"], drone=config["drone"])
    for group in ("algorithm", "network", "training"):
        for field in result[group]:
            if field in config and field != "name":
                result[group][field] = copy.deepcopy(config[field])
    original_task = result["env"]["task"]
    for field in original_task:
        if field in config and field != "name":
            original_task[field] = copy.deepcopy(config[field])
    original_task["numerical_guard"] = config.get("numerical_guard", False)
    original_task["reference_count"] = config.get("reference_count", 256)
    network = result["network"]
    network.update(
        hidden_sizes=config.get("hidden_sizes", [64, 64]),
        normalize_observations=config.get("normalize_observations", False),
    )
    if algorithm == "ppo":
        network.update(
            distribution_type=config.get("distribution_type", "tanh_normal"),
            init_noise_std=config.get("init_noise_std", 0.367879),
        )
    else:
        network["layer_norm"] = config.get("layer_norm", True)
    result["runtime"].update(
        device=config.get("device", "cpu"), action_delay_steps=0, action_delay_ms=None
    )
    result["source"] = (
        "Explicit copy of historical P1--P4 flat checkpoint; audited legacy field mapping"
    )
    result["migration"] = dict(
        source_schema=1,
        source_config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        source_fields=sorted(config),
        transport_delay="historical zero additional delay",
    )
    return result


def require_current(config: dict) -> dict:
    """Validate the artifact version before a current runtime consumes its fields."""
    value = config.get("components", config)
    if value.get("config_version") != 3:
        raise ValueError(
            "旧配置需要显式迁移：python scripts/tools/migrate_artifact.py 源文件 目标目录"
        )
    return copy.deepcopy(value)


def write_migrated_config(source: Path, destination: Path) -> Path:
    """Write a new configuration; refuse every existing destination."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    raw = json.loads(source.read_text())
    value = migrate_v2(raw.get("config", raw))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return destination


class _RelocatedUnpickler(pickle.Unpickler):
    """Load trusted local experiment files with explicitly renamed Python classes."""

    def find_class(self, module, name):
        return super().find_class(relocated_module(module), name)


def migrate_checkpoint(source: Path, destination_directory: Path) -> Path:
    """Copy an authenticated, trusted checkpoint into schema 3; never edit its run.

    Pickle files must come from the user's trusted experiments. Class relocation
    preserves array/optimizer contents. Selected older checkpoints are copied into
    the same new artifact directory and selection paths are rewritten explicitly.
    """
    source = Path(source).resolve()
    destination_directory = Path(destination_directory).resolve()
    if destination_directory.exists():
        raise FileExistsError(destination_directory)
    source_meta = json.loads(source.with_suffix(".json").read_text())
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != source_meta.get("sha256"):
        raise ValueError("Checkpoint digest differs from its metadata")
    original_config = source_meta["config"]
    config = migrate_v2(original_config)
    import io

    payload = _RelocatedUnpickler(io.BytesIO(raw)).load()
    new_meta = copy.deepcopy(source_meta)
    if "components" in original_config:
        native = copy.deepcopy(original_config)
        native["components"] = config
        native["config_version"] = 3
        new_meta["config"] = native
    else:
        new_meta["config"] = config
    destination_directory.mkdir(parents=True, exist_ok=False)
    destination = destination_directory / source.name
    try:
        encoded = pickle.dumps(payload)
        destination.write_bytes(encoded)
        new_meta["config_version"] = 3
        new_meta["sha256"] = hashlib.sha256(encoded).hexdigest()
        new_meta["migration"] = dict(
            source_checkpoint=str(source),
            source_sha256=digest,
            source_config_version=original_config.get("config_version", 2),
        )
        selected = new_meta.get("selection")
        if selected and selected.get("checkpoint"):
            previous = Path(selected["checkpoint"]).resolve()
            if previous == source:
                selected["checkpoint"] = str(destination)
            else:
                older = migrate_checkpoint(previous, destination_directory / "selected")
                selected["checkpoint"] = str(older)
        destination.with_suffix(".json").write_text(
            json.dumps(new_meta, indent=2, ensure_ascii=False) + "\n"
        )
        (destination_directory / "source-metadata.json").write_text(
            json.dumps(source_meta, indent=2, ensure_ascii=False) + "\n"
        )
    except BaseException:
        shutil.rmtree(destination_directory)
        raise
    return destination
