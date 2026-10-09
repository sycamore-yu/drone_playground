"""Construct configured methods and evaluate them under an explicit protocol."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from drone_playground.simulation.environment import Environment
from drone_playground.simulation.records import acceptance, atomic_json, write_episodes
from drone_playground.simulation.runner import rollout, rollout_host, save_evaluation

_NAVIGATION_SCENES = ["S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06"]


def create_environment(config: DictConfig, *, num_envs=None, scene=None) -> Environment:
    """Construct the physical input from the chosen method rather than its task name."""
    simulation = OmegaConf.to_container(config.simulation, resolve=True)
    simulation["num_envs"] = config.simulation.num_envs if num_envs is None else num_envs
    implementation = get_class(config.method.implementation._target_)
    action = OmegaConf.to_container(config.method, resolve=True).get("action", {})
    level = None
    if not implementation.trainable:
        level = get_class(config.controller._target_).output_level
    return Environment(
        task=config.task.name,
        scene=config.scene.name if scene is None else scene,
        reference=config.task.reference,
        duration=config.task.duration,
        reference_seed=config.task.reference_seed,
        task_config=OmegaConf.to_container(config.task, resolve=True).get("options", {}),
        sensor=config.sensor.name,
        sensor_config=OmegaConf.to_container(config.sensor.config, resolve=True),
        point_count=config.sensor.point_count,
        action=action,
        control_level=level,
        observation_config=OmegaConf.to_container(config.observation, resolve=True),
        **simulation,
    )


def create_method(config):
    """Use Hydra's recursive construction instead of a second method registry."""
    return instantiate(config.method.implementation, _convert_="all")


def next_evaluation_directory(run_directory):
    """Allocate a new independent evaluation without overwriting earlier attempts."""
    root = Path(run_directory) / "eval"
    root.mkdir(parents=True, exist_ok=True)
    number = (
        max((int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdecimal()), default=0)
        + 1
    )
    destination = root / f"{number:03d}"
    destination.mkdir()
    return destination


def evaluate(
    config, directory, *, actor=None, parameters=None, evaluation_index=0, environments=None
):
    """Save one flat episode table and complete per-scene statistics for an evaluation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    protocol = config.benchmark.protocol
    if protocol not in {"learning", "s6", "diagnostic"}:
        raise ValueError("Evaluation protocol must be learning, s6 or diagnostic")
    task = config.task.name
    count = config.benchmark.episodes
    if count is None:
        count = (
            config.simulation.num_envs
            if config.mode == "sim"
            else (25 if task == "navigation" else 100)
        )
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Evaluation episode count must be a positive integer")
    scenes = list(
        config.benchmark.scenes
        or (
            _NAVIGATION_SCENES
            if task == "navigation" and config.mode != "sim"
            else [config.scene.name]
        )
    )
    seed_base = config.benchmark.seed_base
    if seed_base is None:
        seed_base = {"sim": 100_000, "checkpoint_eval": 1_000_000, "benchmark": 2_000_000}[
            config.mode
        ]
    if config.mode == "benchmark" and seed_base < 2_000_000:
        raise ValueError("Frozen benchmark seeds must be in the separate >=2,000,000 partition")
    if config.mode == "checkpoint_eval" and not 1_000_000 <= seed_base < 2_000_000:
        raise ValueError("Checkpoint evaluation uses the [1,000,000, 2,000,000) seed partition")
    own_cache = environments is None
    environments = {} if environments is None else environments
    if "method" not in environments:
        environments["method"] = create_method(config)
    method = environments["method"]
    reports, all_rows, recorded, decisions = {}, [], {}, []
    keep_traces = bool(config.benchmark.record_trajectories or config.benchmark.record_replay)
    label = config.method.name
    if not method.trainable:
        label = f"{label}:{type(method.controller).__name__}"
    try:
        for scene_index, scene in enumerate(scenes):
            seed = int(seed_base + 10_000 * config.seed + 100 * evaluation_index + scene_index)
            batch = count if method.execution == "jax" else 1
            key = (scene, batch)
            if key not in environments:
                environments[key] = create_environment(config, num_envs=batch, scene=scene)
            env = environments[key]
            scene_rows, replays = [], []
            repetitions = 1 if method.execution == "jax" else count
            for episode_index in range(repetitions):
                plans = None
                if method.execution == "jax":
                    episodes, traces = rollout(
                        env,
                        seed=seed,
                        method=method,
                        actor=actor,
                        parameters=parameters,
                        method_name=label,
                        chunk_steps=config.benchmark.chunk_steps,
                        record=keep_traces,
                    )
                else:
                    episodes, traces, plans, iteration = rollout_host(
                        env,
                        method,
                        seed=seed + 1000 * episode_index,
                        method_name=label,
                        record=keep_traces,
                    )
                    episodes[0]["episode"] = episode_index
                    decisions.extend(
                        {"scene": scene, "episode": episode_index, **x} for x in iteration
                    )
                scene_rows.extend(episodes)
                if config.benchmark.record_trajectories:
                    recorded.update(
                        {
                            f"{scene}__{episode_index:04d}__{name}": value
                            for name, value in traces.items()
                        }
                    )
                if config.benchmark.record_replay:
                    saved = save_evaluation(
                        directory,
                        env,
                        episodes,
                        traces,
                        report={},
                        plans=plans,
                        name=scene,
                        replay_episodes=config.benchmark.replay_episodes,
                        write_tables=False,
                        record_trajectories=False,
                    )
                    replays.extend(saved["replays"])
            outcomes = Counter(row["event"] for row in scene_rows)
            report = (
                acceptance(task, scene_rows)
                if protocol == "learning"
                else {
                    "episodes": len(scene_rows),
                    "successes": outcomes["SUCCESS"],
                    "success_rate": outcomes["SUCCESS"] / len(scene_rows),
                    "successful_position_rmse": None,
                    "passed": None,
                }
            )
            report.update(
                task=task,
                task_contract=env.task.contract,
                scene=scene,
                geometry=env.scene.geometry_identity,
                method=label,
                outcomes=dict(outcomes),
                replays=replays,
                execution="ideal_tracking" if env.control_level == "ideal" else "crazyflow",
            )
            reports[scene] = report
            all_rows.extend(scene_rows)
            write_episodes(directory / "episodes.csv", all_rows)
            atomic_json(
                directory / "report.json",
                {"mode": config.mode, "reports": reports, "complete": False, "passed": False},
            )
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "evaluation": config.mode,
                        "episodes": len(scene_rows),
                        "successes": outcomes["SUCCESS"],
                        "passed": report["passed"],
                    }
                ),
                flush=True,
            )
    finally:
        if own_cache:
            method.close()
    groups = OmegaConf.to_container(config.benchmark.primary_scenes, resolve=True)
    static_scenes, dynamic_scenes = groups["static"], groups["dynamic"]
    primary = static_scenes + dynamic_scenes if task == "navigation" else scenes
    complete = bool(primary) and set(primary) <= reports.keys()
    summary = {
        "mode": config.mode,
        "protocol": protocol,
        "training_seed": config.seed,
        "seed_base": int(seed_base),
        "evaluation_index": evaluation_index,
        "reports": reports,
        "complete": True,
        "passed": None,
    }
    if protocol == "learning":
        summary["passed"] = complete and all(reports[s]["passed"] for s in primary)
        if task == "navigation":
            summary["static_passed"] = all(
                reports.get(s, {}).get("passed", False) for s in static_scenes
            )
            summary["dynamic_passed"] = all(
                reports.get(s, {}).get("passed", False) for s in dynamic_scenes
            )
    elif protocol == "s6":
        static_count = sum(reports.get(s, {}).get("episodes", 0) for s in static_scenes)
        dynamic_count = sum(reports.get(s, {}).get("episodes", 0) for s in dynamic_scenes)
        static_passed = static_count >= 10 and all(
            reports.get(s, {}).get("successes", 0) >= 1 for s in static_scenes
        )
        dynamic_passed = dynamic_count >= 10 and all(
            reports.get(s, {}).get("episodes", 0) > 0 for s in dynamic_scenes
        )
        summary.update(
            criterion="S6",
            static_episodes=static_count,
            dynamic_episodes=dynamic_count,
            static_passed=static_passed,
            dynamic_passed=dynamic_passed,
            passed=static_passed and dynamic_passed,
        )
        for scene, report in reports.items():
            report["criterion"] = "S6_static_success" if scene in static_scenes else "diagnostic"
            report["passed"] = report["successes"] >= 1 if scene in static_scenes else None
    if recorded:
        np.savez_compressed(directory / "trajectories.npz", **recorded)
    if decisions:
        atomic_json(directory / "decisions.json", {"decisions": decisions})
    atomic_json(directory / "report.json", summary)
    return summary
