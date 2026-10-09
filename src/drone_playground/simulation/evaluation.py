"""Configured Episode evaluation with task-specific acceptance and complete records."""

import json
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from drone_playground.simulation.environment import Environment
from drone_playground.simulation.records import acceptance, atomic_json
from drone_playground.simulation.ros_planner import RosPlanner
from drone_playground.simulation.runner import rollout, rollout_ros, save_evaluation

_NAVIGATION_SCENES = ["S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06"]


def create_environment(
    config: DictConfig, *, num_envs: int | None = None, scene: str | None = None
) -> Environment:
    """Construct the single physical contract used by every execution mode."""
    if config.method.action_delay != 0:
        raise ValueError("The initial execution recipes declare zero action delay")
    simulation = OmegaConf.to_container(config.simulation, resolve=True)
    simulation["num_envs"] = config.simulation.num_envs if num_envs is None else num_envs
    return Environment(
        task=config.task.name,
        scene=config.scene.name if scene is None else scene,
        reference=config.task.reference,
        duration=config.task.duration,
        reference_seed=config.task.reference_seed,
        sensor=config.sensor.name,
        sensor_config=OmegaConf.to_container(config.sensor.config, resolve=True),
        point_count=config.sensor.point_count,
        **simulation,
    )


def evaluate(
    config, directory, *, actor=None, parameters=None, evaluation_index=0, environments=None
):
    """Evaluate independent initialized episodes and save full failed/successful denominators."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
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
    scenes = config.benchmark.scenes
    if scenes is None:
        scenes = (
            _NAVIGATION_SCENES
            if task == "navigation" and config.mode != "sim"
            else [config.scene.name]
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
    if environments is None:
        environments = {}
    reports = {}
    ros = config.method.name in {"ego", "super"}
    for scene_index, scene in enumerate(scenes):
        seed = int(seed_base + 10_000 * config.seed + 100 * evaluation_index + scene_index)
        if ros:
            env = create_environment(config, num_envs=1, scene=scene)
            all_episodes, runs = [], []
            with RosPlanner(
                config.method.name,
                config.method.address,
                timeout=config.method.timeout,
                replan_interval=1 / config.method.replan_hz,
            ) as planner:
                for episode_index in range(count):
                    episodes, traces, plans, decisions = rollout_ros(
                        env,
                        planner,
                        seed=seed + 1000 * episode_index,
                        planner_name=config.method.name,
                    )
                    destination = directory / scene / f"episode-{episode_index:04d}"
                    successes = sum(episode["event"] == "SUCCESS" for episode in episodes)
                    diagnostic = {
                        "criterion": "diagnostic",
                        "passed": None,
                        "episodes": len(episodes),
                        "successes": successes,
                        "success_rate": successes / len(episodes),
                        "successful_position_rmse": None,
                    }
                    save_evaluation(
                        destination,
                        env,
                        episodes,
                        traces,
                        report=diagnostic,
                        replay_episodes=int(config.benchmark.record_replay),
                        plans=plans,
                    )
                    atomic_json(destination / "decisions.json", {"decisions": decisions})
                    all_episodes.extend(episodes)
                    runs.append(str(destination.relative_to(directory)))
                    print(
                        json.dumps(
                            {
                                "scene": scene,
                                "episode": episode_index,
                                "event": episodes[0]["event"],
                            }
                        ),
                        flush=True,
                    )
            outcomes = Counter(e["event"] for e in all_episodes)
            static = scene in {"S01", "S02", "S03"}
            reports[scene] = {
                "episodes": len(all_episodes),
                "successes": outcomes["SUCCESS"],
                "outcomes": dict(outcomes),
                "runs": runs,
                "criterion": "S6_static_success" if static else "diagnostic",
                "passed": outcomes["SUCCESS"] >= 1 if static else None,
            }
            atomic_json(directory / scene / "report.json", reports[scene])
        else:
            key = (scene, count)
            if key not in environments:
                environments[key] = create_environment(config, num_envs=count, scene=scene)
            env = environments[key]
            episodes, traces = rollout(
                env,
                seed=seed,
                method=config.method.name,
                actor=actor,
                parameters=parameters,
                chunk_steps=config.benchmark.chunk_steps,
            )
            reports[scene] = save_evaluation(
                directory / scene,
                env,
                episodes,
                traces,
                report=acceptance(task, episodes),
                replay_episodes=config.benchmark.replay_episodes
                if config.benchmark.record_replay
                else 0,
            )
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "evaluation": config.mode,
                        **{
                            key: reports[scene][key]
                            for key in (
                                "episodes",
                                "successes",
                                "successful_position_rmse",
                                "passed",
                            )
                        },
                    }
                ),
                flush=True,
            )
    primary = [scene for scene in scenes if scene not in {"S06", "D06"}]
    if task == "navigation" and not ros:
        primary_complete = set(primary) == {"S01", "S02", "S03", "D01", "D02", "D03"}
    else:
        primary_complete = bool(primary)
    summary = {
        "mode": config.mode,
        "training_seed": config.seed,
        "seed_base": seed_base,
        "evaluation_index": evaluation_index,
        "reports": reports,
        "passed": primary_complete and all(reports[scene]["passed"] for scene in primary),
    }
    if task == "navigation" and not ros:
        summary["static_passed"] = all(
            reports.get(scene, {}).get("passed", False) for scene in ["S01", "S02", "S03"]
        )
        summary["dynamic_passed"] = all(
            reports.get(scene, {}).get("passed", False) for scene in ["D01", "D02", "D03"]
        )
    if task == "navigation" and ros:
        static_scenes = {"S01", "S02", "S03"}
        dynamic_scenes = {"D01", "D02", "D03"}
        static_count = sum(reports.get(scene, {}).get("episodes", 0) for scene in static_scenes)
        dynamic_count = sum(reports.get(scene, {}).get("episodes", 0) for scene in dynamic_scenes)
        static_passed = static_count >= 10 and all(
            reports.get(scene, {}).get("successes", 0) >= 1 for scene in static_scenes
        )
        dynamic_passed = dynamic_count >= 10 and all(
            reports.get(scene, {}).get("episodes", 0) > 0 for scene in dynamic_scenes
        )
        summary.update(
            criterion="S6",
            static_episodes=static_count,
            dynamic_episodes=dynamic_count,
            static_passed=static_passed,
            dynamic_passed=dynamic_passed,
            passed=static_passed and dynamic_passed,
        )
    atomic_json(directory / "report.json", summary)
    return summary
