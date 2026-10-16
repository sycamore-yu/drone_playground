"""Evaluate frozen methods on a declared set of flight cases."""

import json
from pathlib import Path

import jax
import numpy as np

from drone_playground.benchmark.geometry import ReferencePath, traversability
from drone_playground.benchmark.metrics import attach_trajectory_metrics, summarize
from drone_playground.benchmark.reporting import save_evaluation, write_episodes
from drone_playground.benchmark.timing import profile
from drone_playground.simulation.factory import create_environment, create_method
from drone_playground.simulation.records import atomic_json
from drone_playground.simulation.runner import rollout, rollout_host, slice_initial_world

STATIC_OA = ("S01", "S02", "S03", "S06")
DYNAMIC_OA = ("D01", "D02", "D03", "D06")


def evaluate(config, directory, *, actor=None, parameters=None, checkpoint_config=None):
    """Save all outcomes and metric sample counts without a generic quality gate."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = config.benchmark.episodes
    if count is None:
        count = (
            config.simulation.num_envs
            if config.mode == "play"
            else (25 if config.task.name == "navigation" else 100)
        )
    if type(count) is not int or count < 1:
        raise ValueError("Benchmark episode count must be a positive integer")
    scenes = list(
        config.benchmark.scenes
        or (
            (*STATIC_OA, *DYNAMIC_OA)
            if config.task.name == "navigation" and config.mode == "benchmark"
            else (config.scene.name,)
        )
    )
    if len(scenes) != len(set(scenes)):
        raise ValueError("Benchmark scenes must be unique")
    seed_base = config.benchmark.seed_base
    if type(seed_base) is not int or seed_base < 0:
        raise ValueError("benchmark.seed_base must be a nonnegative integer")
    method = create_method(config)
    label = config.method.name
    if not method.trainable:
        label = f"{label}:{type(method.controller).__name__}"
    reports, all_rows, decisions = {}, [], []
    try:
        for scene_index, scene in enumerate(scenes):
            # Stable scene offsets keep partial suites comparable to the full suite.
            seed = seed_base + (
                (*STATIC_OA, *DYNAMIC_OA).index(scene)
                if scene in (*STATIC_OA, *DYNAMIC_OA)
                else scene_index
            )
            # Build the same cohort for every method. Host planners execute its
            # worlds serially; JAX methods retain vectorized execution.
            cohort_env = create_environment(config, num_envs=count, scene=scene)
            env = (
                cohort_env
                if method.execution == "jax"
                else create_environment(config, num_envs=1, scene=scene)
            )
            cohort_initial = cohort_env.reset(jax.random.PRNGKey(seed))
            host_template = (
                env.reset(jax.random.PRNGKey(seed)) if method.execution == "host" else None
            )
            if checkpoint_config is not None and not config.benchmark.allow_environment_change:
                sensor = None if env.sensor is None else env.sensor.specification
                if checkpoint_config["learning"]["sensor"] != sensor:
                    raise ValueError("Frozen evaluation changes checkpoint sensor contract")
            if config.mode == "benchmark":
                measurement_env = create_environment(config, num_envs=1, scene=scene)
                computation = profile(measurement_env, method, actor, parameters)
            else:
                computation = None
            complexity = None
            reference = None
            if config.mode == "benchmark" and scene in (*STATIC_OA, *DYNAMIC_OA):
                complexity = traversability(
                    env.scene,
                    env.task.low,
                    env.task.high,
                    env.task.radius,
                    times=(0.0,) if scene in STATIC_OA else (0.0, 5.0, 10.0, 15.0),
                )
            scene_rows, replays = [], []
            for episode_index in range(1 if method.execution == "jax" else count):
                plans = None
                if method.execution == "jax":
                    episodes, traces = rollout(
                        env,
                        seed=seed,
                        initial_state=cohort_initial,
                        method=method,
                        actor=actor,
                        parameters=parameters,
                        method_name=label,
                        chunk_steps=config.benchmark.chunk_steps,
                        record=True,
                        record_sensors=config.benchmark.record_replay,
                    )
                else:
                    episodes, traces, plans, iteration = rollout_host(
                        env,
                        method,
                        seed=seed + 1000 * episode_index,
                        initial_state=slice_initial_world(
                            cohort_initial, host_template, episode_index, count
                        ),
                        method_name=label,
                        record=True,
                        record_sensors=config.benchmark.record_replay,
                    )
                    episodes[0].update(
                        episode=episode_index,
                        world_index=episode_index,
                        initialization_seed=seed,
                    )
                    decisions.extend(
                        {"scene": scene, "episode": episode_index, **x} for x in iteration
                    )
                attach_trajectory_metrics(episodes, traces)
                for world, row in enumerate(episodes):
                    row["case_id"] = f"{scene}:{row['episode']:04d}"
                    row.update(
                        reference_path_metres=None,
                        path_excess_percent=None,
                        reference_path_status="not_applicable",
                    )
                    if (
                        config.mode == "benchmark"
                        and scene in STATIC_OA
                        and row["event"] == "SUCCESS"
                    ):
                        if reference is None:
                            reference = ReferencePath(
                                env.scene, env.task.low, env.task.high, env.task.radius
                            )
                        length = reference.length(
                            traces["position"][0, world], traces["position"][-1, world]
                        )
                        row["reference_path_metres"] = length
                        row["reference_path_status"] = (
                            "measured" if length else "unreachable_at_resolution"
                        )
                        if length:
                            row["path_excess_percent"] = (
                                row["path_length_metres"] / length - 1
                            ) * 100
                scene_rows.extend(episodes)
                trace_name = f"{scene.replace('/', '_')}__{episode_index:04d}.npz"
                failed = [i for i, row in enumerate(episodes) if row["event"] != "SUCCESS"]
                if failed:
                    failure_dir = directory / "trajectories" / "failed"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        failure_dir / trace_name,
                        **{name: value[:, failed] for name, value in traces.items()},
                    )
                if config.benchmark.record_trajectories:
                    trace_dir = directory / "trajectories"
                    trace_dir.mkdir(exist_ok=True)
                    np.savez_compressed(trace_dir / trace_name, **traces)
                if config.benchmark.record_replay:
                    saved = save_evaluation(
                        directory,
                        env,
                        episodes,
                        traces,
                        report={},
                        plans=plans,
                        name=scene.replace("/", "_"),
                        replay_episodes=config.benchmark.replay_episodes,
                        write_tables=False,
                        record_trajectories=False,
                    )
                    replays.extend(saved["replays"])
            if len(scene_rows) != count:
                raise RuntimeError("Evaluation did not produce every requested episode")
            reports[scene] = {
                **summarize(scene_rows),
                "task": config.task.name,
                "task_contract": env.task.contract,
                "scene": scene,
                "geometry": env.scene.geometry_identity,
                "sensor": None if env.sensor is None else env.sensor.specification,
                "method": label,
                "replays": replays,
                "timing": computation,
                "traversability": complexity,
                "execution": "ideal_tracking" if env.control_level == "ideal" else "crazyflow",
            }
            all_rows.extend(scene_rows)
            write_episodes(directory / "episodes.csv", all_rows)
            failed_rows = [row for row in all_rows if row["event"] != "SUCCESS"]
            if failed_rows:
                write_episodes(directory / "failures.csv", failed_rows)
            atomic_json(
                directory / "report.json",
                {
                    "schema_version": 3,
                    "mode": config.mode,
                    "reports": reports,
                    "complete": False,
                },
            )
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "episodes": len(scene_rows),
                        "successes": reports[scene]["successes"],
                    }
                ),
                flush=True,
            )
    finally:
        method.close()
    summary = {
        "schema_version": 3,
        "mode": config.mode,
        "seed_base": seed_base,
        "reports": reports,
        "complete": True,
        "trajectory_measurement": {
            "frame": "world",
            "differentiation": "central differences of recorded velocity",
            "sampling_hz": config.simulation.method_hz,
            "aggregation": "successful episodes; per-metric contributing count",
        },
        "reference_path_protocol": {
            "algorithm": "A* on a conservative 26-connected grid",
            "resolution_metres": 0.5,
            "endpoints": "actual initial and terminal position",
            "body": "task sphere radius",
            "missing": "reported with contributing counts",
            "interpretation": "grid approximation; shorter flown paths can yield negative excess",
        },
        "suites": {
            name: summarize(rows)
            for name, suite in (("static_oa", STATIC_OA), ("dynamic_oa", DYNAMIC_OA))
            if (rows := [row for row in all_rows if row["scene"] in suite])
        },
    }
    if decisions:
        atomic_json(directory / "decisions.json", {"decisions": decisions})
    atomic_json(directory / "report.json", summary)
    return summary
