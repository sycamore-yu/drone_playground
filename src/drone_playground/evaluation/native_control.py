"""Real EGO/SUPER execution on the canonical hover/tracking/racing tasks.

An explicit reference-to-goal adapter supplies future target positions to each
original planner; the returned native trajectory is tracked by the same declared
controller. Sensors ray-cast the actual course geometry. Task success remains
owned by the canonical environment, including directed racing gate crossings.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.composition import build_environment, build_sensor
from drone_playground.environments.scenes.control_geometry import bank_from_environment
from drone_playground.environments.sensors.depth import cast_depth, sensor_pose
from drone_playground.environments.sensors.lidar import cast_lidar
from drone_playground.execution.native_tracking import NativeTracking
from drone_playground.integrations.native_planner import NativePlanner
from drone_playground.integrations.native_service import create_native_planner
from drone_playground.runs.record import RunRecorder
from drone_playground.runtime.host_runner import run_steps
from drone_playground.visualization.rscope_io import export_rollout

from .native_planners import pack_array
from .racing import summarize_race
from .tracking import save_report, summarize_trials


def native_execution_evidence(diagnostics, expected_cases):
    """Require actual planner trajectories/commands in every recorded trial."""
    missing = [
        row["case"]
        for row in diagnostics
        if row.get("commands", 0) <= 0 or row.get("trajectories", 0) <= 0
    ]
    return dict(
        passed=len(diagnostics) == expected_cases and not missing,
        fallback_only_cases=missing,
        recorded_cases=len(diagnostics),
        expected_cases=expected_cases,
        rule="every episode must contain a native trajectory and a native command",
    )


def bootstrap_position(task, position):
    """Fly vertically clear of the planner's inflated ground before its first command.

    The canonical racing start lies on the floor. The physical start, clock and
    success conditions stay intact; this is an explicitly counted controller
    fallback until the native planner supplies a valid trajectory.
    """
    target = np.asarray(position).copy()
    if task == "racing":
        target[2] = max(float(target[2]), 0.8)
    return target


def control_sensor(sensor, bank, kind):
    if kind in ("ego", "depth"):

        @jax.jit
        def sample(pos, quat, clock, tick):
            del tick
            frame = cast_depth(sensor, bank, 0, pos, quat, clock, 1)
            origin, rotation = sensor_pose(sensor, pos, quat)
            return frame.depth, origin, rotation
    else:
        sensor = replace(sensor, downsample=1)
        sensor.directions(0)

        @jax.jit
        def sample(pos, quat, clock, tick):
            return cast_lidar(sensor, bank, 0, pos, quat, clock, tick)

    return sample


def evaluate_native_control(config, root, run_id):
    settings = config["method"]
    count = int(config["evaluation"]["episodes"])
    if count < 1:
        raise ValueError("Native control evaluation requires at least one trial")
    split = config["evaluation"]["split"]
    seeds = list(
        range(
            30000 if split == "heldout" else 20000, (30000 if split == "heldout" else 20000) + count
        )
    )
    env = build_environment(config, config["runtime"]["device"], split, count)
    try:
        bank, scene = bank_from_environment(env)
        sensor = build_sensor(config)
        sample = control_sensor(sensor, bank, settings.get("input_sensor", settings.get("method")))
        period = sensor.period_steps(env.freq)
        reset, advance = jax.jit(env.reset), jax.jit(env.step_physical)
        initial = reset(jax.random.PRNGKey(seeds[0]))
        body = env.controller_observation(initial)
        jax.block_until_ready(sample(body["pos"], body["quat"], 0.0, 0))
        jax.block_until_ready(advance(initial, env.physical_action(env.hover_action)))
        adapter = dict(
            interactive_goals=True,
            world_low=np.asarray(bank.world_low).tolist(),
            world_high=np.asarray(bank.world_high).tolist(),
            goal_frequency_hz=5.0,
            lookahead_s=0.6,
            bootstrap_takeoff_m=0.8 if env.task == "racing" else None,
            description="canonical time-reference to receding native planner goals",
        )
        with RunRecorder(root, run_id, config, task_id="final-acceptance/native-control") as rec:
            save_report(rec.path / "scene-manifest.json", scene)
            save_report(rec.path / "task-adapter.json", adapter)
            save_report(rec.path / "sensor-calibration.json", sensor.calibration())
            worker_path = None
            if settings["implementation"] != "native_service":
                worker_path = NativePlanner.install_worker(
                    Path(root) / "native_planners/bridge/worker.py", rec.path, settings["container"]
                )
            traces, diagnostics = [], []
            for case, seed in enumerate(seeds):
                state = reset(jax.random.PRNGKey(seed))
                reference_id = int(state.pipeline_state.reference_id) if env.task != "racing" else 0
                reference = np.asarray(env.trajectories)[reference_id]
                initial_goal = reference[
                    min(round(adapter["lookahead_s"] * env.freq), len(reference) - 1)
                ]
                worker = create_native_planner(
                    settings, rec.path / "native" / str(case), settings["port"], worker_path
                )
                held = bootstrap_position(env.task, env.controller_observation(state)["pos"])
                tracker = NativeTracking(
                    config["env"]["execution"]["tracker"], env, state, worker.directory
                )
                tracker.hold = held
                rows = []
                missing = 0
                try:
                    worker.start(
                        sensor.calibration(), initial_goal, settings.get("limits"), adapter
                    )

                    def decide(current, tick):
                        nonlocal held, missing
                        body = env.controller_observation(current)
                        packet = dict(
                            time=tick * env.dt,
                            position=body["pos"].tolist(),
                            quaternion=body["quat"].tolist(),
                            velocity=body["vel"].tolist(),
                        )
                        if tick % period == 0:
                            data = jax.tree.map(
                                np.asarray,
                                sample(body["pos"], body["quat"], tick * env.dt, tick // period),
                            )
                            if settings.get("input_sensor", settings.get("method")) in (
                                "ego",
                                "depth",
                            ):
                                depth, origin, rotation = data
                                packet.update(
                                    depth=pack_array(depth.reshape(sensor.width, sensor.height).T),
                                    width=sensor.width,
                                    height=sensor.height,
                                    camera_position=origin.tolist(),
                                    camera_quaternion=Rotation.from_matrix(rotation)
                                    .as_quat()
                                    .tolist(),
                                )
                            else:
                                points = data.points_world[data.valid]
                                packet.update(
                                    points=pack_array(np.c_[points, np.ones(len(points))]),
                                    point_count=len(points),
                                )
                        if tick % round(env.freq / adapter["goal_frequency_hz"]) == 0:
                            index = min(
                                tick + round(adapter["lookahead_s"] * env.freq), len(reference) - 1
                            )
                            packet["goal"] = reference[index].tolist()
                        reply = worker.step(packet)
                        target = reply.get("reference")
                        if target is None:
                            missing += 1
                            target = dict(
                                position=held,
                                velocity=[0.0, 0.0, 0.0],
                                acceleration=[0.0, 0.0, 0.0],
                                yaw=0.0,
                            )
                        else:
                            held = body["pos"]
                        physical = tracker.command(reply, current, tick)
                        return physical, dict(
                            commands=reply["commands"], trajectories=reply["trajectories"]
                        )

                    for tick, record, _ in run_steps(state, env.episode_length, decide, advance):
                        nxt = record.after
                        x = nxt.pipeline_state.sim_data.states
                        row = dict(
                            pos=x.pos[0, 0],
                            quat=x.quat[0, 0],
                            obs=nxt.obs,
                            time=(tick + 1) * env.dt,
                            actions=2
                            * (record.command - np.asarray(env.low))
                            / np.asarray(env.high - env.low)
                            - 1,
                            reward=nxt.reward,
                            metrics=nxt.metrics,
                            active=True,
                            failed=bool(nxt.metrics["failure"] > 0),
                        )
                        rows.append(jax.tree.map(np.asarray, row))
                    trace = jax.tree.map(lambda *x: np.stack(x)[:, None], *rows)
                    diagnostics.append(
                        dict(case=case, missing_command_steps=missing, **record.diagnostics)
                    )
                    export_rollout(env.sim, rec.path / "rollouts" / f"case-{case:03d}", trace)
                    traces.append(trace)
                finally:
                    worker.close()
                    tracker.close()
                rec.phase("evaluating", step=case + 1, completed_cases=case + 1)
            length = env.episode_length

            def pad(trace):
                n = len(trace["time"])
                value = jax.tree.map(
                    lambda x: np.concatenate([x, np.repeat(x[-1:], length - n, axis=0)]), trace
                )
                value["active"][n:] = False
                return value

            full = jax.tree.map(lambda *x: np.concatenate(x, axis=1), *(pad(t) for t in traces))
            report = (summarize_race if env.task == "racing" else summarize_trials)(
                full, seeds, env.dt
            )
            evidence = native_execution_evidence(diagnostics, count)
            report["task_quality_passed"] = report["quality_passed"]
            report["quality_passed"] = report["quality_passed"] and evidence["passed"]
            report.update(
                task=env.task,
                method=settings["name"],
                split=split,
                task_adapter=adapter,
                diagnostics=diagnostics,
                all_trials_recorded=True,
                native_execution=evidence,
            )
            save_report(rec.path / "eval/report.json", report)
            rec.finish(
                "completed",
                quality_passed=report["quality_passed"],
                completed=report["completed"],
                num_trials=count,
                full_budget_completed=True,
                engineer_passed=evidence["passed"],
            )
            return report
    finally:
        env.close()
