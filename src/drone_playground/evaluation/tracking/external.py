"""Real EGO/SUPER execution on the canonical hover/tracking/racing tasks.

An explicit reference-to-goal adapter supplies future target positions to each
original planner; the returned native trajectory is tracked by the same declared
controller. Sensors ray-cast the actual course geometry. Task success remains
owned by the canonical environment, including directed racing gate crossings.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.actions.external_tracking import ExternalTracking
from drone_playground.artifacts.decisions import NativeDecisionRecorder
from drone_playground.artifacts.record import RunRecorder
from drone_playground.artifacts.reporting import save_report
from drone_playground.composition import build_environment
from drone_playground.environments.environment import build_sensor
from drone_playground.environments.scenes.mujoco_geometry import bank_from_environment
from drone_playground.environments.sensors.depth import cast_depth, sensor_pose
from drone_playground.environments.sensors.lidar import cast_lidar
from drone_playground.integrations.sensors import pack_array
from drone_playground.evaluation.racing import summarize_race
from drone_playground.evaluation.tracking.metrics import summarize_trials
from drone_playground.integrations.grpc_service import create_native_planner
from drone_playground.integrations.ros1 import NativePlanner
from drone_playground.runtime.host_runner import run_steps
from drone_playground.visualization.rscope_io import export_rollout


def native_execution_evidence(diagnostics, expected_cases):
    """Require actual planner trajectories/commands in every recorded trial."""
    missing = [
        row["case"]
        for row in diagnostics
        if row.get("commands", 0) <= 0
        or row.get("plans", row.get("trajectories", 0)) <= 0
        or row.get("executed_native_steps", row.get("commands", 0)) <= 0
    ]
    return dict(
        passed=len(diagnostics) == expected_cases and not missing,
        fallback_only_cases=missing,
        recorded_cases=len(diagnostics),
        expected_cases=expected_cases,
        rule="every episode must contain a native physical decision and an executed command",
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
    if kind == "none":
        return lambda pos, quat, clock, tick: None
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
        raise ValueError(
            "Native control evaluation requires at least one trial"
        )
    role = config["evaluation"]["role"]
    seed_start = config["evaluation"].get("seed_start")
    if seed_start is None:
        seed_start = 30000
    seeds = list(range(seed_start, seed_start + count))
    env = build_environment(config, config["runtime"]["device"], role, count)
    try:
        bank, scene = bank_from_environment(env)
        sensor = build_sensor(config)
        sample = control_sensor(
            sensor, bank, settings.get("input_sensor", settings.get("method"))
        )
        period = sensor.period_steps(env.freq) if sensor else 1
        reset, advance = jax.jit(env.reset), jax.jit(env.step_physical)
        initial = reset(jax.random.PRNGKey(seeds[0]))
        body = env.controller_observation(initial)
        jax.block_until_ready(sample(body["pos"], body["quat"], 0.0, 0))
        jax.block_until_ready(
            advance(initial, env.physical_action(env.hover_action))
        )
        adapter = dict(
            record_planner_visualization=config["evaluation"].get(
                "record_planner_visualization", True
            ),
            interactive_goals=True,
            world_low=np.asarray(bank.world_low).tolist(),
            world_high=np.asarray(bank.world_high).tolist(),
            goal_frequency_hz=5.0,
            lookahead_s=0.6,
            bootstrap_takeoff_m=0.8 if env.task == "racing" else None,
            description="canonical time-reference to receding native planner goals",
        )
        with RunRecorder(
            root, run_id, config, task_id="final-acceptance/native-control"
        ) as rec:
            rec.record_environment(env)
            save_report(rec.path / "scene-manifest.json", scene)
            save_report(rec.path / "task-adapter.json", adapter)
            calibration = sensor.calibration() if sensor else {}
            save_report(rec.path / "sensor-calibration.json", calibration)
            worker_path = None
            if settings["implementation"] not in ("native_service", "pipeline"):
                worker_path = NativePlanner.install_worker(
                    Path(root) / "ros_integrations/ros1/bridge/worker.py",
                    rec.path,
                    settings["container"],
                )
            traces, diagnostics, reset_details = [], [], []
            for case, seed in enumerate(seeds):
                state = reset(jax.random.PRNGKey(seed))
                body = env.controller_observation(state)
                reset_details.append(
                    dict(
                        initial_position_m=np.asarray(body["pos"]).tolist(),
                        initial_velocity_mps=np.asarray(body["vel"]).tolist(),
                        initial_quaternion_xyzw=np.asarray(
                            body["quat"]
                        ).tolist(),
                        **{
                            key: float(state.info[key])
                            for key in (
                                "delay_requested_ms",
                                "delay_effective_ms",
                            )
                            if key in state.info
                        },
                    )
                )
                reference_id = (
                    int(state.pipeline_state.reference_id)
                    if env.task != "racing"
                    else 0
                )
                reference = np.asarray(env.trajectories)[reference_id]
                initial_goal = reference[
                    min(
                        round(adapter["lookahead_s"] * env.freq),
                        len(reference) - 1,
                    )
                ]
                worker = create_native_planner(
                    settings,
                    rec.path / "native" / str(case),
                    settings["port"],
                    worker_path,
                    env=env,
                )
                held = bootstrap_position(
                    env.task, env.controller_observation(state)["pos"]
                )
                tracker = ExternalTracking(
                    config["env"]["action"].get("tracker"),
                    env,
                    state,
                    worker.directory,
                )
                tracker.hold = held
                rows = []
                latencies = []
                missing = 0
                with NativeDecisionRecorder(
                    worker.directory / "decision-trace",
                    env.controller.input_kind,
                ) as decisions:
                    try:
                        worker.start(
                            calibration,
                            initial_goal,
                            settings.get("limits"),
                            adapter,
                        )

                        def decide(current, tick):
                            nonlocal held, missing
                            observation_start = time.perf_counter()
                            body = env.controller_observation(current)
                            packet = dict(
                                time=tick * env.dt,
                                policy_observation=np.asarray(
                                    current.obs
                                ).tolist(),
                                position=body["pos"].tolist(),
                                quaternion=body["quat"].tolist(),
                                velocity=body["vel"].tolist(),
                                angular_velocity=body["ang_vel"].tolist(),
                            )
                            if sensor is not None and tick % period == 0:
                                data = jax.tree.map(
                                    np.asarray,
                                    sample(
                                        body["pos"],
                                        body["quat"],
                                        tick * env.dt,
                                        tick // period,
                                    ),
                                )
                                if settings.get(
                                    "input_sensor", settings.get("method")
                                ) in (
                                    "ego",
                                    "depth",
                                ):
                                    depth, origin, rotation = data
                                    packet.update(
                                        depth=pack_array(
                                            depth.reshape(
                                                sensor.width, sensor.height
                                            ).T
                                        ),
                                        width=sensor.width,
                                        height=sensor.height,
                                        camera_position=origin.tolist(),
                                        camera_quaternion=Rotation.from_matrix(
                                            rotation
                                        )
                                        .as_quat()
                                        .tolist(),
                                    )
                                else:
                                    points = data.points_world[data.valid]
                                    packet.update(
                                        points=pack_array(
                                            np.c_[points, np.ones(len(points))]
                                        ),
                                        point_count=len(points),
                                    )
                            if (
                                tick
                                % round(env.freq / adapter["goal_frequency_hz"])
                                == 0
                            ):
                                index = min(
                                    tick
                                    + round(adapter["lookahead_s"] * env.freq),
                                    len(reference) - 1,
                                )
                                packet["goal"] = reference[index].tolist()
                            observation_seconds = (
                                time.perf_counter() - observation_start
                            )
                            reply = worker.step(packet)
                            target = reply.get("reference")
                            if target is None and reply.get("output") is None:
                                missing += 1
                                target = dict(
                                    position=held,
                                    velocity=[0.0, 0.0, 0.0],
                                    acceleration=[0.0, 0.0, 0.0],
                                    yaw=0.0,
                                )
                            else:
                                held = body["pos"]
                            physical = None
                            try:
                                physical = tracker.command(reply, current, tick)
                            finally:
                                recording_start = time.perf_counter()
                                decisions.record(
                                    tick, tick * env.dt, body, reply, physical
                                )
                            return physical, dict(
                                observation_seconds=observation_seconds,
                                recording_seconds=time.perf_counter()
                                - recording_start,
                                commands=reply["commands"],
                                trajectories=reply["trajectories"],
                                plans=reply.get("plans", reply["trajectories"]),
                                executed_native_steps=tracker.consumed,
                                clipped_command_steps=tracker.clipped,
                            )

                        for tick, record, _ in run_steps(
                            state, env.episode_length, decide, advance
                        ):
                            nxt = record.after
                            x = nxt.pipeline_state.sim_data.states
                            latencies.append(
                                record.diagnostics["decision_seconds"]
                            )
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
                        trace = jax.tree.map(
                            lambda *x: np.stack(x)[:, None], *rows
                        )
                        diagnostics.append(
                            dict(
                                case=case,
                                missing_command_steps=missing,
                                decision_samples_seconds=latencies,
                                **record.diagnostics,
                            )
                        )
                        if hasattr(worker, "module_calls"):
                            diagnostics[-1]["module_calls"] = list(
                                worker.module_calls
                            )
                        traces.append(trace)
                    finally:
                        worker.close()
                        tracker.close()
                from drone_playground.artifacts.decisions import load_native_decisions
                from drone_playground.visualization.layers import layers_from_decisions, sensor_view

                visualization = layers_from_decisions(
                    load_native_decisions(worker.directory / "decision-trace"),
                    sensor=sensor_view(calibration),
                )
                if config["evaluation"].get("record_replays", False):
                    export_rollout(
                        env.sim,
                        rec.path / "rollouts" / f"case-{case:03d}",
                        trace,
                        visualization=visualization,
                    )
                rec.phase("evaluating", step=case + 1, completed_cases=case + 1)
            length = env.episode_length

            def pad(trace):
                n = len(trace["time"])
                value = jax.tree.map(
                    lambda x: np.concatenate(
                        [x, np.repeat(x[-1:], length - n, axis=0)]
                    ),
                    trace,
                )
                value["active"][n:] = False
                return value

            full = jax.tree.map(
                lambda *x: np.concatenate(x, axis=1), *(pad(t) for t in traces)
            )
            report = (
                summarize_race if env.task == "racing" else summarize_trials
            )(full, seeds, env.dt)
            for episode, initial_conditions in zip(
                report["episodes"], reset_details, strict=True
            ):
                episode.update(initial_conditions)
            from drone_playground.benchmarks import apply_quality

            apply_quality(report, config)
            evidence = native_execution_evidence(diagnostics, count)
            report["task_quality_passed"] = report["quality_passed"]
            report["quality_passed"] = (
                report["quality_passed"] and evidence["passed"]
            )
            report.update(
                task=env.task,
                method=settings["name"],
                role=role,
                task_adapter=adapter,
                diagnostics=diagnostics,
                all_trials_recorded=True,
                native_execution=evidence,
            )
            from drone_playground.runtime.timing import decision_statistics

            stable = [
                value
                for case in diagnostics
                for value in case["decision_samples_seconds"][1:]
            ]
            report.update(decision_statistics(stable, env.dt, warmup=0))
            report["warmup_decisions"] = len(diagnostics)
            report["sensor_timing"] = getattr(env, "sensor_timing", None)
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
