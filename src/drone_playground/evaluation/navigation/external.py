"""Native EGO/SUPER closed loop on the shared navigation scenes and physics."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.artifacts.decisions import NativeDecisionRecorder
from drone_playground.artifacts.record import RunRecorder
from drone_playground.artifacts.reporting import save_report
from drone_playground.artifacts.traces import record_native_case
from drone_playground.control.external_tracking import ExternalTracking
from drone_playground.environments.environment import build_environment
from drone_playground.environments.scenes.geometry import DIFFICULTIES
from drone_playground.environments.sensors.depth import cast_depth, sensor_pose
from drone_playground.environments.sensors.lidar import cast_lidar
from drone_playground.evaluation.navigation.cases import navigation_cases, navigation_resets
from drone_playground.evaluation.navigation.metrics import combine_cells, summarize_cell
from drone_playground.evaluation.navigation.policy import export_navigation_replays
from drone_playground.integrations.grpc_service import create_native_planner
from drone_playground.integrations.sensors import sensor_packet
from drone_playground.runtime.host_runner import run_steps


def sensor_function(env, method):
    if method == "none":
        return lambda data: None
    if method in ("ego", "depth"):

        @jax.jit
        def sample(data):
            state = data.sim_data.states
            frame = cast_depth(
                env.sensor,
                env.bank,
                data.scenario_id,
                state.pos[0, 0],
                state.quat[0, 0],
                data.step_index * env.dt,
                1,
            )
            pos, rotation = sensor_pose(env.sensor, state.pos[0, 0], state.quat[0, 0])
            return frame.depth, pos, rotation

        return sample
    lidar = replace(env.sensor, downsample=1)
    # Materialize source scan windows before tracing, as in the environment.
    lidar.directions(0)

    @jax.jit
    def sample(data):
        state = data.sim_data.states
        return cast_lidar(
            lidar,
            env.bank,
            data.scenario_id,
            state.pos[0, 0],
            state.quat[0, 0],
            data.step_index * env.dt,
            data.step_index // env.sensor_period,
        )

    return sample


def evaluate_native(config, root: Path, run_id: str):
    settings = config["method"]
    method = settings.get("input_sensor", settings.get("method"))
    role = config["evaluation"]["role"]
    count = int(config["evaluation"]["episodes"])
    per_scene = config["evaluation"].get("per_scene", False)
    rec = RunRecorder(root, run_id, config, task_id="p5/06-native-planners")
    env = None
    try:
        rec.phase("initializing")
        env = build_environment(
            config,
            config["runtime"]["device"],
            role,
            max(2, count) if per_scene else count,
        )
        rec.record_environment(env)
        save_report(rec.path / "scene-manifest.json", env.scene_manifest)
        seed_start = config["evaluation"].get("seed_start")
        if seed_start is None:
            seed_start = 30000
        from drone_playground.benchmarks import load_protocol

        protocol = (
            load_protocol(config["evaluation"]["protocol"])
            if config["evaluation"].get("protocol")
            else None
        )
        cases = navigation_cases(env.bank, count, seed_start, per_scene, protocol=protocol)
        ordered = [case for group in cases.values() for case in group]
        scenario_groups = {name: [c["scenario_id"] for c in group] for name, group in cases.items()}
        initials = None
        if config["evaluation"].get("initial_conditions"):
            initials = navigation_resets(
                env.bank,
                [c["scenario_id"] for c in ordered],
                [c["seed"] for c in ordered],
                config["evaluation"]["initial_conditions"],
                env.body_radius,
            )
            save_report(rec.path / "initial-conditions.json", initials["record"])
        for index, case in enumerate(ordered):
            case["reset_index"] = index
        worker_path = None
        sample = sensor_function(env, method)
        advance = jax.jit(env.step_physical)
        reset = jax.jit(env.reset)
        # Compile all physics/sensing before starting /clock or native timeout accounting.
        warm = reset(jax.random.PRNGKey(0), jnp.int32(0))
        jax.block_until_ready(advance(warm, env.physical_action(env.hover_action)))
        jax.block_until_ready(sample(warm.pipeline_state))
        cells = {}
        total_trajectories = total_commands = 0
        diagnostics = []
        from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
        from queue import Queue
        from threading import Event

        workers = min(len(ordered), int(settings.get("workers", 1)))
        if workers < 1:
            raise ValueError("Native evaluation needs at least one worker")
        cancelled = Event()
        ports = Queue()
        for offset in range(workers):
            ports.put(int(settings["port"]) + offset)

        def run_case(difficulty, case, port):
            if cancelled.is_set():
                raise CancelledError("Native evaluation cancelled")
            episode = cases[difficulty][case]
            scenario_id = episode["scenario_id"]
            initial_state = (
                None
                if initials is None
                else {
                    field: jnp.asarray(initials[field][episode["reset_index"]])
                    for field in ("position", "velocity", "quaternion")
                }
            )
            state = reset(
                jax.random.PRNGKey(episode["seed"]),
                jnp.int32(scenario_id),
                initial_state=initial_state,
            )
            actual = env.controller_observation(state)
            reset_evidence = dict(
                seed=episode["seed"],
                scene_id=episode["scene_id"],
                initial_position_m=actual["pos"].tolist(),
                initial_velocity_mps=actual["vel"].tolist(),
                initial_quaternion_xyzw=actual["quat"].tolist(),
            )
            if "delay_effective_ms" in state.info:
                reset_evidence["delay_effective_ms"] = float(state.info["delay_effective_ms"])
            case_directory = rec.path / "native" / difficulty / str(case)
            identity = dict(
                difficulty=difficulty,
                case=case,
                scenario_id=scenario_id,
                **reset_evidence,
            )
            with (
                record_native_case(env, case_directory, identity) as rows,
                NativeDecisionRecorder(
                    case_directory / "decision-trace", env.controller.contract()
                ) as decisions,
            ):
                worker = controller = None
                unavailable = rejected = 0
                commands = trajectories = 0
                hold = np.asarray(state.pipeline_state.sim_data.states.pos[0, 0])
                latencies = []
                tic = time.monotonic()
                try:
                    worker = create_native_planner(
                        settings,
                        rec.path / "native" / difficulty / str(case),
                        port,
                        worker_path,
                        env=env,
                    )
                    controller = ExternalTracking(
                        env,
                        state,
                        worker.directory,
                    )
                    worker.start(
                        env.sensor_calibration,
                        env.bank.goal[scenario_id],
                        settings.get("limits"),
                        task_adapter=dict(
                            world_low=np.asarray(env.bank.world_low).tolist(),
                            world_high=np.asarray(env.bank.world_high).tolist(),
                            record_planner_visualization=config["evaluation"].get(
                                "record_planner_visualization", True
                            ),
                        ),
                    )

                    def decide(current, tick):
                        nonlocal commands, trajectories, unavailable, rejected, hold
                        if cancelled.is_set():
                            raise CancelledError("Native evaluation cancelled")
                        observation_start = time.perf_counter()
                        packet = sensor_packet(env, method, sample, current)
                        if case == 0 and tick in (0, 150):
                            save_report(
                                worker.directory / f"sensor-packet-{tick:04d}.json",
                                packet,
                            )
                        observation_seconds = time.perf_counter() - observation_start
                        reply = worker.step(packet)
                        reference = reply.get("reference")
                        commands, trajectories = (
                            reply["commands"],
                            reply["trajectories"],
                        )
                        if reference is None and reply.get("output") is None:
                            unavailable += 1
                            rejected += int(reply.get("rejected_reference", False))
                            reference = dict(
                                position=hold,
                                velocity=[0.0, 0.0, 0.0],
                                acceleration=[0.0, 0.0, 0.0],
                                yaw=0.0,
                            )
                        else:
                            hold = np.asarray(env.controller_observation(current)["pos"])
                        physical = None
                        try:
                            physical = controller.command(reply, current, tick)
                        finally:
                            recording_start = time.perf_counter()
                            decisions.record(
                                tick,
                                tick * env.dt,
                                env.controller_observation(current),
                                reply,
                                physical,
                            )
                        return physical, dict(
                            commands=commands,
                            trajectories=trajectories,
                            observation_seconds=observation_seconds,
                            recording_seconds=time.perf_counter() - recording_start,
                        )

                    for tick, transition, _ in run_steps(
                        state, env.episode_length, decide, advance
                    ):
                        state, physical = transition.after, transition.command
                        action = (
                            2 * (physical - np.asarray(env.low)) / np.asarray(env.high - env.low)
                            - 1
                        )
                        latencies.append(transition.diagnostics["decision_seconds"])
                        row = dict(
                            pos=state.pipeline_state.sim_data.states.pos[0, 0],
                            quat=state.pipeline_state.sim_data.states.quat[0, 0],
                            obs=state.info["terminal_proprioception"],
                            time=(tick + 1) * env.dt,
                            actions=action,
                            reward=state.reward,
                            metrics=state.metrics,
                            active=True,
                            failed=state.done,
                            done=state.done,
                            outcome=state.info["outcome"],
                        )
                        rows.append(jax.tree.map(np.asarray, row))
                finally:
                    try:
                        if worker is not None:
                            worker.close()
                    finally:
                        if controller is not None:
                            controller.close()
                diag = dict(
                    decision_seconds=latencies,
                    difficulty=difficulty,
                    case=case,
                    scenario_id=scenario_id,
                    commands=commands,
                    trajectories=trajectories,
                    missing_command_steps=unavailable,
                    rejected_commands=rejected,
                    downstream_tracker=controller.name,
                    downstream_missing_steps=controller.missing,
                    insufficient_horizon_steps=controller.short_horizon,
                    executed_native_steps=controller.consumed,
                    clipped_command_steps=controller.clipped,
                    wall_seconds=time.monotonic() - tic,
                    rpc_p95_s=float(np.percentile(worker.latencies, 95)),
                    **reset_evidence,
                )
                if hasattr(worker, "module_calls"):
                    diag["module_calls"] = list(worker.module_calls)
                save_report(worker.directory / "diagnostics.json", diag)
                trace = jax.tree.map(lambda *values: np.stack(values), *rows)
                label = {
                    "scenario_id": scenario_id,
                    **env.bank.labels(scenario_id),
                    **reset_evidence,
                }
                return trace, label, diag

        def run_slot(difficulty, case):
            port = ports.get()
            try:
                return run_case(difficulty, case, port)
            finally:
                ports.put(port)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            try:
                for difficulty in DIFFICULTIES:
                    size = len(cases[difficulty])
                    traces, labels = [None] * size, [None] * size
                    futures = {
                        pool.submit(run_slot, difficulty, case): case for case in range(size)
                    }
                    rec.phase("evaluating", difficulty=difficulty, completed_cases=0)
                    for finished, future in enumerate(as_completed(futures), 1):
                        case = futures[future]
                        trace, label, diag = future.result()
                        traces[case], labels[case] = trace, label
                        diagnostics.append(diag)
                        total_trajectories += diag["trajectories"]
                        total_commands += diag["executed_native_steps"]
                        save_report(rec.path / "native-progress.json", diagnostics)
                        rec.phase(
                            "evaluating",
                            difficulty=difficulty,
                            completed_cases=finished,
                        )
                    # Ragged episodes are padded only for in-memory aggregation. Every
                    # archive and replay trims on active; the worker stops at termination.
                    length = max(item["pos"].shape[0] for item in traces)
                    padded = []
                    for item in traces:
                        n = item["pos"].shape[0]
                        padded_item = jax.tree.map(
                            lambda x: np.concatenate([x, np.repeat(x[-1:], length - n, axis=0)]),
                            item,
                        )
                        padded_item["active"][n:] = False
                        padded.append(padded_item)
                    trace = jax.tree.map(lambda *values: np.stack(values, axis=1), *padded)
                    cells[difficulty] = summarize_cell(trace, labels, env.dt, env.duration)
                    from drone_playground.artifacts.traces import save_navigation_traces

                    save_navigation_traces(
                        env,
                        {difficulty: trace},
                        rec.path / "traces",
                        scenario_groups,
                    )
                    from drone_playground.evaluation.navigation.policy import select_episodes

                    selection = select_episodes({"cells": {difficulty: cells[difficulty]}})
                    export_navigation_replays(
                        env,
                        {difficulty: trace},
                        rec.path / "rollouts",
                        case_indices=selection,
                        scenario_groups=scenario_groups,
                        decision_root=rec.path / "native",
                    )
                    save_report(
                        rec.path / "eval" / (difficulty + ".json"),
                        cells[difficulty],
                    )
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise
        report = combine_cells(cells)
        runtime_identities = [
            json.loads(path.read_text())
            for path in sorted((rec.path / "native").glob("*/*/runtime-identity.json"))
        ]
        report.update(
            role=role,
            episodes_per_difficulty=None if per_scene else count,
            episodes_per_scene=count if per_scene else None,
            episodes=[row for cell in cells.values() for row in cell["episodes"]],
            initial_conditions=None if initials is None else initials["record"],
            scenario_groups=scenario_groups,
            config=config,
            parameter_sha256=hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            parameter_identity_kind="resolved native configuration",
            runtime_identities=runtime_identities,
            method=method,
            scene_bank_sha256=env.bank.digest(),
            native_trajectories=total_trajectories,
            executed_native_steps=total_commands,
            sensor_calibration=env.sensor_calibration,
            parameters_frozen=True,
            quality_passed=None,
            quality_rule="benchmark validation pending",
            diagnostics=diagnostics,
            native_modification="pinned upstream with declared native/ros1/patches; runtime hashes identify deployed build; gRPC/ROS adapters",
        )
        from drone_playground.runtime.timing import decision_statistics

        stable = [value for case in diagnostics for value in case["decision_seconds"][1:]]
        report.update(decision_statistics(stable, env.dt, warmup=0))
        report["warmup_decisions"] = len(diagnostics)
        report["sensor_timing"] = getattr(env, "sensor_timing", None)
        save_report(rec.path / "eval" / "report.json", report)
        if total_commands == 0:
            raise RuntimeError("Native process produced no executable physical commands")
        from drone_playground.benchmarks import benchmark_id

        benchmark = benchmark_id(config)
        if benchmark in (
            "native-navigation-v1",
            "native-navigation-primary-v1",
        ):
            from drone_playground.benchmarks import validate_navigation_report

            if (
                len(runtime_identities) != len(ordered)
                or not runtime_identities[0]
                or any(identity != runtime_identities[0] for identity in runtime_identities)
            ):
                raise ValueError(
                    "Every native benchmark episode requires the same authenticated runtime"
                )
            if any(
                row["arrived"] and diag["executed_native_steps"] == 0
                for cell in cells.values()
                for row in cell["episodes"]
                for diag in diagnostics
                if (diag["difficulty"], diag["case"]) == (row["difficulty"], row["case"])
            ):
                raise ValueError("A fallback-only arrival cannot certify the native algorithm")
            task = "dynamic" if config["env"]["task"]["dynamic"] else "static"
            criterion = (
                "navigation-primary-v1"
                if benchmark == "native-navigation-primary-v1"
                else "navigation-v1"
            )
            validation = validate_navigation_report(
                report, tasks=(task,), criterion=criterion, protocol=protocol
            )
            save_report(rec.path / "eval/benchmark-validation.json", validation)
            report.update(
                quality_passed=validation["passed"],
                quality_rule=validation["protocol"],
            )
            save_report(rec.path / "eval/report.json", report)
        rec.finish(
            "completed",
            engineer_passed=True,
            full_budget_completed=True,
            num_trials=report["num_trials"],
            arrived=report["arrived"],
            quality_passed=report["quality_passed"],
        )
        return report
    except BaseException as exc:
        rec.finish("failed", error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close()
