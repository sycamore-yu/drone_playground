"""Native EGO/SUPER closed loop on the shared navigation scenes and physics."""
from __future__ import annotations

import base64
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.composition import build_environment
from drone_playground.controllers.trajectory import TrajectoryTracking
from drone_playground.evaluation.navigation import (
    combine_cells,
    export_navigation_replays,
    summarize_cell,
)
from drone_playground.evaluation.tracking import save_report
from drone_playground.policies.native_planner import NativePlanner
from drone_playground.runs.record import RunRecorder
from drone_playground.tasks.scenes.navigation import DIFFICULTIES
from drone_playground.tasks.sensors.depth import cast_depth, sensor_pose
from drone_playground.tasks.sensors.lidar import cast_lidar


def pack_array(value):
    return base64.b64encode(np.asarray(value, dtype="<f4").tobytes()).decode("ascii")

def sensor_function(env, method):
    if method == "ego":
        @jax.jit
        def sample(data):
            state = data.sim_data.states
            frame = cast_depth(env.sensor, env.bank, data.scenario_id,
                               state.pos[0, 0], state.quat[0, 0], data.step_index * env.dt, 1)
            pos, rotation = sensor_pose(env.sensor, state.pos[0, 0], state.quat[0, 0])
            return frame.depth, pos, rotation
        return sample
    lidar = replace(env.sensor, downsample=1)
    # Materialize source scan windows before tracing, as in the environment.
    lidar.directions(0)
    @jax.jit
    def sample(data):
        state = data.sim_data.states
        return cast_lidar(lidar, env.bank, data.scenario_id,
                          state.pos[0, 0], state.quat[0, 0],
                          data.step_index * env.dt, data.step_index // env.sensor_period)
    return sample

def sensor_packet(env, method, sample, state):
    data = state.pipeline_state
    body = env.controller_observation(state)
    packet = {"time": int(data.step_index) * env.dt, "position": body["pos"].tolist(),
              "quaternion": body["quat"].tolist(), "velocity": body["vel"].tolist()}
    if int(data.step_index) % env.sensor_period:
        return packet
    if method == "ego":
        depth, pos, rotation = jax.tree.map(np.asarray, sample(data))
        # Ray grid is width-major; ROS images are row-major.
        depth = depth.reshape(env.sensor.width, env.sensor.height).T
        packet.update(depth=pack_array(depth), width=env.sensor.width, height=env.sensor.height,
                      camera_position=pos.tolist(),
                      camera_quaternion=Rotation.from_matrix(rotation).as_quat().tolist())
    else:
        frame = jax.tree.map(np.asarray, sample(data))
        points = frame.points_world[frame.valid]
        points = np.c_[points, np.ones(len(points), dtype=np.float32)]
        packet.update(points=pack_array(points), point_count=len(points))
    return packet

def evaluate_native(config, root: Path, run_id: str):
    settings = config["policy"]
    method = settings["method"]
    split = config["evaluation"]["split"]
    count = int(config["evaluation"]["episodes"])
    rec = RunRecorder(root, run_id, config, task_id="p5/06-native-planners")
    env = None
    try:
        rec.phase("initializing")
        env = build_environment(config, config["training"]["device"], split, count)
        save_report(rec.path / "scene-manifest.json", env.scene_manifest)
        worker_path = NativePlanner.install_worker(
            Path(__file__).resolve().parents[3] / "scripts/p5_ros_bridge.py",
            rec.path, settings["container"])
        sample = sensor_function(env, method)
        advance = jax.jit(env.step_physical)
        reset = jax.jit(env.reset)
        controller = TrajectoryTracking(**{k: v for k, v in config["controller"].items() if k != "name"})
        controller.bind(env.low, env.high)
        # Compile all physics/sensing before starting /clock or native timeout accounting.
        warm = reset(jax.random.PRNGKey(0), jnp.int32(0))
        jax.block_until_ready(advance(warm, env.physical_action(env.hover_action)))
        jax.block_until_ready(sample(warm.pipeline_state))
        cells = {}
        total_trajectories = 0
        diagnostics = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from queue import Queue

        workers = min(count, int(settings.get("workers", 1)))
        if workers < 1:
            raise ValueError("Native evaluation needs at least one worker")
        ports = Queue()
        for offset in range(workers):
            ports.put(int(settings["port"]) + offset)

        def run_case(difficulty_index, difficulty, case, port):
            scenario_id = difficulty_index * count + case
            state = reset(jax.random.PRNGKey(30000 + case if split == "heldout" else 20000 + case),
                          jnp.int32(scenario_id))
            worker = NativePlanner(method, rec.path / "native" / difficulty / str(case),
                                   settings["container"], port, worker_path)
            rows = []
            unavailable = rejected = 0
            commands = trajectories = 0
            hold = np.asarray(state.pipeline_state.sim_data.states.pos[0, 0])
            tic = time.monotonic()
            try:
                worker.start(env.sensor_calibration, env.bank.goal[scenario_id])
                for tick in range(env.episode_length):
                    packet = sensor_packet(env, method, sample, state)
                    if case == 0 and tick in (0, 150):
                        save_report(worker.directory / f"sensor-packet-{tick:04d}.json", packet)
                    reply = worker.step(packet)
                    reference = reply.get("reference")
                    commands, trajectories = reply["commands"], reply["trajectories"]
                    if reference is None:
                        unavailable += 1
                        rejected += int(reply.get("rejected_reference", False))
                        reference = dict(position=hold, velocity=[0., 0., 0.],
                                         acceleration=[0., 0., 0.], yaw=0.)
                    else:
                        hold = np.asarray(env.controller_observation(state)["pos"])
                    physical = controller.command(env.controller_observation(state), reference,
                                                  float(env.default.params.mass[0]))
                    state = advance(state, jnp.asarray(physical, jnp.float32))
                    action = 2 * (physical - np.asarray(env.low)) / np.asarray(env.high - env.low) - 1
                    row = dict(pos=state.pipeline_state.sim_data.states.pos[0, 0],
                               quat=state.pipeline_state.sim_data.states.quat[0, 0],
                               obs=state.info["terminal_proprioception"],
                               time=(tick + 1) * env.dt, actions=action,
                               reward=state.reward, metrics=state.metrics, active=True,
                               failed=state.done, done=state.done, outcome=state.info["outcome"])
                    rows.append(jax.tree.map(np.asarray, row))
                    if bool(state.done):
                        break
            finally:
                worker.close()
            diag = dict(difficulty=difficulty, case=case, scenario_id=scenario_id,
                        commands=commands, trajectories=trajectories,
                        missing_command_steps=unavailable, rejected_commands=rejected,
                        wall_seconds=time.monotonic() - tic,
                        rpc_p95_s=float(np.percentile(worker.latencies, 95)))
            save_report(worker.directory / "diagnostics.json", diag)
            final = {**rows[-1], "active": np.asarray(False)}
            rows.extend([final] * (env.episode_length - len(rows)))
            trace = jax.tree.map(lambda *values: np.stack(values), *rows)
            label = {"scenario_id": scenario_id, **env.bank.labels(scenario_id)}
            return trace, label, diag

        def run_slot(difficulty_index, difficulty, case):
            port = ports.get()
            try:
                return run_case(difficulty_index, difficulty, case, port)
            finally:
                ports.put(port)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for difficulty_index, difficulty in enumerate(DIFFICULTIES):
                traces, labels = [None] * count, [None] * count
                futures = {pool.submit(run_slot, difficulty_index, difficulty, case): case
                           for case in range(count)}
                rec.phase("evaluating", difficulty=difficulty, completed_cases=0)
                for finished, future in enumerate(as_completed(futures), 1):
                    case = futures[future]
                    trace, label, diag = future.result()
                    traces[case], labels[case] = trace, label
                    diagnostics.append(diag)
                    total_trajectories += diag["trajectories"]
                    save_report(rec.path / "native-progress.json", diagnostics)
                    rec.phase("evaluating", difficulty=difficulty, completed_cases=finished)
                trace = jax.tree.map(lambda *values: np.stack(values, axis=1), *traces)
                cells[difficulty] = summarize_cell(trace, labels, env.dt, env.duration)
                from drone_playground.evaluation.navigation import select_episodes
                selection = select_episodes({"cells": {difficulty: cells[difficulty]}})
                export_navigation_replays(env, {difficulty: trace}, rec.path / "rollouts",
                                          case_indices=selection)
                save_report(rec.path / "eval" / (difficulty + ".json"), cells[difficulty])
        report = combine_cells(cells)
        report.update(split=split, episodes_per_difficulty=count, method=method,
                      scene_bank_sha256=env.bank.digest(), native_trajectories=total_trajectories,
                      sensor_calibration=env.sensor_calibration, parameters_frozen=True,
                      quality_passed=None, quality_rule="No user-approved numerical quality threshold",
                      diagnostics=diagnostics, native_modification="none; ROS parameters only")
        save_report(rec.path / "eval" / "report.json", report)
        if total_trajectories == 0:
            raise RuntimeError("Native process produced no trajectories in the entire evaluation")
        rec.finish("completed", engineer_passed=True, full_budget_completed=True,
                   num_trials=report["num_trials"], arrived=report["arrived"], quality_passed=None)
        return report
    except BaseException as exc:
        rec.finish("failed", error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close()
