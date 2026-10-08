"""Frozen control-task evaluation of the explicitly qualified paper-policy transfer."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.artifacts.reporting import save_report, tree_digest
from drone_playground.control.transition import delayed_schedule
from drone_playground.environments.tasks.tracking.events import _select
from drone_playground.evaluation.racing import summarize_race
from drone_playground.evaluation.tracking.metrics import summarize_trials
from drone_playground.networks.factory import build_network


class ControlEvaluator:
    """One compiled rollout per fixed run role; no weights or normalization are updated."""

    def __init__(self, task, network, seeds):
        self.task, self.network, self.seeds = task, network, list(seeds)
        self.keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, jnp.uint32))
        self.delay, self.requested = jax.vmap(
            lambda key: tuple(value[0] for value in task.delays(jax.random.fold_in(key, 73), 1))
        )(self.keys)
        self._run = jax.jit(self._rollout)

    def _rollout(self, params):
        task, network = self.task, self.network
        count = len(self.seeds)
        initial = task.initial(self.keys)
        hidden = jnp.zeros((count, network.hidden_size))
        previous = jnp.zeros((count, 3))
        clock = jnp.zeros(count)
        outcome = jnp.zeros(count, jnp.int32)
        gates = jnp.zeros(count, jnp.int32)

        def step(carry, index):
            state, memory, last, timestamp, result, passed = carry
            active = result == 0
            points, valid, proprio = task.measure(state, timestamp)
            action, next_memory = network.apply(params, points, valid, proprio, memory)
            command = task.command(action, state)
            nxt, next_time, next_result, next_gates, clearance = task.transition.checked(
                state,
                delayed_schedule(command, last, self.delay, task.substeps),
                task.transition_events(),
                timestamp=timestamp,
                outcome=result,
                memory=passed,
            )
            # Hover/tracking require their full prescribed duration; racing requires gates.
            terminal = 5 if task.name == "racing" else 1
            next_result = jnp.where(
                (index == task.episode_length - 1) & (next_result == 0),
                terminal,
                next_result,
            )
            target, _ = task.reference(next_time)
            error = jnp.linalg.norm(nxt.pos - target, axis=-1)
            failed = (next_result >= 2) & (next_result <= 4)
            row = dict(
                pos=nxt.pos,
                rotation=nxt.rotation,
                velocity=nxt.vel,
                time=next_time,
                observation_pos=state.pos,
                observation_rotation=state.rotation,
                obs=proprio,
                actions=jnp.where(active[:, None], command, 0.0),
                reward=jnp.where(active, -jnp.square(error), 0.0),
                active=active,
                done=next_result != 0,
                failed=failed,
                outcome=next_result,
                metrics=dict(
                    tracking_error=error,
                    clearance=clearance,
                    success=(next_result == 1).astype(jnp.float32),
                    collision=(next_result == 2).astype(jnp.float32),
                    failure=failed.astype(jnp.float32),
                    gates_passed=next_gates.astype(jnp.float32),
                    reference_x=target[:, 0],
                    reference_y=target[:, 1],
                    reference_z=target[:, 2],
                ),
            )
            return (
                nxt,
                _select(active, next_memory, memory),
                _select(active, command, last),
                next_time,
                next_result,
                next_gates,
            ), row

        _, trace = jax.lax.scan(
            step,
            (initial, hidden, previous, clock, outcome, gates),
            jnp.arange(task.episode_length),
        )
        return trace

    def run(self, params):
        before = tree_digest(params)
        trace = jax.tree.map(np.asarray, self._run(params))
        if before != tree_digest(params):
            raise RuntimeError("Frozen point-cloud control evaluation changed parameters")
        summarize = summarize_race if self.task.name == "racing" else summarize_trials
        report = summarize(trace, self.seeds, self.task.dt)
        for case, row in enumerate(report["episodes"]):
            length = row["steps"]
            duration = float(trace["time"][length - 1, case])
            row.update(
                duration_s=duration,
                outcome=int(trace["outcome"][length - 1, case]),
                requested_delay_ms=float(self.requested[case]),
                effective_delay_ms=float(self.delay[case] * self.task.physics_dt * 1000),
            )
            if self.task.name == "racing":
                row["completion_time_s"] = duration if row["completed"] else None
        if self.task.name == "racing":
            times = [r["duration_s"] for r in report["episodes"] if r["completed"]]
            report["completion_time_mean_s"] = float(np.mean(times)) if times else None
        report.update(
            parameters_frozen=True,
            parameter_sha256=before,
            task=self.task.name,
            recipe_identity=self.task.settings["provenance"],
            plant="point_mass_lag; not the Crazyflow rigid-body plant",
            policy_hz=self.task.freq,
            physics_hz=self.task.physics_freq,
            requested_delay_ms=np.asarray(self.requested).tolist(),
            effective_delay_ms=(np.asarray(self.delay) * self.task.physics_dt * 1000).tolist(),
            geometry=self.task.geometry_identity,
            reference_sha256=hashlib.sha256(np.asarray(self.task.references).tobytes()).hexdigest(),
            sensor_calibration=self.task.sensor_calibration,
            evaluation_scope=(
                "fixed canonical reference/geometry, independent initial-state and delay seeds"
            ),
        )
        from drone_playground.runtime.timing import measure_decision, sensor_schedule

        physical = self.task.initial(self.keys[:1])
        points, valid, proprio = self.task.measure(physical, jnp.zeros(1))
        memory = jnp.zeros((1, self.network.hidden_size))
        decide = jax.jit(
            lambda p, x, mask, obs, hidden, state: self.task.command(
                self.network.apply(p, x, mask, obs, hidden)[0], state
            )
        )
        report.update(
            measure_decision(
                lambda: decide(params, points, valid, proprio, memory, physical),
                self.task.dt,
            )
        )
        report["sensor_timing"] = sensor_schedule(self.task.sensor, self.task.freq)
        return report, trace


def export_replays(task, trace, report, directory):
    """Keep every case, an initial frame, the terminal frame, and no post-terminal padding."""
    import mujoco
    from rscope import rollout

    from drone_playground.visualization.navigation_scene import active_indices, create_replay_model
    from drone_playground.visualization.rscope_io import export_rollout

    directory = Path(directory)
    model = create_replay_model(task, 0)
    obstacles = np.asarray(task.bank.origin)[0, active_indices(task.bank, 0)]
    rows = []
    for case, episode in enumerate(report["episodes"]):
        length = episode["steps"]
        single = {
            name: trace[name][:length, case : case + 1]
            for name in ("pos", "time", "obs", "reward", "actions")
        }
        initial_position = trace["observation_pos"][0, case]
        matrices = np.concatenate(
            [
                trace["observation_rotation"][0, case : case + 1],
                trace["rotation"][:length, case],
            ],
            axis=0,
        )
        for name in single:
            first = single[name][:1].copy()
            if name == "pos":
                first[0, 0] = initial_position
            elif name in ("time", "reward", "actions"):
                first[...] = 0
            single[name] = np.concatenate([first, single[name]], axis=0)
        single["quat"] = Rotation.from_matrix(matrices).as_quat()[:, None]
        target, _ = task.reference(jnp.array(0.0))
        first_error = float(np.linalg.norm(initial_position - np.asarray(target)))
        single["metrics"] = {}
        for name, values in trace["metrics"].items():
            selected = values[:length, case : case + 1]
            first = selected[:1].copy()
            if name == "tracking_error":
                first[...] = first_error
            elif name == "clearance":
                first[...] = float(task.clearance(jnp.asarray(initial_position[None]))[0])
            elif name.startswith("reference_"):
                first[...] = float(target["xyz".index(name[-1])])
            else:
                first[...] = 0
            single["metrics"][name] = np.concatenate([first, selected], axis=0)
        single["obstacle_pos"] = np.broadcast_to(obstacles, (length + 1, 1, *obstacles.shape))
        case_dir = directory / f"case-{case:03d}"
        path = export_rollout(model, case_dir, single)
        rollout.rollouts.clear()
        rollout.num_evals = 0
        try:
            rollout.append_unroll(path)
            restored = rollout.rollouts[-1]
            np.testing.assert_allclose(restored.mocap_pos[:, 0, 0], single["pos"][:, 0], atol=1e-6)
            for axis in range(3):
                np.testing.assert_allclose(
                    restored.metrics[f"action/{axis}"],
                    single["actions"][..., axis],
                    atol=1e-6,
                )
            mujoco.MjModel.from_xml_path(str(case_dir / "scene.xml"))
        finally:
            rollout.rollouts.clear()
            rollout.num_evals = 0
        row = dict(
            case=case,
            seed=episode["seed"],
            path=str(path.relative_to(directory)),
            transitions=length,
            frames=length + 1,
            final_time_s=float(single["time"][-1, 0]),
            initial_frame_included=True,
            readback_verified=True,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        save_report(case_dir / "readback-verification.json", row)
        rows.append(row)
    save_report(directory / "index.json", dict(replays=rows))
    return rows


def evaluate_pointcloud_control(config, root, run_id):
    """Evaluate acceleration-based point-cloud control on tracking benchmarks."""
    from drone_playground.artifacts.record import RunRecorder
    from drone_playground.artifacts.training_state import load_training_state
    from drone_playground.environments.factory import build_environment

    state, metadata = load_training_state(config["checkpoint"])
    source = metadata["config"]
    for field in ("method", "network", "algorithm", "env"):
        if config[field] != source[field]:
            raise ValueError(f"Frozen point-cloud control contract differs: {field}")
    role = config["evaluation"]["role"]
    count = int(config["evaluation"]["episodes"])
    if role not in ("train", "eval") or count < 1:
        raise ValueError("Select a valid evaluation role and positive episode count")
    seed = config["evaluation"].get("seed_start")
    seed = config["runtime"]["scene_seed_" + role] if seed is None else int(seed)
    env = build_environment(config, config["runtime"]["device"], role, count)
    task = env.task
    network = build_network(config["network"])
    started = time.monotonic()
    try:
        with RunRecorder(
            root, run_id, config, task_id="final-acceptance/pointcloud-control"
        ) as rec:
            rec.record_environment(env)
            evaluator = ControlEvaluator(task, network, range(seed, seed + count))
            report, trace = evaluator.run(state.params)
            from drone_playground.benchmarks import apply_quality

            apply_quality(report, config)
            report.update(
                role=role,
                checkpoint=str(Path(config["checkpoint"]).resolve()),
                trained_updates=int(state.updates),
                elapsed_seconds=time.monotonic() - started,
            )
            save_report(rec.path / "components.json", env.component_identity)
            save_report(rec.path / "geometry.json", task.geometry_identity)
            save_report(rec.path / "eval/report.json", report)
            arrays = {name: value for name, value in trace.items() if name != "metrics"}
            arrays.update({"metric_" + name: value for name, value in trace["metrics"].items()})
            np.savez_compressed(rec.path / "eval/trace.npz", **arrays)
            if config["evaluation"].get("record_replays", False):
                export_replays(task, trace, report, rec.path / "rollouts")
            rec.finish(
                "completed",
                num_trials=count,
                completed=report["completed"],
                quality_passed=report["quality_passed"],
                engineer_passed=True,
                full_budget_completed=True,
                parameters_frozen=True,
            )
            return report
    finally:
        env.close()
