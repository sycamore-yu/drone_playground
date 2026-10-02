"""Frozen paper-policy transfer evaluation on the explicitly selected scene catalog."""

import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

from drone_playground.networks.factory import build_network

from drone_playground.artifacts.console import capture_console
from drone_playground.artifacts.record import RunRecorder
from drone_playground.artifacts.reporting import save_report, tree_digest
from drone_playground.artifacts.training_state import load_training_state
from drone_playground.environments.scenes.geometry import clearance_and_collision, euclidean_norm
from drone_playground.environments.tasks.navigation.rigid_body import OUTCOME_NAMES


def _select(mask, new, old):
    return jax.tree.map(
        lambda n, o: jnp.where(
            mask.reshape(mask.shape + (1,) * (n.ndim - mask.ndim)), n, o
        ),
        new,
        old,
    )


def advance_checked(task, bank, state, command, timestamp, outcome):
    """Preserve the training step while sampling its dense path for collisions.

    Each sample evaluates the same tick-start state at an intermediate elapsed
    time. The last sample therefore equals the 0.1-second training transition;
    collision sampling never silently changes the integration discretization.
    """
    substeps = task.physics_freq // task.freq
    base_state = state
    base_time = timestamp

    def substep(carry, offset):
        physical, clock, result, minimum = carry
        active = result == 0
        elapsed = (offset + 1) * (task.dt / substeps)
        proposed = task.model.step(base_state, command, elapsed)
        next_time = base_time + elapsed
        finite = jnp.all(jnp.isfinite(proposed.vector()), axis=-1)
        centre = proposed.pos + jnp.einsum(
            "...ij,j->...i", proposed.rotation, jnp.array([0.0, 0.0, 0.005])
        )
        clearance, collision = jax.vmap(
            lambda i, t, p: task.task_definition.clearance(bank, i, p, t)
        )(jnp.arange(bank.num_instances), next_time, centre)
        _, _, _, result_next = task.task_definition.events(
            bank, proposed.pos, bank.goal, collision, ~finite
        )
        if getattr(task, "arrival_sampling", "policy") != "physics":
            result_next = jnp.where(result_next == 1, 0, result_next)
        physical = _select(active & finite, proposed, physical)
        clock = jnp.where(active, next_time, clock)
        result = jnp.where(active, result_next, result)
        # Keep finite geometry evidence when the proposed state itself is invalid.
        minimum = jnp.where(
            active & finite, jnp.minimum(minimum, clearance), minimum
        )
        return (physical, clock, result, minimum), None

    initial_minimum = jnp.full((bank.num_instances,), jnp.inf)
    (state, timestamp, outcome, minimum), _ = jax.lax.scan(
        substep,
        (state, timestamp, outcome, initial_minimum),
        jnp.arange(substeps),
    )
    arrived = euclidean_norm(bank.goal - state.pos) <= task.goal_radius
    outcome = jnp.where((outcome == 0) & arrived, 1, outcome)
    minimum = jnp.where(jnp.isfinite(minimum), minimum, 0.0)
    return state, timestamp, outcome, minimum


def make_rollout(task, network, bank, initial_state=None):
    count = bank.num_instances

    @jax.jit
    def run(params, speeds):
        state = (
            task.initial_state(bank) if initial_state is None else initial_state
        )
        hidden = jnp.zeros((count, network.hidden_size))
        clock = jnp.zeros(count)
        outcome = jnp.zeros(count, jnp.int32)

        def advance(carry, index):
            physical, memory, timestamp, result = carry
            active = result == 0
            # Scene time is physical episode time. Each still-running case has this tick's time.
            points, valid, proprio, _ = task.observation(
                bank, physical, index * task.dt, speeds
            )
            body_action, next_memory = network.apply(
                params, points, valid, proprio, memory
            )
            command = task.command(body_action, physical)
            call_time = jnp.where(active, index * task.dt, timestamp)
            nxt, new_time, new_result, clearance = advance_checked(
                task, bank, physical, command, call_time, result
            )
            new_result = jnp.where(
                (index == task.episode_length - 1) & (new_result == 0),
                5,
                new_result,
            )
            memory = jnp.where(active[:, None], next_memory, memory)
            distance = euclidean_norm(bank.goal - nxt.pos)
            progress = euclidean_norm(bank.goal - physical.pos) - distance
            row = dict(
                pos=nxt.pos,
                velocity=nxt.vel,
                rotation=nxt.rotation,
                observation_pos=physical.pos,
                observation_velocity=physical.vel,
                observation_rotation=physical.rotation,
                observation_time=jnp.full((count,), index * task.dt),
                obs=proprio,
                time=new_time,
                actions=jnp.where(active[:, None], command, 0.0),
                reward=jnp.where(active, progress, 0.0),
                active=active,
                done=new_result != 0,
                outcome=new_result,
                metrics=dict(
                    clearance=clearance,
                    goal_distance=distance,
                    speed=euclidean_norm(nxt.vel),
                ),
            )
            return (nxt, memory, new_time, new_result), row

        def inactive(carry, index):
            physical, memory, timestamp, result = carry
            proprio, _ = task.observer.proprioception(
                physical, bank.goal, speeds, task.body_radius
            )
            zeros = jnp.zeros(count)
            row = dict(
                pos=physical.pos,
                velocity=physical.vel,
                rotation=physical.rotation,
                observation_pos=physical.pos,
                observation_velocity=physical.vel,
                observation_rotation=physical.rotation,
                observation_time=jnp.full((count,), index * task.dt),
                obs=proprio,
                time=timestamp,
                actions=jnp.zeros_like(physical.pos),
                reward=zeros,
                active=jnp.zeros(count, bool),
                done=result != 0,
                outcome=result,
                metrics=dict(
                    clearance=zeros,
                    goal_distance=euclidean_norm(bank.goal - physical.pos),
                    speed=euclidean_norm(physical.vel),
                ),
            )
            return (physical, memory, timestamp, result), row

        def step(carry, index):
            return jax.lax.cond(
                jnp.any(carry[3] == 0), advance, inactive, carry, index
            )

        _, trace = jax.lax.scan(
            step,
            (state, hidden, clock, outcome),
            jnp.arange(task.episode_length),
        )
        return trace

    return run


def summarize_trace(trace, scene_ids, speed, duration, start):
    episodes = []
    for case, scene_id in enumerate(scene_ids):
        length = int(np.asarray(trace["active"])[:, case].sum())
        if length < 1:
            raise ValueError(
                "Every evaluation case must contain an actual transition"
            )
        result = int(np.asarray(trace["outcome"])[length - 1, case])
        result = 5 if result == 0 else result
        positions = np.vstack(
            [np.asarray(start)[case], np.asarray(trace["pos"])[:length, case]]
        )
        speed_values = np.asarray(trace["metrics"]["speed"])[:length, case]
        elapsed = float(np.asarray(trace["time"])[length - 1, case])
        episodes.append(
            dict(
                scene_id=scene_id,
                command_speed_m_s=float(speed),
                outcome=OUTCOME_NAMES[result],
                arrived=result == 1,
                collision=result == 2,
                out_of_bounds=result == 3,
                numerical_failure=result == 4,
                timeout=result == 5,
                steps=length,
                elapsed_s=elapsed,
                arrival_time_s=elapsed if result == 1 else None,
                path_length_m=float(
                    np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum()
                ),
                peak_speed_m_s=float(speed_values.max()),
                mean_speed_m_s=float(speed_values.mean()),
                min_clearance_m=float(
                    np.asarray(trace["metrics"]["clearance"])[
                        :length, case
                    ].min()
                ),
                final_goal_distance_m=float(
                    np.asarray(trace["metrics"]["goal_distance"])[
                        length - 1, case
                    ]
                ),
            )
        )
    counts = {
        key: sum(int(row[key]) for row in episodes)
        for key in (
            "arrived",
            "collision",
            "out_of_bounds",
            "numerical_failure",
            "timeout",
        )
    }
    return dict(
        num_trials=len(episodes),
        **counts,
        success_rate=counts["arrived"] / len(episodes),
        constrained_time_mean_s=float(
            np.mean(
                [
                    row["elapsed_s"] if row["arrived"] else duration
                    for row in episodes
                ]
            )
        ),
        episodes=episodes,
    )


def export_case(task, trace, case, directory):
    from drone_playground.visualization.navigation_scene import (
        active_indices,
        create_replay_model,
        obstacle_track,
    )
    from drone_playground.visualization.rscope_io import export_rollout

    length = int(np.asarray(trace["active"])[:, case].sum())
    single = {
        key: np.asarray(trace[key])[:length, case : case + 1]
        for key in ("pos", "time", "obs", "actions", "reward")
    }
    matrices = np.asarray(trace["rotation"])[:length, case]
    single["quat"] = Rotation.from_matrix(matrices).as_quat()[:, None]
    single["metrics"] = {
        k: np.asarray(v)[:length, case : case + 1]
        for k, v in trace["metrics"].items()
    }
    # Include the real pre-action state. RScope needs two timestamps even when
    # the first transition terminates; frame count and transition count differ.
    initial_position = np.asarray(trace["observation_pos"])[0, case]
    initial_rotation = np.asarray(trace["observation_rotation"])[0, case]
    initial_centre = initial_position + initial_rotation @ np.array(
        [0.0, 0.0, 0.005]
    )
    initial_clearance = float(
        clearance_and_collision(
            task.bank, case, 0.0, initial_centre, task.body_radius
        )[0]
    )
    initial_metrics = {
        "clearance": initial_clearance,
        "goal_distance": float(
            np.linalg.norm(np.asarray(task.bank.goal[case]) - initial_position)
        ),
        "speed": float(
            np.linalg.norm(np.asarray(trace["observation_velocity"])[0, case])
        ),
    }
    initial_fields = {
        "pos": initial_position[None, None],
        "quat": Rotation.from_matrix(initial_rotation).as_quat()[None, None],
        "time": np.zeros_like(single["time"][:1]),
        "obs": single["obs"][:1],
        "actions": np.zeros_like(single["actions"][:1]),
        "reward": np.zeros_like(single["reward"][:1]),
    }
    for name, initial in initial_fields.items():
        single[name] = np.concatenate([initial, single[name]], axis=0)
    single["metrics"] = {
        name: np.concatenate(
            [np.full_like(values[:1], initial_metrics[name]), values], axis=0
        )
        for name, values in single["metrics"].items()
    }
    active = active_indices(task.bank, case)
    single["obstacle_pos"] = obstacle_track(
        task.bank, case, single["time"][:, 0]
    )[:, active, None, :].swapaxes(1, 2)
    model = create_replay_model(task, case)
    directory = Path(directory)
    path = export_rollout(model, directory, single)
    import mujoco
    from rscope import rollout

    rollout.rollouts.clear()
    rollout.num_evals = 0
    rollout.append_unroll(path)
    restored = rollout.rollouts[-1]
    np.testing.assert_allclose(
        restored.mocap_pos[:, 0, 0], single["pos"][:, 0], atol=1e-6
    )
    for axis in range(single["actions"].shape[-1]):
        np.testing.assert_allclose(
            restored.metrics[f"action/{axis}"],
            single["actions"][..., axis],
            atol=1e-6,
        )
    loaded_model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    if loaded_model.nmocap != model.mj_model.nmocap:
        raise ValueError("Replay model lost an obstacle or drone motion body")
    save_report(
        directory / "readback-verification.json",
        dict(
            frames=length + 1,
            transitions=length,
            initial_frame_included=True,
            action_channels=single["actions"].shape[-1],
            positions_and_action_channels_match=True,
            model_xml_recompiled=True,
            motion_bodies=loaded_model.nmocap,
            replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )
    rollout.rollouts.clear()
    rollout.num_evals = 0
    return path


def training_provenance(run, checkpoint, selected_updates, parameter_sha256):
    """Distinguish a completed training budget from the age of its selected policy."""
    run = Path(run).resolve()
    path = run / "result.json"
    result = json.loads(path.read_text())
    selected = result["selected"]
    complete = bool(result["full_budget_completed"])
    if result["status"] != ("completed" if complete else "paused"):
        raise ValueError("Training source has an invalid terminal status")
    if (
        Path(selected["checkpoint"]).resolve() != Path(checkpoint).resolve()
        or selected["updates"] != selected_updates
        or selected["parameter_sha256"] != parameter_sha256
        or result["actual_updates"] < selected_updates
    ):
        raise ValueError(
            "The evaluated policy is not the declared checkpoint_eval-selected checkpoint"
        )
    if complete != (result["actual_updates"] == result["target_updates"]):
        raise ValueError("Training source has inconsistent update counts")
    if (
        result["actual_steps"] * result["target_updates"]
        != result["target_steps"] * result["actual_updates"]
    ):
        raise ValueError("Training source has inconsistent interaction counts")
    return dict(
        run=str(run),
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        actual_updates=result["actual_updates"],
        target_updates=result["target_updates"],
        actual_steps=result["actual_steps"],
        target_steps=result["target_steps"],
        full_budget_completed=complete,
        selected_checkpoint_updates=selected_updates,
    )


def write_summary(report, directory):
    """Write a human-readable table and a complete per-case CSV alongside the JSON."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = report["episodes"]
    with (directory / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outcome_names = {
        "arrived": "到达",
        "collision": "碰撞",
        "out_of_bounds": "越界",
        "numerical_failure": "数值失败",
        "timeout": "超时",
    }
    lines = [
        "# 点云论文方法重建：navigation 测试",
        "",
        f"冻结检查点：`{report['checkpoint']}`。所选权重训练至 {report['trained_updates']} 次更新。",
        f"参数摘要：`{report['parameter_sha256']}`。",
        "统计单位为场景、命令速度与具名重置种子的组合；训练内评测选模，冻结 checkpoint 后评测。",
        "",
    ]
    evidence = report.get("training_run_evidence")
    if evidence:
        lines.append(
            f"来源训练运行实际 {evidence['actual_updates']}/{evidence['target_updates']} 次更新，"
            f"{evidence['actual_steps']}/{evidence['target_steps']} 次交互。"
        )
    else:
        lines.append(
            "完整训练运行预算以独立运行结果为准；本表记录当前冻结权重的迭代数。"
        )
    lines += [
        f"到达 {report['arrived']}/{report['num_trials']}，碰撞 {report['collision']}，"
        f"越界 {report['out_of_bounds']}，超时 {report['timeout']}，数值失败 {report['numerical_failure']}。",
        "",
        "| 场景 | 命令速度（米/秒） | 结果 | 结束时间（秒） | 路径长（米） | 终点距离（米） | 最小净空（米） |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene_id']} | {row['command_speed_m_s']:g} | {outcome_names[row['outcome']]} | "
            f"{row['elapsed_s']:.3f} | {row['path_length_m']:.3f} | "
            f"{row['final_goal_distance_m']:.3f} | {row['min_clearance_m']:.3f} |"
        )
    lines += [
        "",
        f"策略／状态转移频率 {report['policy_hz']:g} Hz，步内碰撞采样 {report['collision_sampling_hz']:g} Hz；无碰撞步末状态与训练映射一致。",
        "机器报告、逐帧归档与逐场景 RScope 回放保留全部失败，图元和运动来自验收后的 navigation 目录。",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))


def evaluate_pointcloud(config, root: Path, run_id: str):
    from drone_playground.composition import build_environment

    state, metadata = load_training_state(config["checkpoint"])
    trained = metadata["config"]
    for slot in ("method", "network"):
        if config[slot] != trained[slot]:
            raise ValueError(
                f"Frozen-policy evaluation changed the {slot} slot; declare a separate ablation"
            )
    for slot in ("sensor", "observation", "action", "dynamics"):
        if config["env"][slot] != trained["env"][slot]:
            raise ValueError(f"Frozen-policy evaluation changed env.{slot}")
    if config["algorithm"]["gradient"] != trained["algorithm"]["gradient"]:
        raise ValueError(
            "Frozen-policy evaluation changed its derivative identity"
        )
    if config["env"]["task"]["freq"] != trained["env"]["task"]["freq"]:
        raise ValueError(
            "The recurrent policy tick must match its training time semantics"
        )
    if config["env"]["scene"]["name"] != "navigation":
        raise ValueError(
            "Choose experiment=navigation/differentiable_pointcloud_acceleration_benchmark for the transfer benchmark"
        )
    repeats = config["evaluation"]["episodes"]
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError(
            "Evaluation episodes per scene must be a positive integer"
        )
    cfg = copy.deepcopy(config)
    cfg["env"]["task"]["duration"] = float(cfg["evaluation"]["duration"])
    cfg["mode"] = "eval"
    role = cfg["evaluation"]["role"]
    task = build_environment(cfg, cfg["runtime"]["device"], role, repeats)
    network = build_network(cfg["network"])
    bank, manifest = task.scene.build()
    ids = list(manifest["scene_ids"])
    if ids != list(cfg["evaluation"]["scene_ids"]):
        raise ValueError("Scene slot and evaluation scene IDs disagree")
    indices = jnp.tile(jnp.arange(bank.num_instances), repeats)
    bank = bank.select(indices)
    ids *= repeats
    from drone_playground.environments.environment import ROLE_SEEDS

    seed_start = cfg["evaluation"].get("seed_start")
    seeds = list(
        range(
            ROLE_SEEDS[role] if seed_start is None else seed_start,
            (ROLE_SEEDS[role] if seed_start is None else seed_start) + len(ids),
        )
    )
    initial = task.initial_state(bank).replace(
        measurement_key=jnp.asarray(
            [jax.random.PRNGKey(seed) for seed in seeds]
        )
    )
    initial_record = None
    from drone_playground.benchmarks import load_protocol

    initial_spec = cfg["evaluation"].get("initial_conditions")
    if not initial_spec and cfg["evaluation"].get("protocol"):
        initial_spec = load_protocol(cfg["evaluation"]["protocol"])[
            "initial_conditions"
        ]
    if initial_spec:
        from drone_playground.evaluation.navigation.cases import navigation_resets

        reset = navigation_resets(
            bank, np.arange(len(ids)), seeds, initial_spec, task.body_radius
        )
        initial = initial.replace(
            pos=jnp.asarray(reset["position"]),
            vel=jnp.asarray(reset["velocity"]),
            rotation=jnp.asarray(reset["rotation"]),
        )
        bank = bank.replace(start=initial.pos)
        initial_record = reset["record"]
    task.bank = bank
    before = tree_digest(state.params)
    source_run = cfg["evaluation"].get("training_run")
    provenance = (
        training_provenance(
            source_run, cfg["checkpoint"], int(state.updates), before
        )
        if source_run
        else None
    )
    selected_is_final = (
        int(state.updates) == trained["training"]["policy_updates"]
    )
    run = make_rollout(task, network, bank, initial)
    started = time.monotonic()
    with RunRecorder(
        root, run_id, cfg, task_id="DP-005-navigation-transfer"
    ) as rec:
        rec.record_environment(task)
        with capture_console(rec.path / "console.log"):
            save_report(rec.path / "components.json", task.component_identity)
            save_report(rec.path / "scene-manifest.json", manifest)
            save_report(
                rec.path / "sensor-calibration.json", task.sensor_calibration
            )
            cells = {}
            replays = []
            for speed in cfg["evaluation"]["speeds"]:
                label = f"speed-{float(speed):g}"
                rec.phase("evaluating", speed_m_s=float(speed), scenes=ids)
                trace = jax.tree.map(
                    np.asarray,
                    run(state.params, jnp.full((len(ids),), float(speed))),
                )
                report = summarize_trace(
                    trace, ids, speed, task.duration, bank.start
                )
                for episode, seed in zip(
                    report["episodes"], seeds, strict=True
                ):
                    episode["seed"] = seed
                cells[label] = report
                archive = rec.path / "traces" / f"{label}.npz"
                archive.parent.mkdir(parents=True, exist_ok=True)
                arrays = {k: v for k, v in trace.items() if k != "metrics"}
                arrays.update(
                    {f"metric_{k}": v for k, v in trace["metrics"].items()}
                )
                np.savez_compressed(archive, **arrays)
                report["trace_sha256"] = hashlib.sha256(
                    archive.read_bytes()
                ).hexdigest()
                save_report(rec.path / "eval" / f"{label}.json", report)
                if cfg["evaluation"].get("record_replays", False):
                    for case, scene_id in enumerate(ids):
                        directory = (
                            rec.path
                            / "rollouts"
                            / label
                            / scene_id
                            / f"seed-{seeds[case]}"
                        )
                        path = export_case(task, trace, case, directory)
                        replays.append(
                            dict(
                                scene_id=scene_id,
                                speed_m_s=float(speed),
                                path=str(path.relative_to(rec.path)),
                                frames=report["episodes"][case]["steps"] + 1,
                                transitions=report["episodes"][case]["steps"],
                                initial_frame_included=True,
                                sha256=hashlib.sha256(
                                    path.read_bytes()
                                ).hexdigest(),
                                readback_verified=True,
                            )
                        )
                print(
                    json.dumps(
                        dict(
                            speed=speed,
                            arrived=report["arrived"],
                            trials=len(ids),
                        )
                    ),
                    flush=True,
                )
            after = tree_digest(state.params)
            if before != after:
                raise RuntimeError("Evaluation changed frozen parameters")
            episodes = [
                row for cell in cells.values() for row in cell["episodes"]
            ]
            counts = {
                key: sum(int(row[key]) for row in episodes)
                for key in (
                    "arrived",
                    "collision",
                    "out_of_bounds",
                    "numerical_failure",
                    "timeout",
                )
            }
            report = dict(
                num_trials=len(episodes),
                **counts,
                success_rate=counts["arrived"] / len(episodes),
                episodes=episodes,
                cells=cells,
                parameters_frozen=True,
                role=role,
                initial_conditions=initial_record,
                episodes_per_scene=repeats,
                parameter_sha256=before,
                checkpoint=str(Path(cfg["checkpoint"]).resolve()),
                trained_updates=int(state.updates),
                declared_training_updates=trained["training"]["policy_updates"],
                selected_checkpoint_is_final_update=selected_is_final,
                training_budget_completed=(
                    provenance["full_budget_completed"]
                    if provenance
                    else (True if selected_is_final else None)
                ),
                training_run_evidence=provenance,
                catalog_sha256=manifest["catalog_sha256"],
                scene_ids=ids,
                policy_hz=task.freq,
                dynamics_transition_hz=task.freq,
                collision_sampling_hz=task.physics_freq,
                collision_interpolation="same tick-start paper step evaluated at intermediate times",
                duration_s=task.duration,
                body_radius_m=task.body_radius,
                goal_radius_m=task.goal_radius,
                dynamics=task.model.forward,
                elapsed_s=time.monotonic() - started,
                reward_semantics="distance progress for replay only; training uses trajectory losses",
                evaluation_scope="fixed navigation geometry transfer; deterministic cells; protocol identity below",
                protocol=cfg["evaluation"].get("protocol"),
                arrival_sampling=task.arrival_sampling,
                training_action_delay_ms=trained.get("runtime", {}).get(
                    "action_delay_ms"
                ),
            )
            from drone_playground.runtime.timing import measure_decision

            single_bank, physical = bank.select(jnp.array([0])), jax.tree.map(
                lambda value: value[:1], initial
            )
            points, valid, proprio, _ = task.observation(
                single_bank,
                physical,
                0.0,
                jnp.array([cfg["evaluation"]["speeds"][0]]),
            )
            memory = jnp.zeros((1, network.hidden_size))
            decide = jax.jit(
                lambda params: task.command(
                    network.apply(params, points, valid, proprio, memory)[0],
                    physical,
                )
            )
            report.update(
                measure_decision(lambda: decide(state.params), task.dt)
            )
            report["sensor_timing"] = task.sensor_timing
            save_report(rec.path / "eval/report.json", report)
            write_summary(report, rec.path / "eval")
            save_report(rec.path / "rollouts/index.json", dict(replays=replays))
            rec.finish(
                "completed",
                num_trials=len(episodes),
                arrived=counts["arrived"],
                parameters_frozen=True,
                training_budget_completed=report["training_budget_completed"],
            )
            return report
