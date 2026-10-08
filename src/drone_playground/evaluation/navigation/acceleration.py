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

from drone_playground.artifacts.console import capture_console
from drone_playground.artifacts.record import RunRecorder
from drone_playground.artifacts.reporting import save_report, tree_digest
from drone_playground.artifacts.training_state import load_training_state
from drone_playground.evaluation.navigation.metrics import summarize_trace
from drone_playground.networks.factory import build_network
from drone_playground.numerics import euclidean_norm
from drone_playground.visualization.navigation_replay import export_case


def _select(mask, new, old):
    return jax.tree.map(
        lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (n.ndim - mask.ndim)), n, o),
        new,
        old,
    )


def sample_navigation_interval(task, bank, state, command, timestamp, outcome):
    """Observe the paper's original interval map on the common physical clock."""
    commands = jnp.broadcast_to(command, (task.substeps, *command.shape))
    state, timestamp, outcome, _, minimum = task.transition.checked(
        state,
        commands,
        task.transition_events(bank, arrival_sampling=getattr(task, "arrival_sampling", "policy")),
        timestamp=timestamp,
        outcome=outcome,
        sample_from_start=True,
    )
    arrived = euclidean_norm(bank.goal - state.pos) <= task.goal_radius
    outcome = jnp.where((outcome == 0) & arrived, 1, outcome)
    return state, timestamp, outcome, minimum


def make_rollout(task, network, bank, initial_state=None):
    """Build a compiled navigation rollout using the acceleration policy."""
    count = bank.num_instances

    @jax.jit
    def run(params, speeds):
        state = task.initial_state(bank) if initial_state is None else initial_state
        hidden = jnp.zeros((count, network.hidden_size))
        clock = jnp.zeros(count)
        outcome = jnp.zeros(count, jnp.int32)

        def advance(carry, index):
            physical, memory, timestamp, result = carry
            active = result == 0
            # Scene time is physical episode time. Each still-running case has this tick's time.
            points, valid, proprio, _ = task.measure(
                bank, physical, jnp.full((count,), index * task.dt), speeds
            )
            body_action, next_memory = network.apply(params, points, valid, proprio, memory)
            command = task.command(body_action, physical)
            call_time = jnp.where(active, index * task.dt, timestamp)
            nxt, new_time, new_result, clearance = sample_navigation_interval(
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
            proprio, _ = task.observation.proprioception(
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
            return jax.lax.cond(jnp.any(carry[3] == 0), advance, inactive, carry, index)

        _, trace = jax.lax.scan(
            step,
            (state, hidden, clock, outcome),
            jnp.arange(task.episode_length),
        )
        return trace

    return run


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
        (
            f"冻结检查点：`{report['checkpoint']}`。"
            f"所选权重训练至 {report['trained_updates']} 次更新。"
        ),
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
        lines.append("完整训练运行预算以独立运行结果为准；本表记录当前冻结权重的迭代数。")
    lines += [
        f"到达 {report['arrived']}/{report['num_trials']}，碰撞 {report['collision']}，"(
            f"越界 {report['out_of_bounds']}，超时 {report['timeout']}，"
            f"数值失败 {report['numerical_failure']}。"
        ),
        "",
        (
            "| 场景 | 命令速度（米/秒） | 结果 | 结束时间（秒） | "
            "路径长（米） | 终点距离（米） | 最小净空（米） |"
        ),
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene_id']} | {row['command_speed_m_s']:g} | "
            f"{outcome_names[row['outcome']]} | "
            f"{row['elapsed_s']:.3f} | {row['path_length_m']:.3f} | "
            f"{row['final_goal_distance_m']:.3f} | {row['min_clearance_m']:.3f} |"
        )
    lines += [
        "",
        (
            f"策略／状态转移频率 {report['policy_hz']:g} Hz，"
            f"步内碰撞采样 {report['collision_sampling_hz']:g} Hz；"
            "无碰撞步末状态与训练映射一致。"
        ),
        (
            "机器报告、逐帧归档与逐场景 RScope 回放保留全部失败，"
            "图元和运动来自验收后的 navigation 目录。"
        ),
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))


def evaluate_pointcloud(config, root: Path, run_id: str):
    """Evaluate a frozen point-cloud navigation policy on benchmark scenes."""
    from drone_playground.environments.factory import build_environment

    state, metadata = load_training_state(config["checkpoint"])
    trained = metadata["config"]
    for slot in ("method", "network"):
        if config[slot] != trained[slot]:
            raise ValueError(
                f"Frozen-policy evaluation changed the {slot} slot; declare a separate ablation"
            )
    for slot in ("sensor", "controller", "dynamics"):
        if config["env"][slot] != trained["env"][slot]:
            raise ValueError(f"Frozen-policy evaluation changed env.{slot}")
    if config["algorithm"]["gradient"] != trained["algorithm"]["gradient"]:
        raise ValueError("Frozen-policy evaluation changed its derivative identity")
    if config["env"]["freq"] != trained["env"]["freq"]:
        raise ValueError("The recurrent policy tick must match its training time semantics")
    if config["env"]["scene"]["name"] != "navigation":
        raise ValueError(
            "Choose "
            "experiment=navigation/differentiable_pointcloud_acceleration_benchmark "
            "for the transfer benchmark"
        )
    repeats = config["evaluation"]["episodes"]
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Evaluation episodes per scene must be a positive integer")
    cfg = copy.deepcopy(config)
    cfg["env"]["task"]["duration"] = float(cfg["evaluation"]["duration"])
    cfg["mode"] = "eval"
    role = cfg["evaluation"]["role"]
    env = build_environment(cfg, cfg["runtime"]["device"], role, repeats)
    task = env.task
    network = build_network(cfg["network"])
    bank, manifest = task.scene.build()
    ids = list(manifest["scene_ids"])
    if ids != list(cfg["evaluation"]["scene_ids"]):
        raise ValueError("Scene slot and evaluation scene IDs disagree")
    indices = jnp.tile(jnp.arange(bank.num_instances), repeats)
    bank = bank.select(indices)
    ids *= repeats

    seed_start = cfg["evaluation"].get("seed_start")
    seeds = list(
        range(
            config["runtime"]["scene_seed_" + role] if seed_start is None else seed_start,
            (config["runtime"]["scene_seed_" + role] if seed_start is None else seed_start)
            + len(ids),
        )
    )
    initial = task.initial_state(bank).replace(
        measurement_key=jnp.asarray([jax.random.PRNGKey(seed) for seed in seeds])
    )
    initial_record = None
    from drone_playground.benchmarks import load_protocol

    initial_spec = cfg["evaluation"].get("initial_conditions")
    if not initial_spec and cfg["evaluation"].get("protocol"):
        initial_spec = load_protocol(cfg["evaluation"]["protocol"])["initial_conditions"]
    if initial_spec:
        from drone_playground.evaluation.navigation.cases import navigation_resets

        reset = navigation_resets(bank, np.arange(len(ids)), seeds, initial_spec, task.body_radius)
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
        training_provenance(source_run, cfg["checkpoint"], int(state.updates), before)
        if source_run
        else None
    )
    selected_is_final = int(state.updates) == trained["training"]["policy_updates"]
    run = make_rollout(task, network, bank, initial)
    started = time.monotonic()
    with RunRecorder(root, run_id, cfg, task_id="DP-005-navigation-transfer") as rec:
        rec.record_environment(env)
        with capture_console(rec.path / "console.log"):
            save_report(rec.path / "components.json", env.component_identity)
            save_report(rec.path / "scene-manifest.json", manifest)
            save_report(rec.path / "sensor-calibration.json", task.sensor_calibration)
            cells = {}
            replays = []
            for speed in cfg["evaluation"]["speeds"]:
                label = f"speed-{float(speed):g}"
                rec.phase("evaluating", speed_m_s=float(speed), scenes=ids)
                trace = jax.tree.map(
                    np.asarray,
                    run(state.params, jnp.full((len(ids),), float(speed))),
                )
                report = summarize_trace(trace, ids, speed, task.duration, bank.start)
                for episode, seed in zip(report["episodes"], seeds, strict=True):
                    episode["seed"] = seed
                cells[label] = report
                archive = rec.path / "traces" / f"{label}.npz"
                archive.parent.mkdir(parents=True, exist_ok=True)
                arrays = {k: v for k, v in trace.items() if k != "metrics"}
                arrays.update({f"metric_{k}": v for k, v in trace["metrics"].items()})
                np.savez_compressed(archive, **arrays)
                report["trace_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
                save_report(rec.path / "eval" / f"{label}.json", report)
                if cfg["evaluation"].get("record_replays", False):
                    for case, scene_id in enumerate(ids):
                        directory = rec.path / "rollouts" / label / scene_id / f"seed-{seeds[case]}"
                        path = export_case(task, trace, case, directory)
                        replays.append(
                            dict(
                                scene_id=scene_id,
                                speed_m_s=float(speed),
                                path=str(path.relative_to(rec.path)),
                                frames=report["episodes"][case]["steps"] + 1,
                                transitions=report["episodes"][case]["steps"],
                                initial_frame_included=True,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
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
            episodes = [row for cell in cells.values() for row in cell["episodes"]]
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
                scene_ids=ids,
                policy_hz=task.freq,
                dynamics_transition_hz=task.freq,
                collision_sampling_hz=task.physics_freq,
                collision_interpolation=(
                    "same tick-start paper step evaluated at intermediate times"
                ),
                duration_s=task.duration,
                body_radius_m=task.body_radius,
                goal_radius_m=task.goal_radius,
                dynamics=task.dynamics.forward,
                elapsed_s=time.monotonic() - started,
                reward_semantics=(
                    "distance progress for replay only; training uses trajectory losses"
                ),
                evaluation_scope=(
                    "fixed navigation geometry transfer; deterministic cells; protocol "
                    "identity below"
                ),
                protocol=cfg["evaluation"].get("protocol"),
                arrival_sampling=task.arrival_sampling,
                training_action_delay_ms=trained.get("runtime", {}).get("action_delay_ms"),
            )
            from drone_playground.runtime.timing import measure_decision

            single_bank, physical = (
                bank.select(jnp.array([0])),
                jax.tree.map(lambda value: value[:1], initial),
            )
            points, valid, proprio, _ = task.measure(
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
            report.update(measure_decision(lambda: decide(state.params), task.dt))
            report["sensor_timing"] = env.sensor_timing
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
