#!/usr/bin/env python3
"""Verify a point-cloud training/evaluation delivery without changing its runs.

The default gate requires the full 50,000-update budget. Explicit --allow-stage
validates an intermediate checkpoint and labels the result as a stage. This
script only reads recorded evidence and writes an optional new output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from drone_playground.runs.layout import resolve_artifact  # noqa: E402

OUTCOMES = ("arrived", "collision", "out_of_bounds", "numerical_failure", "timeout")
OUTCOME_LABELS = dict(zip(OUTCOMES, ("到达", "碰撞", "越界", "数值失败", "超时"), strict=True))
SLOTS = ("method", "env", "algorithm", "network", "objective", "training", "runtime")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pipeline():
    path = Path(__file__).with_name("run_pointcloud_pipeline.py")
    spec = importlib.util.spec_from_file_location("pointcloud_pipeline_for_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_path(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Artifact path leaves its evaluation directory: {relative}")
    return path


def verify_delivery(training_run, evaluation_run, *, require_complete=True):
    """Validate counts, policy identity, raw terminal states, and replay proofs."""
    training_run = resolve_artifact(training_run).resolve()
    evaluation_run = resolve_artifact(evaluation_run).resolve()
    pipeline = load_pipeline()
    declared = read_json(training_run / "result.json")
    updates = declared["actual_updates"]
    if require_complete and updates != pipeline.TARGET_UPDATES:
        raise ValueError(f"Full training budget is pending: {updates}/{pipeline.TARGET_UPDATES}")
    training = pipeline.verify_training_result(training_run, expected_updates=updates)
    report = pipeline.verify_evaluation_result(
        evaluation_run, expected_selected=training["selected"], expected_training_run=training_run
    )
    selected_metadata = read_json(Path(training["selected"]["checkpoint"]).with_suffix(".json"))
    if (
        selected_metadata["updates"] != training["selected"]["updates"]
        or report["trained_updates"] != selected_metadata["updates"]
    ):
        raise ValueError("Selected checkpoint update age disagrees with its metadata or report")
    if report["catalog_sha256"] != digest(ROOT / "assets/scenes/navigation/catalog.json"):
        raise ValueError("Navigation8 catalog digest differs from the accepted local scene file")
    if report["training_budget_completed"] is not training["full_budget_completed"]:
        raise ValueError("Evaluation training-budget flag disagrees with its source run")
    config = read_json(training_run / "resolved-config.json")
    if (
        config["training"]["num_envs"] != 32
        or config["algorithm"]["horizon_length"] != 160
        or config["training"]["policy_updates"] != 50000
    ):
        raise ValueError(
            "Recorded training configuration differs from the paper reconstruction budget"
        )
    required_protocol = {
        "policy_hz": 10,
        "dynamics_transition_hz": 10,
        "collision_sampling_hz": 500,
        "duration_s": 40,
        "body_radius_m": 0.07,
        "goal_radius_m": 0.5,
        "dynamics": "point_mass_lag",
        "scene_ids": list(pipeline.SCENES),
    }
    for key, expected in required_protocol.items():
        if report.get(key) != expected:
            raise ValueError(f"Navigation8 protocol mismatch: {key}")

    rows = report["episodes"]
    by_case = {(row["scene_id"], float(row["command_speed_m_s"])): row for row in rows}
    expected_cases = {(scene, speed) for scene in pipeline.SCENES for speed in pipeline.SPEEDS}
    if set(by_case) != expected_cases or len(by_case) != len(rows):
        raise ValueError("Repeated or missing scene-speed case")
    counts = {name: 0 for name in OUTCOMES}
    artifacts = []

    def remember(path, kind):
        artifacts.append(
            {"kind": kind, "path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
        )

    for speed in pipeline.SPEEDS:
        label = f"speed-{speed:g}"
        cell = report["cells"][label]
        archive = evaluation_run / "traces" / f"{label}.npz"
        if digest(archive) != cell["trace_sha256"]:
            raise ValueError(f"Trace digest mismatch: {label}")
        if cell["num_trials"] != 8 or len(cell["episodes"]) != 8:
            raise ValueError(f"Incomplete cell denominator: {label}")
        remember(archive, "trajectory")
        with np.load(archive, allow_pickle=False) as trace:
            active = np.asarray(trace["active"], dtype=bool)
            outcome = np.asarray(trace["outcome"])
            times = np.asarray(trace["time"])
            positions = np.asarray(trace["pos"])
            if (
                active.ndim != 2
                or active.shape[1] != 8
                or outcome.shape != active.shape
                or times.shape != active.shape
                or positions.shape != (*active.shape, 3)
            ):
                raise ValueError(f"Trace shape mismatch: {label}")
            for case, scene in enumerate(pipeline.SCENES):
                row = by_case[(scene, speed)]
                count = int(active[:, case].sum())
                if (
                    count < 1
                    or not active[:count, case].all()
                    or active[count:, case].any()
                    or row["steps"] != count
                ):
                    raise ValueError(f"Active-transition count mismatch: {scene}/{speed}")
                code = int(outcome[count - 1, case])
                if code < 1 or code > 5 or np.any(outcome[: count - 1, case] != 0):
                    raise ValueError(f"Invalid first terminal event: {scene}/{speed}")
                result_name = OUTCOMES[code - 1]
                if row["outcome"] != result_name:
                    raise ValueError(f"Reported outcome disagrees with trace: {scene}/{speed}")
                if any(row[name] != (name == result_name) for name in OUTCOMES):
                    raise ValueError(f"Outcome flags disagree: {scene}/{speed}")
                if row != cell["episodes"][case]:
                    raise ValueError(
                        f"Per-cell report disagrees with combined report: {scene}/{speed}"
                    )
                episode_times = times[:count, case]
                if (
                    not np.isfinite(episode_times).all()
                    or np.any(np.diff(episode_times) < 0)
                    or episode_times[0] < 0
                    or episode_times[-1] > 40 + 1e-5
                    or not np.isclose(episode_times[-1], row["elapsed_s"], atol=1e-5)
                    or not np.isfinite(positions[:count, case]).all()
                ):
                    raise ValueError(f"Invalid archived state or time: {scene}/{speed}")
                counts[result_name] += 1
    if any(counts[name] != report[name] for name in OUTCOMES):
        raise ValueError("Summary outcome counts differ from all raw terminal events")
    if not np.isclose(report["success_rate"], counts["arrived"] / 24):
        raise ValueError("Success rate differs from the complete denominator")

    replay_rows = read_json(evaluation_run / "rollouts/index.json")["replays"]
    replay_cases = set()
    replay_paths = set()
    for replay in replay_rows:
        key = (replay["scene_id"], float(replay["speed_m_s"]))
        if key in replay_cases or key not in expected_cases:
            raise ValueError("Duplicate or unexpected replay case")
        replay_cases.add(key)
        path = checked_path(evaluation_run, replay["path"])
        expected_parent = (evaluation_run / "rollouts" / f"speed-{key[1]:g}" / key[0]).resolve()
        if path in replay_paths or path.parent != expected_parent:
            raise ValueError(f"Replay path is repeated or assigned to another case: {key}")
        replay_paths.add(path)
        for filename in ("scene.xml", "rscope_meta.pkl"):
            companion = checked_path(path.parent, filename)
            if not companion.is_file() or companion.stat().st_size == 0:
                raise ValueError(f"Replay companion is missing or empty: {companion}")
            remember(companion, "replay_companion")
        checksum = digest(path)
        proof = read_json(path.parent / "readback-verification.json")
        if checksum != replay["sha256"] or checksum != proof["replay_sha256"]:
            raise ValueError(f"Replay digest mismatch: {key}")
        frames = by_case[key]["steps"] + int(replay.get("initial_frame_included", False))
        if (
            replay["frames"] != frames
            or proof["frames"] != frames
            or not replay.get("readback_verified")
            or not proof.get("positions_and_action_channels_match")
            or not proof.get("model_xml_recompiled")
        ):
            raise ValueError(f"Replay readback proof mismatch: {key}")
        if "transitions" in proof and proof["transitions"] != by_case[key]["steps"]:
            raise ValueError(f"Replay transition count mismatch: {key}")
        remember(path, "replay")
    if replay_cases != expected_cases or len(replay_rows) != 24:
        raise ValueError("Replay set does not contain all 24 cases")
    for path, kind in (
        (training_run / "result.json", "training_result"),
        (evaluation_run / "eval/report.json", "evaluation_report"),
        (Path(training["selected"]["checkpoint"]), "selected_checkpoint"),
    ):
        remember(path, kind)
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "experiment_completed": training["full_budget_completed"],
        "scope": "full_budget" if training["full_budget_completed"] else "intermediate_stage",
        "method_identity": config.get("reproduction", {}).get("identity"),
        "training_run": str(training_run),
        "evaluation_run": str(evaluation_run),
        "training": {
            key: training[key]
            for key in (
                "actual_updates",
                "target_updates",
                "actual_steps",
                "target_steps",
                "full_budget_completed",
            )
        },
        "selection": training["selected"],
        "outcomes": {"num_trials": 24, **counts},
        "success_rate": counts["arrived"] / 24,
        "protocol": required_protocol,
        "catalog_sha256": report["catalog_sha256"],
        "verified_trace_count": 3,
        "verified_replay_count": 24,
        "replay_verification_scope": "file digests and evaluator-generated readback proofs",
        "slots": {slot: config[slot] for slot in SLOTS},
        "reproduction": config.get("reproduction", {}),
        "episodes": rows,
        "artifacts": artifacts,
    }


def describe_method(slots):
    """Describe the recorded settings, including separately named ablations."""
    sensor = slots["env"].get("sensor", {})
    azimuth, elevation = sensor.get("azimuth_count"), sensor.get("elevation_count")
    rays = (
        f"{int(azimuth) * int(elevation)}条射线点云测量"
        if azimuth and elevation
        else "配置指定的点云测量"
    )
    network = slots["network"]
    channels = network.get("point_channels")
    encoder = "逐点编码" + ("→".join(map(str, channels)) if channels else "（通道由模块配置确定）")
    recurrent = f"GRU{network['hidden_size']}" if "hidden_size" in network else "配置指定的时序网络"
    forward = slots["env"]["execution"]["dynamics"].get("forward")
    dynamics = "一阶滞后质点动力学" if forward == "point_mass_lag" else f"前向模型{forward}"
    backward = slots["algorithm"]["gradient"].get("transition")
    derivative = {"direct": "直接状态导数", "exponential": "指数衰减状态导数"}.get(
        backward, f"反向规则{backward}"
    )
    algorithm = slots["algorithm"]
    training = (
        f"{algorithm['horizon_length']}步时间反传"
        if algorithm.get("name") == "pointcloud_bptt"
        else f"训练算法{algorithm.get('name')}"
    )
    state_gradient = sensor.get("state_gradient")
    boundary = (
        "传感器位姿梯度按配置分离，编码器与策略参数通过控制和物理链训练。"
        if state_gradient == "detached"
        else f"传感器状态导数配置：{state_gradient if state_gradient is not None else '由模块默认值确定'}。"
    )
    return "＋".join((rays, encoder, recurrent, dynamics, derivative, training)), boundary


def write_delivery(summary, destination):
    """Write only to a new folder; an existing run or report is never overwritten."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)

    def write_json(name, value):
        (destination / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

    write_json("summary.json", summary)
    write_json("module-slots.json", summary["slots"])
    write_json("reconstruction-assumptions.json", summary["reproduction"])
    with (destination / "episodes.csv").open("w", newline="") as handle:
        rows = summary["episodes"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    training, selected = summary["training"], summary["selection"]
    description = (
        "完整预算与冻结测试已核验"
        if summary["experiment_completed"]
        else "阶段训练与冻结测试已核验，完整预算继续进行"
    )
    lines = [
        "# 点云论文重建与Navigation8交付核验",
        "",
        description + "。",
        "",
        f"实际训练：{training['actual_updates']}/{training['target_updates']} 次更新，"
        f"{training['actual_steps']}/{training['target_steps']} 次交互。",
        f"开发集选择的权重来自第 {selected['updates']} 次更新；参数摘要 `{selected['parameter_sha256']}`。",
        f"训练运行：`{summary['training_run']}`。测试运行：`{summary['evaluation_run']}`。",
        "",
        "| 终止结果 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| {OUTCOME_LABELS[name]} | {summary['outcomes'][name]} |" for name in OUTCOMES)
    lines += [
        "",
        "分母为8张固定场景×3个命令速度，共24个确定性试次。每个失败均计入统计。",
        "已核对3份数值轨迹、24份回放及原评测进程生成的读回证明。",
        "",
        "| 场景 | 4米/秒 | 6米/秒 | 8米/秒 |",
        "|---|---|---|---|",
    ]
    by_case = {
        (row["scene_id"], float(row["command_speed_m_s"])): row for row in summary["episodes"]
    }
    for scene in summary["protocol"]["scene_ids"]:
        labels = [OUTCOME_LABELS[by_case[(scene, speed)]["outcome"]] for speed in (4, 6, 8)]
        lines.append(f"| {scene} | " + " | ".join(labels) + " |")
    method, boundary = describe_method(summary["slots"])
    lines += [
        "",
        f"对应实际组合：{method}。",
        "",
        "实际模块参数见 `module-slots.json`；公开参数与重建假设见 `reconstruction-assumptions.json`。",
        boundary,
        "本结果属于论文公开信息重建的Navigation8迁移评测；作者场景、实机、跨模态及薄障碍实验各自具有独立条件。",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-run", type=Path, default=ROOT / "experiments/paper-pointcloud-seed0-full-v1"
    )
    parser.add_argument(
        "--evaluation-run",
        type=Path,
        default=ROOT / "experiments/paper-pointcloud-navigation8-full-v1",
    )
    parser.add_argument("--allow-stage", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = verify_delivery(
            args.training_run, args.evaluation_run, require_complete=not args.allow_stage
        )
        if args.output:
            write_delivery(summary, args.output)
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"verified": False, "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from error
    print(
        json.dumps(
            {
                "verified": True,
                "experiment_completed": summary["experiment_completed"],
                "training": summary["training"],
                "outcomes": summary["outcomes"],
                "output": str(args.output) if args.output else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
