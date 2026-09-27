#!/usr/bin/env python3
"""Rebuild P5 budget/cell/episode tables without running or changing a method."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 8_388_608
DIFFICULTIES = ("easy", "medium", "hard")
COUNTS = {"dev": 32, "heldout": 128}
METHODS = (("depth", "ppo"), ("depth", "dva"), ("lidar", "ppo"),
           ("lidar", "dva"), ("depth", "ego"), ("lidar", "super"))
OUTCOMES = ("arrived", "collision", "out_of_bounds", "numerical_failure", "timeout")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes, total):
    """95% episode-sampling interval; it does not represent training-seed variance."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [centre-half, centre+half]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="v2")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    target = ROOT / "docs" / "verification" / ("p5-results-" + args.revision)
    target.mkdir(parents=True, exist_ok=True)
    budgets, cells, episodes, issues, artifacts = [], [], [], [], []
    banks = {}
    for task in ("static", "dynamic"):
        for sensor, method in METHODS:
            learned = method in ("ppo", "dva")
            base = (f"p5-formal-{task}-{sensor}-{method}-seed0-{args.revision}" if learned
                    else f"p5-formal-{task}-{method}-{args.revision}")
            train_path = ROOT / "experiments" / base
            expected_parameter = None
            checkpoint = None
            if learned:
                result = read(train_path / "result.json") or {}
                state = read(train_path / "state.json") or {}
                best = read(train_path / "checkpoints" / "best.json") or {}
                training_config = (read(train_path / "manifest.json") or {}).get("config", {})
                actual = result.get("actual_steps")
                complete = (result.get("status") == "completed" and actual == BUDGET
                            and result.get("full_budget_completed") is True)
                budget = dict(task=task, sensor=sensor, method=method, seed=0, run_id=base,
                              requested_steps=BUDGET, actual_steps=actual,
                              last_reported_step=state.get("step"), full_budget_completed=complete,
                              status=result.get("status", state.get("status", "missing")),
                              best_step=best.get("step"), best_selection_split=best.get("selection_split"),
                              elapsed_seconds=result.get("elapsed_seconds", result.get("elapsed_s")),
                              actor_parameter_delta_l2=result.get("actor_parameter_delta_l2"))
                metrics = result.get("trainer_metrics", {})
                budget["net_update_seconds"] = metrics.get("net_update_seconds")
                budget["compile_and_first_update_seconds"] = metrics.get("compile_and_first_update_seconds")
                budget["trainer_reported_sps"] = metrics.get("training/sps")
                algorithm_config = training_config.get("algorithm", {})
                budget["discounting"] = algorithm_config.get("discounting")
                budget["learning_rate"] = algorithm_config.get("learning_rate")
                budget["rollout_horizon"] = algorithm_config.get(
                    "horizon_length", algorithm_config.get("unroll_length"))
                budgets.append(budget)
                if not complete:
                    issues.append(f"{base}: training budget incomplete")
                if best:
                    checkpoint = train_path / "checkpoints" / best["path"]
                    meta = read(checkpoint.with_suffix(".json")) or {}
                    expected_parameter = meta.get("parameter_sha256")
                    valid = (checkpoint.exists() and digest(checkpoint) == meta.get("sha256")
                             and best.get("selection_split") == "dev")
                    budget["checkpoint_verified"] = valid
                    budget["checkpoint"] = str(checkpoint.relative_to(ROOT))
                    if not valid:
                        issues.append(f"{base}: checkpoint digest/development selection mismatch")
                elif complete:
                    issues.append(f"{base}: missing best checkpoint")
            for split, count in COUNTS.items():
                run_id = base + "-" + split
                path = ROOT / "experiments" / run_id
                original_run_id = run_id
                replacement = ROOT / "experiments" / (run_id + "-archive-v1")
                repair = read(replacement / "archive-repair.json")
                if not (path / "traces/index.json").exists() and repair:
                    original_config = copy.deepcopy((read(path / "manifest.json") or {}).get("config", {}))
                    replacement_config = copy.deepcopy((read(replacement / "manifest.json") or {}).get("config", {}))
                    for config in (original_config, replacement_config):
                        config.pop("run_id", None)
                        if not learned:
                            config.get("policy", {}).pop("port", None)
                    repair["configuration_identity_verified"] = original_config == replacement_config
                    worker_equal = True
                    if not learned:
                        old_worker, new_worker = path / "ros_bridge.py", replacement / "ros_bridge.py"
                        worker_equal = (old_worker.exists() and new_worker.exists()
                                        and digest(old_worker) == digest(new_worker))
                        repair["native_worker_identity_verified"] = worker_equal
                    if (repair.get("source_run") != run_id
                            or not repair.get("same_scene_bank")
                            or not repair.get("same_parameter_digest")
                            or not repair["configuration_identity_verified"] or not worker_equal):
                        issues.append(f"{run_id}: archive repair identity mismatch")
                    else:
                        path, run_id = replacement, replacement.name
                report_path = path / "eval" / "report.json"
                report = read(report_path) or {}
                result = read(path / "result.json") or {}
                manifest = read(path / "manifest.json") or {}
                complete = (result.get("status") == "completed"
                            and result.get("full_budget_completed") is True
                            and report.get("parameters_frozen") is True
                            and report.get("num_trials") == count * 3
                            and report.get("split") == split)
                if not complete:
                    issues.append(f"{run_id}: independent evaluation incomplete")
                archive = read(path / "traces/index.json") or {}
                if report:
                    banks.setdefault((task, split), set()).add(report.get("scene_bank_sha256"))
                    if learned and (not expected_parameter or report.get("parameter_sha256") != expected_parameter):
                        issues.append(f"{run_id}: frozen parameter digest mismatch")
                    if learned and checkpoint and Path(report.get("checkpoint", "")) != checkpoint:
                        issues.append(f"{run_id}: not the development-selected checkpoint")
                    if not learned and report.get("native_trajectories", 0) <= 0:
                        issues.append(f"{run_id}: no native trajectory evidence")
                    artifacts.append(dict(run_id=run_id, task=task, split=split,
                                          report=str(report_path.relative_to(ROOT)),
                                          original_run_id=original_run_id, archive_repair=repair,
                                          sha256=digest(report_path), code=manifest.get("code"),
                                          scene_bank_sha256=report.get("scene_bank_sha256"),
                                          parameters_frozen=report.get("parameters_frozen"),
                                          native_trajectories=report.get("native_trajectories")))
                for difficulty_index, difficulty in enumerate(DIFFICULTIES):
                    cell = report.get("cells", {}).get(difficulty)
                    row = dict(task=task, sensor=sensor, method=method, split=split,
                               difficulty=difficulty, run_id=run_id, expected_trials=count,
                               status="completed" if complete and cell else "incomplete",
                               quality_passed=None)
                    if cell:
                        cases = cell.get("episodes", [])
                        valid = (cell.get("num_trials") == count and len(cases) == count
                                 and [x["case"] for x in cases] == list(range(count))
                                 and [x["scenario_id"] for x in cases] ==
                                 list(range(difficulty_index*count, (difficulty_index+1)*count)))
                        if not valid:
                            issues.append(f"{run_id}/{difficulty}: case identity/count mismatch")
                            row["status"] = "invalid"
                        if sum(cell.get(k, 0) for k in OUTCOMES) != count:
                            issues.append(f"{run_id}/{difficulty}: outcomes do not partition denominator")
                        row.update({k: cell[k] for k in (
                            "num_trials", *OUTCOMES, "success_rate", "collision_rate",
                            "constrained_time_mean_s", "success_time_mean_s", "min_clearance_m",
                            "return_mean")})
                        row["success_ci95_low"], row["success_ci95_high"] = wilson(cell["arrived"], count)
                        diagnostics = [x for x in report.get("diagnostics", [])
                                       if x["difficulty"] == difficulty]
                        if diagnostics:
                            row["native_trajectories"] = sum(x["trajectories"] for x in diagnostics)
                            row["missing_command_steps"] = sum(x["missing_command_steps"] for x in diagnostics)
                            row["rejected_commands"] = sum(x["rejected_commands"] for x in diagnostics)
                            row["rpc_case_p95_max_s"] = max(x["rpc_p95_s"] for x in diagnostics)
                            row["episode_wall_sum_s"] = sum(x["wall_seconds"] for x in diagnostics)
                        for episode in cases:
                            episodes.append(dict(task=task, sensor=sensor, method=method,
                                                 split=split, run_id=run_id, **episode))
                        replay_files = sorted((path / "rollouts" / difficulty).rglob("*.mj_unroll"))
                        row["replay_count"] = len(replay_files)
                        row["example_replay"] = str(replay_files[0].relative_to(ROOT)) if replay_files else None
                        if not replay_files:
                            issues.append(f"{run_id}/{difficulty}: missing replay")
                        archive_cell = archive.get("cells", {}).get(difficulty, {})
                        archive_path = path / "traces" / archive_cell.get("path", "missing.npz")
                        archive_valid = (
                            archive_cell.get("episodes") == count
                            and archive_cell.get("episode_steps") == [x["steps"] for x in cases]
                            and archive_path.is_file()
                            and digest(archive_path) == archive_cell.get("sha256"))
                        row["all_case_archive_verified"] = archive_valid
                        row["archive"] = str(archive_path.relative_to(ROOT)) if archive_valid else None
                        if not archive_valid:
                            issues.append(f"{run_id}/{difficulty}: all-case trajectory archive incomplete")
                    cells.append(row)
    for (task, split), hashes in banks.items():
        if None in hashes or len(hashes) != 1:
            issues.append(f"{task}/{split}: scene banks differ across methods")
    for task in ("static", "dynamic"):
        if banks.get((task, "dev"), set()) & banks.get((task, "heldout"), set()):
            issues.append(f"{task}: development and heldout scene banks overlap by full digest")
    scene_audit = read(target / "scene-splits.json") or {}
    if scene_audit.get("passed") is not True:
        issues.append("Per-instance train/dev/heldout geometry-motion audit incomplete")
    else:
        expected_banks = {(x["task"], x["split"]): x["bank_digest"] for x in scene_audit["banks"]}
        for artifact in artifacts:
            if artifact["scene_bank_sha256"] != expected_banks.get((artifact["task"], artifact["split"])):
                issues.append(artifact["run_id"] + ": actual scene bank differs from audited bank")
        for source in scene_audit["training_sources"]:
            if digest(ROOT / "experiments" / source["run_id"] / "manifest.json") != source["manifest_sha256"]:
                issues.append(source["run_id"] + ": training manifest changed since scene audit")
        if digest(ROOT / "src/drone_playground/tasks/scenes/navigation.py") != scene_audit["generator_sha256"]:
            issues.append("Scene generator changed since split audit")
    totals = {split: sum(x.get("num_trials", 0) for x in cells
                         if x["split"] == split and x["status"] == "completed") for split in COUNTS}
    finished = not issues and totals == {"dev": 1152, "heldout": 4608}
    summary = dict(revision=args.revision, generated_at=datetime.now(timezone.utc).isoformat(),
                   matrix_complete=finished, training_units_complete=sum(x["full_budget_completed"] for x in budgets),
                   training_units_expected=8, training_interactions_expected=8*BUDGET,
                   training_interactions_completed=sum(x["actual_steps"] or 0 for x in budgets
                                                       if x["full_budget_completed"]),
                   cell_counts={split: sum(x["status"] == "completed" and x["split"] == split for x in cells)
                                for split in COUNTS},
                   episode_counts=totals, quality_passed=None,
                   quality_rule="No pre-registered numerical quality threshold; performance is reported, not certified.",
                   training_seeds=[0], ci_scope="Wilson 95% over episodes; excludes training-seed variation.",
                   issues=issues, budgets=budgets, cells=cells, artifacts=artifacts)
    write_json(target / "summary.json", summary)
    write_csv(target / "training.csv", budgets)
    write_csv(target / "cells.csv", cells)
    write_csv(target / "episodes.csv", episodes)
    units = []
    for task in ("static", "dynamic"):
        for sensor, method in METHODS:
            for split, per_cell in COUNTS.items():
                matching = [x for x in cells if (x["task"], x["sensor"], x["method"], x["split"])
                            == (task, sensor, method, split)]
                valid = len(matching) == 3 and all(x["status"] == "completed" for x in matching)
                unit = dict(task=task, sensor=sensor, method=method, split=split,
                            status="completed" if valid else "incomplete", expected_trials=3*per_cell)
                if valid:
                    unit.update({key: sum(x[key] for x in matching) for key in ("num_trials", *OUTCOMES)})
                    unit["success_rate"] = unit["arrived"] / unit["num_trials"]
                    unit["constrained_time_mean_s"] = sum(x["constrained_time_mean_s"] for x in matching) / 3
                    successful = [x["arrival_time_s"] for x in episodes
                                  if (x["task"], x["sensor"], x["method"], x["split"])
                                  == (task, sensor, method, split) and x["arrived"]]
                    unit["success_time_mean_s"] = sum(successful) / len(successful) if successful else None
                units.append(unit)
    write_csv(target / "units.csv", units)
    lines = [f"# P5 正式矩阵（{args.revision}）", "",
             f"状态：{'完整' if finished else '尚未完整'}。8 个训练单元已完成 {summary['training_units_complete']} 个；"
             f"独立开发 {totals['dev']}/1152 回合，留出 {totals['heldout']}/4608 回合。",
             "", "以下数值来自原始逐回合记录。缺失格保留“未完成”，不填零、不剔除失败。质量门槛尚未冻结，quality_passed=null。",
             "单训练种子为 0；置信区间仅描述场景回合抽样，不估计跨训练种子稳定性。",
             "", "## 训练预算", "", "| Task | Sensor | Method | 实际交互 | 最佳开发步 | 状态 |",
             "|---|---|---|---:|---:|---|"]
    for row in budgets:
        lines.append(f"| {row['task']} | {row['sensor']} | {row['method']} | {row['actual_steps'] or '—'} | "
                     f"{row['best_step'] if row['best_step'] is not None else '—'} | {row['status']} |")
    lines += ["", "## 留出结果总览", "", "每行包含三个难度各 128 回合。成功条件时间仅统计到达回合，完整分母仍为 384。", "",
              "| Task | Sensor | Method | 到达/N | 碰撞 | 越界 | 数值失败 | 超时 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in units:
        if row["split"] != "heldout":
            continue
        prefix = f"| {row['task']} | {row['sensor']} | {row['method']} |"
        if row["status"] == "completed":
            lines.append(prefix + f" {row['arrived']}/{row['num_trials']} | {row['collision']} | "
                         f"{row['out_of_bounds']} | {row['numerical_failure']} | {row['timeout']} |")
        else:
            lines.append(prefix + " 未完成 | — | — | — | — |")
    if finished:
        lines += ["", "## 图表", "", "![开发集训练曲线](development-curves.png)",
                  "", "![开发集失败类型与回报](development-diagnostics.png)",
                  "", "![正式留出成功率](heldout-matrix.png)"]
    for split in COUNTS:
        lines += ["", f"## {'正式留出' if split == 'heldout' else '最终独立开发'} 36 格", "",
                  "| Task | Sensor | Method | Difficulty | N | 成功率 | 碰撞率 | 受限时间/s |",
                  "|---|---|---|---|---:|---:|---:|---:|"]
        for row in cells:
            if row["split"] != split:
                continue
            label = f"| {row['task']} | {row['sensor']} | {row['method']} | {row['difficulty']} |"
            if row["status"] == "completed":
                lines.append(label + f" {row['num_trials']} | {row['success_rate']:.3f} | "
                             f"{row['collision_rate']:.3f} | {row['constrained_time_mean_s']:.2f} |")
            else:
                lines.append(label + " — | 未完成 | — | — |")
    lines += ["", "## 比较边界", "",
              "- 学习组四帧压缩观测：D435 每帧 300 点；MID360 每帧 120 点。原生规划器读取完整 120×90 深度或每帧 24000 条 MID360 射线的有效点。",
              "- 两模态策略观测均为 2420 维，但编码器、视场和采样率不同；四帧覆盖的历史时长也不同，不宣称网络或信息带宽相同。",
              "- 同传感器 PPO/D.VA 共享输入及执行链；EGO/SUPER 是完整方法比较，不能将差值归因于单独的规划算法。",
              "- 算法各自采用冻结的折扣和更新配方，实际值见 training.csv 和运行 manifest；这不是只替换损失函数的受控消融。",
              "- 全部执行 Crazyflow first_principles/cf2x_L250；MuJoCo 用于几何核验与回放，ROS 仅存在于原生规划器外部工作进程。",
              "- 未加动态预测器；动态任务评测原生重规划表现。质量合格与工程完成分列。",
              "- v1 在地面碰撞遗漏被发现后中止并保留，未并入 v2。",
              "- 67,108,864 仅为 v2 正式训练交互；工程运行与已中止 v1 另计。v1 PPO 最后记录至少 2,097,152 交互，不能把该额外开销抹去。",
              "- 场景生成器、种子和参数预先固定；完整清单在运行初始化时落盘。scene-splits.json 按不含种子/编号的几何与运动参数指纹检查三个集合交集，并核对实际评测库摘要。训练清单为事后确定性重建，不宣称所有清单都在训练前物化保存。",
              "- 早期 v2 评测仅导出代表轨迹；缺少全回合逐帧归档的单元按固定规则重评，使用后缀 archive-v1 的完整证据。选择规则只看归档缺失，不看得分；原结果及前后差异保留在 archive-repair.json，新增训练交互为零。",
              "", "## 证据", "",
              "training.csv：预算与选模；units.csv：各方法汇总（成功条件时间按全部成功回合加权）；cells.csv：逐格指标、原生 RPC 延迟和回放位置；episodes.csv：全分母逐回合；summary.json：报告 SHA256、代码身份、场景身份与完整性问题。",
              "", "图表通过 scripts/plot_p5.py 重建；RScope 读取校验通过 scripts/verify_p5_replays.py 重建。",
              "规划器 rpc_case_p95_max_s 是各回合 RPC 延迟第 95 百分位的最大值，包含通信和等待，不等于纯求解耗时。"]
    if issues:
        lines += ["", "## 尚缺证据", ""] + ["- " + x for x in issues]
    (target / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ("matrix_complete", "training_units_complete",
                                            "cell_counts", "episode_counts")}))
    if args.require_complete and not finished:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
