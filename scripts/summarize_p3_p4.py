"""Rebuild phase tables from completed, digest-checked runs without executing training."""

import json
from datetime import datetime, timezone
from pathlib import Path

from drone_playground.evaluation.reporting import controller_result, learning_result

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = json.loads((ROOT / "docs/verification/p3-matrix.json").read_text())
    tracking = []
    for cell in original["rows"]:
        row = learning_result(ROOT, cell["run_id"])
        if cell["run_id"] == "p3-random-ppo-so_rpy_rotor_drag-seed0-v1":
            failed_retry = learning_result(ROOT, "p3-random-ppo-so_rpy_rotor_drag-seed0-v2")
            retry = learning_result(ROOT, "p3-random-ppo-so_rpy_rotor_drag-seed0-v3")
            retry["previous_attempts"] = [row, failed_retry]
            retry["recipe_change"] = (
                "Original PPO hyperparameters restored. finite-square-v1 explicitly counts "
                "numerically divergent integration as failure before it contaminates the learner; "
                "v1 and reduced-learning-rate v2 failures retained."
            )
            row = retry
        row["reused_from_p2"] = bool(cell.get("reused"))
        tracking.append(row)
    racing = [
        learning_result(ROOT, f"p4-racing-{method}-first_principles-seed0-v1")
        for method in ("ppo", "apg", "shac")
    ]
    controllers = [
        controller_result(ROOT, f"p4-racing-{method}-heldout-v2")
        for method in ("attitude-mpc", "sampling-mpc")
    ]
    all_rows = tracking + racing + controllers
    output = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        completed=all(row["experiment_completed"] for row in all_rows),
        expected_tracking_cells=24,
        expected_racing_learners=3,
        expected_controllers=2,
        training_seeds=[0],
        tracking=tracking,
        racing=racing,
        controllers=controllers,
        completed_experiments=sum(row["experiment_completed"] for row in all_rows),
        quality_passed_experiments=sum(row.get("quality_passed", False) for row in all_rows),
        interpretation={
            "scope": "model-matched tracking; LSY Level0 preset-reference racing with native disturbances",
            "race_trials": "fixed course; unique disturbance seeds, not 128 different tracks",
            "timings": "shared-server measurements; differing budgets; controller delays were measured, not injected",
            "selected_step_zero": "untrained policy selected on dev: task-quality failure, not a successfully trained policy",
            "failed_rmse": "error over active steps until failure; interpret with completion count",
            "recovery_recipe": "one PPO cell uses explicit finite-square-v1 numerical failure handling; original learning hyperparameters restored; both previous failures retained",
        },
    )
    destination = ROOT / "docs/verification/p3-p4-results.json"
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = [
        "# P3/P4 实际结果",
        "",
        "由 `scripts/summarize_p3_p4.py` 从逐回合记录和检查点校验值重建。",
        "",
        f"实验完成 {output['completed_experiments']}/29；达到当前任务门槛 {output['quality_passed_experiments']}/29。",
        "",
        "## P3：同动力学训练与评测",
        "",
        "训练种子0；开发32回合选模；留出128回合独立重评。",
        "",
        "| 任务 | 动力学 | 方法 | 训练交互 | 留出完成 | 位置均方根误差（米） | 备注 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    task_names = {"figure8": "八字", "random": "随机样条"}
    for row in tracking:
        if not row["experiment_completed"]:
            lines.append(f"| 待完成 | — | — | — | — | — | {row['run_id']}：{row['status']} |")
            continue
        h = row["heldout"]
        note = (
            "初始策略被开发集选中，质量未达标"
            if row["initial_policy_selected"]
            else ("达标" if row["quality_passed"] else "有效低分")
        )
        if "recipe_change" in row:
            note += "；数值失效边界修复v3，原学习参数；v1/v2失败保留"
        lines.append(
            f"| {task_names[row['task']]} | `{row['dynamics']}` | {row['algorithm'].upper()} | {row['actual_steps']:,} | {h['completed']}/128 | {h['rmse_all_mean']:.6f} | {note} |"
        )
    lines += [
        "",
        "失败回合的误差仅覆盖存活步数，结合完成率解释。PPO/APG/SHAC的网络与训练预算各有冻结配方。",
        "",
        "## P4：LSY Level0 竞速轨迹跟踪",
        "",
        "固定作者门序 `[1,2,3,4,2]`、原生扰动和碰撞判定，实际穿越5次指定门。",
        "",
        "| 方法 | 训练交互 | 留出完成 | 平均完成时间（秒） | 位置误差（米） | 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in racing:
        if not row["experiment_completed"]:
            lines.append(f"| {row['run_id']} | — | — | — | — | {row['status']} |")
            continue
        h = row["heldout"]
        time = h.get("completion_time_mean_s")
        time_text = f"{time:.4f}" if time is not None else "—"
        state_text = (
            "初始策略被开发集选中，训练质量未达标"
            if row["initial_policy_selected"]
            else ("达标" if row["quality_passed"] else "有效低分")
        )
        lines.append(
            f"| {row['algorithm'].upper()} | {row['actual_steps']:,} | {h['completed']}/128 | {time_text} | {h['rmse_all_mean']:.6f} | {state_text} |"
        )
    for row in controllers:
        if not row["experiment_completed"]:
            lines.append(f"| {row['run_id']} | 在线求解 | — | — | — | {row['status']} |")
            continue
        time = row["completion_time_mean_s"]
        time_text = f"{time:.4f}" if time is not None else "—"
        lines.append(
            f"| {row['controller']} | 在线求解 | {row['completed']}/128 | {time_text} | {row['rmse_all_mean']:.6f} | {'达标' if row['quality_passed'] else '有效低分'} |"
        )
    lines += [
        "",
        "控制器耗时包含共享服务器竞争，真实耗时未注入仿真。完整每步求解状态和延迟保存在各分片记录。",
        "",
        "## 检查点与回放",
        "",
    ]
    for row in tracking + racing:
        if row["experiment_completed"]:
            lines += [
                f"- `{row['run_id']}`：`{row['checkpoint']}`；回放在同运行的 `independent-heldout/rollouts/`。"
            ]
            if row["initial_policy_selected"]:
                lines += [
                    f"  训练结束策略另见同运行 `checkpoints/step-{row['actual_steps']:010d}.pkl`，"
                    f"对应回放 `rollouts/step-{row['actual_steps']:010d}/`；保留开发集原选模结果。"
                ]
    for row in controllers:
        if row["experiment_completed"]:
            lines += [
                f"- `{row['run_id']}`：回放在 `experiments/{row['run_id']}/rollouts/shard-0/` 至 `shard-3/`。"
            ]
    lines += [
        "",
        "查看回放使用已交付的 RScope Viewer。报告包含所有32/128试次，回放文件保存固定前4及最差试次的子集。",
        "",
    ]
    (ROOT / "docs/verification/p3-p4-results.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                key: output[key]
                for key in ("completed", "completed_experiments", "quality_passed_experiments")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
