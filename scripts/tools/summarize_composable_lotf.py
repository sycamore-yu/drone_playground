"""Rebuild the delivered LOTF tables and plots from immutable experiment outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"悬停": "lotf-hybrid-hover-seed0-v1", "八字跟踪": "lotf-hybrid-tracking-seed0-v1"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / "docs/verification"
    figures = target / "composable-lotf-figures"
    figures.mkdir(parents=True, exist_ok=True)
    candidates = [f for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
    if candidates:
        plt.rcParams["font.family"] = candidates[0].name
    plt.rcParams["axes.unicode_minus"] = False
    rows = []
    reader = ROOT.parents[2] / "opensource_projects/rscope-vscode/python/bridge.py"
    spec = importlib.util.spec_from_file_location("replay_verification", reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay_dirs = []
    for title, run_id in RUNS.items():
        run = ROOT / "experiments" / run_id
        result = json.loads((run / "result.json").read_text())
        best = json.loads((run / "checkpoints/best.json").read_text())
        checkpoint = run / "checkpoints" / best["path"]
        meta = json.loads(checkpoint.with_suffix(".json").read_text())
        assert digest(checkpoint) == meta["sha256"]
        records = {}
        for split, n in [("dev", 32), ("heldout", 128)]:
            path = run / f"independent-{split}/report.json"
            report = json.loads(path.read_text())
            assert report["num_trials"] == n and len(report["episodes"]) == n
            assert (
                report["parameters_frozen"]
                and report["parameter_sha256"] == meta["parameter_sha256"]
            )
            assert [r["seed"] for r in report["episodes"]] == list(
                range(20000 if split == "dev" else 30000, (20000 if split == "dev" else 30000) + n)
            )
            records[split] = {
                k: report[k]
                for k in (
                    "num_trials",
                    "completed",
                    "failed",
                    "quality_passed",
                    "rmse_all_mean",
                    "last_second_rmse_mean_m",
                    "return_mean",
                    "process_id",
                    "parameter_sha256",
                )
            }
            records[split]["report"] = str(path.relative_to(ROOT))
        assert records["dev"]["process_id"] != records["heldout"]["process_id"]
        row = dict(
            task=title,
            run_id=run_id,
            checkpoint=str(checkpoint.relative_to(ROOT)),
            sha256=meta["sha256"],
            actual_steps=result["actual_steps"],
            updates=result["updates"],
            elapsed_seconds=result["elapsed_seconds"],
            net_update_seconds=result["net_update_seconds"],
            compile_and_first_update_seconds=result["compile_and_first_update_seconds"],
            actor_parameter_delta_l2=result["actor_parameter_delta_l2"],
            **records,
        )
        assert result["full_budget_completed"] and row["actor_parameter_delta_l2"] > 0
        rows.append(row)
        logs = [
            json.loads(line)
            for line in (run / "metrics/metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]

        def series(tag):
            return (
                [x["step"] for x in logs if tag in x["metrics"]],
                [x["metrics"][tag] for x in logs if tag in x["metrics"]],
            )

        for tag, label, suffix in [
            ("training/loss", "训练损失", "loss"),
            (
                "eval/last_second_rmse_m" if title == "悬停" else "eval/rmse_all_m",
                "最后一秒位置均方根误差（米）" if title == "悬停" else "全程位置均方根误差（米）",
                "error",
            ),
        ]:
            x, y = series(tag)
            assert all(np.isfinite(y)) and len(x) > 1
            fig, axis = plt.subplots(figsize=(8, 4.5))
            axis.plot(np.asarray(x) / 1e6, y, marker="o" if suffix == "error" else None)
            axis.set(
                xlabel="训练交互（百万次）",
                ylabel=label,
                title=title + "：LOTF 高保真前向＋解析反向",
            )
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / (run_id + "-" + suffix + ".png"), dpi=180)
            plt.close(fig)
        files = sorted((run / "independent-heldout/rollouts").glob("*.mj_unroll"))
        replay = module.Replay(files[0])
        episode = replay.episode(0)
        p = np.asarray(episode["actual"])
        ref = np.asarray(episode["reference"])
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot(*p.T, label="实际轨迹")
        if np.max(np.ptp(ref, axis=0)) < 1e-6:
            axis.scatter(*ref[0], marker="x", s=80, label="悬停目标")
        else:
            axis.plot(*ref.T, linestyle="--", label="参考轨迹")
        axis.set(
            xlabel="位置 x（米）",
            ylabel="位置 y（米）",
            zlabel="高度（米）",
            title=title + "：独立留出试次 30000",
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / (run_id + "-trajectory.png"), dpi=180)
        plt.close(fig)
        replay_dirs.extend(path.parent for path in run.rglob("*.mj_unroll"))
    data = dict(
        passed=True,
        rows=rows,
        total_training_interactions=sum(x["actual_steps"] for x in rows),
        independent_episodes=sum(x[s]["num_trials"] for x in rows for s in ("dev", "heldout")),
        training_seeds=[0],
        source="cba6e5370773ace8a08107f02810eecabf16c793",
        scope="LOTF hybrid-gradient offline training subsystem; online adaptation excluded",
    )
    (target / "composable-lotf-results.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/tools/verify_phase_replays.py"),
        "--reader",
        str(reader),
        "--output",
        str(target / "composable-lotf-replays.json"),
    ]
    for directory in dict.fromkeys(replay_dirs):
        command += ["--directory", str(directory)]
    subprocess.run(command, check=True)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
