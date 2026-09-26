"""Small public commands for execution, independent evaluation and observation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crazyflow 无人机训练、评测与记录")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="调用原生 Brax 完整训练")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    train.add_argument("--warm-start", type=Path)
    train.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    demo = commands.add_parser("demo", help="运行原生控制器完整飞行")
    demo.add_argument("--run-id", required=True)
    demo.add_argument("--duration", type=float, default=10.0)
    demo.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    evaluate = commands.add_parser("evaluate", help="独立进程重载与评测冻结策略")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=["dev", "heldout"], default="dev")
    evaluate.add_argument("--episodes", type=int, default=32)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    replay = commands.add_parser("replay", help="发布指定轨迹供原版 rscope 远程读取")
    replay.add_argument("--directory", type=Path, required=True)
    replay.add_argument("--launch", action="store_true", help="在当前桌面打开原版查看器")
    replay.add_argument("--show-metrics", action="store_true")
    metrics = commands.add_parser("metrics", help="在回环地址启动 TensorBoard")
    metrics.add_argument("--port", type=int, default=6006)
    metrics.add_argument("--logdir", type=Path, default=ROOT / "experiments")
    status = commands.add_parser("status", help="报告真实运行状态与最近更新")
    status.add_argument("--run-id")
    return parser


def _set_device(device: str) -> None:
    # Set before any JAX/Crazyflow import.
    os.environ["JAX_PLATFORMS"] = "cpu" if device == "cpu" else "cuda"
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if hasattr(args, "device"):
        _set_device(args.device)
    if args.command == "train":
        from drone_playground.learning.train import train

        config = json.loads(args.config.read_text())
        for setting in args.set:
            key, value = setting.split("=", 1)
            if key not in config:
                raise ValueError(f"Unknown configuration key: {key}")
            config[key] = json.loads(value)
        result = train(config, ROOT, args.run_id, args.device, args.warm_start)
    elif args.command == "demo":
        from drone_playground.controllers.demo import run_demo

        result = run_demo(ROOT, args.run_id, args.duration, args.device)
    elif args.command == "evaluate":
        from drone_playground.evaluation.tracking import (
            save_report,
            select_replays,
        )
        from drone_playground.learning.train import load_policy, make_evaluator, make_task
        from drone_playground.runs.rscope_io import export_rollout

        if args.output.exists():
            raise FileExistsError(f"Use a new evaluation directory: {args.output}")
        args.output.mkdir(parents=True)
        maker, params, meta = load_policy(args.checkpoint)
        env = make_task(meta["config"], args.device, args.split, args.episodes)
        try:
            offset = 20000 if args.split == "dev" else 30000
            evaluator = make_evaluator(env, maker, list(range(offset, offset + args.episodes)))
            result, trace = evaluator.run(params)
            result.update(
                split=args.split,
                checkpoint=str(args.checkpoint.resolve()),
                command=sys.argv,
                process_id=os.getpid(),
            )
            save_report(args.output / "report.json", result)
            export_rollout(env.sim, args.output / "rollouts", select_replays(trace, result))
        finally:
            env.close()
    elif args.command == "replay":
        from drone_playground.runs.rscope_io import publish_run

        result = {"active_directory": str(publish_run(args.directory))}
        if args.launch:
            command = [sys.executable, str(ROOT / "scripts/rscope_client.py")]
            if args.show_metrics:
                command.append("--show-metrics")
            subprocess.run(command, check=True)
    elif args.command == "metrics":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(args.logdir),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            check=True,
        )
        return
    else:
        paths = (
            [ROOT / "experiments" / args.run_id]
            if args.run_id
            else sorted((ROOT / "experiments").glob("*"))
        )
        result = []
        for path in paths:
            state = path / "state.json"
            if not state.exists():
                continue
            row = json.loads(state.read_text())
            process = row.get("process", {})
            pid = process.get("pid")
            actual_marker = None
            try:
                ticks = Path(f"/proc/{pid}/stat").read_text().split()[21]
                boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                actual_marker = f"{boot}:{pid}:{ticks}"
            except (OSError, IndexError):
                pass
            row["process_identity_matches"] = (
                actual_marker is not None and actual_marker == process.get("start_marker")
            )
            updated = datetime.fromisoformat(row["updated_at"])
            row["seconds_since_update"] = (datetime.now(timezone.utc) - updated).total_seconds()
            row["needs_attention"] = row.get("status") == "running" and (
                not row["process_identity_matches"] or row["seconds_since_update"] > 180
            )
            row["run_id"] = path.name
            row["checked_at"] = datetime.now(timezone.utc).isoformat()
            result.append(row)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
