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
    for mode in ("train", "eval", "play"):
        entry = commands.add_parser(mode, help="使用 method=... env=... 和配置覆盖")
        entry.add_argument("overrides", nargs=argparse.REMAINDER)
    demo = commands.add_parser("demo", help="运行原生控制器完整飞行")
    demo.add_argument("--run-id", required=True)
    demo.add_argument("--duration", type=float, default=10.0)
    demo.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
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
    actual = list(sys.argv[1:] if argv is None else argv)
    if actual and actual[0] in ("train", "eval", "play"):
        from drone_playground.app import script_main

        sys.argv = [sys.argv[0], *actual[1:]]
        return script_main(actual[0])
    args = build_parser().parse_args(argv)
    if hasattr(args, "device"):
        _set_device(args.device)
    if args.command == "demo":
        from drone_playground.execution.controllers.demo import run_demo

        result = run_demo(ROOT, args.run_id, args.duration, args.device)
    elif args.command == "replay":
        from drone_playground.visualization.rscope_io import publish_run

        result = {"active_directory": str(publish_run(args.directory))}
        if args.launch:
            command = [sys.executable, str(ROOT / "scripts/tools/rscope_client.py")]
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
