"""Resume the declared P3 matrix from immutable run results and real processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS = ["so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag", "first_principles"]
REUSE = {
    ("figure8", "ppo"): "p2-figure8-ppo-seed0-v2",
    ("figure8", "apg"): "p2-figure8-apg-seed0-v1",
    ("random", "ppo"): "p2-random-ppo-seed0-v1",
    ("random", "apg"): "p2-random-apg-seed0-v1",
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def run_command(args, log, timeout=3900):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        SCIPY_ARRAY_API="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    with log.open("w") as stream:
        child = subprocess.Popen(
            args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        print(
            json.dumps({"action": "start", "pid": child.pid, "command": args, "log": str(log)}),
            flush=True,
        )
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal

            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 124


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="Substring of run id, optional")
    args = parser.parse_args()
    folder = ROOT / "experiments/campaign-p3-seed0"
    folder.mkdir(exist_ok=True)
    lock = (folder / "queue.lock").open("w")
    import fcntl

    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = []
    state_path = folder / "state.json"
    summary_path = ROOT / "docs/verification/p3-matrix.json"
    for dynamics in DYNAMICS:
        for task in ("figure8", "random"):
            for algorithm in ("ppo", "apg", "shac"):
                reused = dynamics == "so_rpy" and algorithm in ("ppo", "apg")
                run_id = (
                    REUSE[(task, algorithm)]
                    if reused
                    else f"p3-{task}-{algorithm}-{dynamics}-seed0-v1"
                )
                if args.only and args.only not in run_id:
                    continue
                row = dict(
                    task=task, algorithm=algorithm, dynamics=dynamics, run_id=run_id, reused=reused
                )
                rows.append(row)
                directory = ROOT / "experiments" / run_id
                atomic_json(
                    state_path,
                    dict(
                        status="running",
                        current=row,
                        pid=os.getpid(),
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    ),
                )
                config = json.loads(
                    (ROOT / f"configs/experiments/{task}_{algorithm}.json").read_text()
                )
                config.update(dynamics=dynamics, publish_live=False)
                config_path = folder / f"{run_id}.json"
                atomic_json(config_path, config)
                if not (directory / "result.json").exists():
                    if directory.exists():
                        row.update(
                            status="incomplete-existing-run",
                            error="Explicit recovery or a new version is required",
                        )
                        atomic_json(summary_path, dict(rows=rows, completed=False))
                        continue
                    code = run_command(
                        [
                            sys.executable,
                            "-m",
                            "drone_playground.cli",
                            "train",
                            "--config",
                            str(config_path),
                            "--run-id",
                            run_id,
                            "--device",
                            "gpu",
                        ],
                        folder / f"{run_id}.log",
                    )
                    if code != 0:
                        row.update(
                            status="execution-error",
                            exit_code=code,
                            log=str(folder / f"{run_id}.log"),
                        )
                        atomic_json(summary_path, dict(rows=rows, completed=False))
                        continue
                result = json.loads((directory / "result.json").read_text())
                if not result.get("full_budget_completed", False):
                    row.update(
                        status="execution-error", error=result.get("error", "Incomplete budget")
                    )
                    atomic_json(summary_path, dict(rows=rows, completed=False))
                    continue
                best = json.loads((directory / "checkpoints/best.json").read_text())
                checkpoint = directory / "checkpoints" / best["path"]
                checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                row.update(
                    actual_steps=result["actual_steps"],
                    budget_completed=result["full_budget_completed"],
                    elapsed_seconds=result["elapsed_seconds"],
                    checkpoint=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=checkpoint_hash,
                )
                ok = True
                for split, count in [("dev", 32), ("heldout", 128)]:
                    target = directory / f"independent-{split}"
                    report = target / "report.json"
                    if not report.exists():
                        if target.exists():
                            row.update(status="incomplete-evaluation", error=str(target))
                            ok = False
                            break
                        code = run_command(
                            [
                                sys.executable,
                                "-m",
                                "drone_playground.cli",
                                "evaluate",
                                "--checkpoint",
                                str(checkpoint),
                                "--split",
                                split,
                                "--episodes",
                                str(count),
                                "--output",
                                str(target),
                                "--device",
                                "cpu",
                            ],
                            folder / f"{run_id}-{split}.log",
                            timeout=1200,
                        )
                        if code != 0:
                            row.update(status="evaluation-error", split=split, exit_code=code)
                            ok = False
                            break
                    evidence = json.loads(report.read_text())
                    if evidence["num_trials"] != count or not evidence["parameters_frozen"]:
                        raise RuntimeError(f"Invalid independent evaluation: {report}")
                    row[split] = dict(
                        completed=evidence["completed"],
                        num_trials=count,
                        rmse_m=evidence["rmse_all_mean"],
                        quality_passed=evidence["quality_passed"],
                        parameter_sha256=evidence["parameter_sha256"],
                        report=str(report.relative_to(ROOT)),
                    )
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
                    raise RuntimeError("Checkpoint changed during evaluation")
                if ok:
                    row.update(status="completed", quality_passed=row["heldout"]["quality_passed"])
                print(json.dumps(row, ensure_ascii=False), flush=True)
                atomic_json(
                    summary_path,
                    dict(
                        rows=rows,
                        completed=False,
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    ),
                )
    done = all(row.get("status") == "completed" for row in rows)
    atomic_json(
        summary_path,
        dict(
            rows=rows,
            completed=done,
            updated_at=datetime.now(timezone.utc).isoformat(),
            training_seeds=[0],
            expected_cells=24,
        ),
    )
    atomic_json(
        state_path,
        dict(
            status="completed" if done else "needs-attention",
            rows=len(rows),
            completed=sum(row.get("status") == "completed" for row in rows),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    raise SystemExit(0 if done else 1)


if __name__ == "__main__":
    main()
