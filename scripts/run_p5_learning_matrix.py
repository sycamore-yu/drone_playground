#!/usr/bin/env python3
"""Execute the frozen eight training units and their independent evaluations.

Completed runs are verified and reused. Failed/incomplete run IDs are never
overwritten or silently warm-started. Logs and queue state survive interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 8_388_608
UNITS = [(task, sensor, algorithm) for task in ("static", "dynamic")
         for sensor in ("depth", "lidar") for algorithm in ("ppo", "dva")]

def read(path):
    return json.loads(path.read_text())

def completed(path, steps=None):
    result = path / "result.json"
    if not result.exists():
        return False
    data = read(result)
    return (data.get("status") == "completed" and data.get("full_budget_completed") is True
            and (steps is None or data.get("actual_steps") == steps))

def run_command(command, run_id, logs):
    directory = ROOT / "experiments" / run_id
    if directory.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete run {run_id}; inspect its evidence")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["JAX_COMPILATION_CACHE_DIR"] = str(ROOT / "tmp" / "p5-jax-cache")
    with (logs / (run_id + ".log")).open("w") as handle:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"{run_id} exited {proc.returncode}; see its preserved log")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logs = ROOT / "experiments" / ("p5-matrix-" + args.revision)
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for task, sensor, algorithm in UNITS:
        experiment = f"p5_{task}_{sensor}_{algorithm}"
        run_id = f"p5-formal-{task}-{sensor}-{algorithm}-seed0-{args.revision}"
        row = dict(task=task, sensor=sensor, algorithm=algorithm, run_id=run_id,
                   requested_steps=BUDGET, status="pending")
        rows.append(row)
        command = [sys.executable, "-m", "drone_playground.app", "mode=train",
                   f"experiment={experiment}", f"run_id={run_id}"]
        if args.dry_run:
            row["command"] = command
            continue
        try:
            path = ROOT / "experiments" / run_id
            if not completed(path, BUDGET):
                run_command(command, run_id, logs)
            if not completed(path, BUDGET):
                raise RuntimeError(f"Budget verification failed for {run_id}")
            best = read(path / "checkpoints" / "best.json")
            checkpoint = path / "checkpoints" / best["path"]
            if best.get("selection_split") != "dev":
                raise RuntimeError("Best checkpoint was not selected exclusively on development data")
            row.update(status="trained", best_checkpoint=str(checkpoint), best_step=best["step"])
            for split, episodes in (("dev", 32), ("heldout", 128)):
                evaluation_id = run_id + "-" + split
                if not completed(ROOT / "experiments" / evaluation_id):
                    run_command(
                        [sys.executable, "-m", "drone_playground.app", "mode=evaluate",
                         f"experiment={experiment}", f"checkpoint={checkpoint}",
                         f"evaluation.split={split}", f"evaluation.episodes={episodes}",
                         f"run_id={evaluation_id}"], evaluation_id, logs)
                report = read(ROOT / "experiments" / evaluation_id / "eval" / "report.json")
                if report["num_trials"] != episodes * 3 or not report["parameters_frozen"]:
                    raise RuntimeError("Independent evaluation count/freeze verification failed")
                row[split] = dict(run_id=evaluation_id, num_trials=report["num_trials"],
                                  success_rate=report["success_rate"])
            row["status"] = "completed"
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            failures.append(str(exc))
        finally:
            temporary = logs / "queue-state.tmp"
            temporary.write_text(json.dumps(rows, indent=2) + "\n")
            temporary.replace(logs / "queue-state.json")
            print(json.dumps(row), flush=True)
    if args.dry_run:
        print(json.dumps(rows, indent=2))
    if failures:
        raise SystemExit("\n".join(failures))

if __name__ == "__main__":
    main()

