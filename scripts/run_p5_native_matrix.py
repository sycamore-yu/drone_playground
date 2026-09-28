#!/usr/bin/env python3
"""Evaluate four frozen native planner units with isolated ROS masters."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_REVISION = "navigation8-v1"
HISTORICAL_REVISIONS = {"v1", "v2"}
UNITS = [(task, method) for task in ("static", "dynamic") for method in ("ego", "super")]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=CURRENT_REVISION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.revision in HISTORICAL_REVISIONS:
        raise SystemExit(
            f"revision {args.revision} is a preserved historical protocol; "
            f"new runs use {CURRENT_REVISION} with Navigation8"
        )
    logs = ROOT / "experiments" / ("p5-native-matrix-" + args.revision)
    logs.mkdir(parents=True, exist_ok=True)

    def run_unit(index, task, method):
        row = dict(task=task, method=method, status="pending")
        try:
            for split, count in (("dev", 32), ("heldout", 128)):
                run_id = f"p5-formal-{task}-{method}-{args.revision}-{split}"
                path = ROOT / "experiments" / run_id
                command = [sys.executable, "-m", "drone_playground.app", "mode=evaluate",
                           f"experiment=p5_{task}_{method}", f"evaluation.split={split}",
                           f"evaluation.episodes={count}", "policy.workers=4",
                           f"policy.port={12400 + index * 20}", f"run_id={run_id}"]
                if args.dry_run:
                    row[split] = command
                    continue
                result = path / "result.json"
                done = result.exists() and json.loads(result.read_text()).get("status") == "completed"
                if not done:
                    if path.exists():
                        raise RuntimeError(f"Refusing to overwrite incomplete {run_id}")
                    env = dict(os.environ)
                    env.pop("PYTHONPATH", None)
                    env["JAX_PLATFORMS"] = "cpu"
                    with (logs / (run_id + ".log")).open("w") as handle:
                        process = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                                 stderr=subprocess.STDOUT)
                    if process.returncode:
                        raise RuntimeError(f"{run_id} exited {process.returncode}")
                report = json.loads((path / "eval" / "report.json").read_text())
                if report["num_trials"] != 3 * count or report["native_trajectories"] == 0:
                    raise RuntimeError(f"{run_id} failed count/native-solve verification")
                row[split] = dict(run_id=run_id, num_trials=report["num_trials"],
                                  success_rate=report["success_rate"])
            row["status"] = "planned" if args.dry_run else "completed"
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        return row

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_unit, i, *unit) for i, unit in enumerate(UNITS)]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            temporary = logs / "queue-state.tmp"
            temporary.write_text(json.dumps(results, indent=2) + "\n")
            temporary.replace(logs / "queue-state.json")
            print(json.dumps(row), flush=True)
    if any(row["status"] == "failed" for row in results):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
