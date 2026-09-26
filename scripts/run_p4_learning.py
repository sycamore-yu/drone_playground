"""Run all three real racing training budgets, reusing durable completed results."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from run_p3_matrix import ROOT, atomic_json, run_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--config-suffix", default="")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    folder = ROOT / f"experiments/campaign-p4-{args.version}"
    folder.mkdir(exist_ok=True)
    state_path = folder / "state.json"
    atomic_json(
        state_path,
        dict(
            status="waiting-for-p3-gpu",
            pid=os.getpid(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    # Share the existing queue lock; workloads remain serial on the single GPU.
    lock = (ROOT / "experiments/campaign-p3-seed0/queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    rows = []
    summary_path = ROOT / f"docs/verification/p4-learning-{args.version}.json"
    for algorithm in ("ppo", "apg", "shac"):
        if args.only and algorithm != args.only:
            continue
        run_id = f"p4-racing-{algorithm}-first_principles-seed0-{args.version}"
        directory = ROOT / "experiments" / run_id
        row = dict(task="racing", algorithm=algorithm, dynamics="first_principles", run_id=run_id)
        rows.append(row)
        atomic_json(
            state_path,
            dict(
                status="running",
                current=row,
                pid=os.getpid(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        experiment = f"racing_{algorithm}{args.config_suffix}"
        if not (directory / "result.json").exists():
            if directory.exists():
                row.update(status="incomplete-existing-run")
                atomic_json(summary_path, dict(rows=rows, completed=False))
                continue
            code = run_command(
                [
                    sys.executable,
                    "-m",
                    "drone_playground.cli",
                    "train",
                    "--experiment",
                    experiment,
                    "--run-id",
                    run_id,
                    "--device",
                    "gpu",
                ],
                folder / f"{run_id}.log",
            )
            if code != 0:
                row.update(status="execution-error", exit_code=code)
                atomic_json(summary_path, dict(rows=rows, completed=False))
                continue
        result = json.loads((directory / "result.json").read_text())
        if not result.get("full_budget_completed", False):
            row.update(status="execution-error", error=result.get("error", "Incomplete budget"))
            atomic_json(summary_path, dict(rows=rows, completed=False))
            continue
        best = json.loads((directory / "checkpoints/best.json").read_text())
        checkpoint = directory / "checkpoints" / best["path"]
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        row.update(
            actual_steps=result["actual_steps"],
            budget_completed=result["full_budget_completed"],
            elapsed_seconds=result["elapsed_seconds"],
            checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=before,
        )
        okay = True
        for split, count in [("dev", 32), ("heldout", 128)]:
            target = directory / f"independent-{split}"
            report_path = target / "report.json"
            if not report_path.exists():
                if target.exists():
                    row.update(status="incomplete-evaluation", error=str(target))
                    okay = False
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
                    timeout=1800,
                )
                if code != 0:
                    row.update(status="evaluation-error", split=split, exit_code=code)
                    okay = False
                    break
            evidence = json.loads(report_path.read_text())
            if evidence["num_trials"] != count or not evidence["parameters_frozen"]:
                raise RuntimeError(f"Invalid evaluation: {report_path}")
            row[split] = {
                key: evidence.get(key)
                for key in (
                    "completed",
                    "num_trials",
                    "completion_rate",
                    "gates_passed_mean",
                    "collisions",
                    "timeouts",
                    "completion_time_mean_s",
                    "rmse_all_mean",
                    "quality_passed",
                    "parameter_sha256",
                )
            }
            row[split]["report"] = str(report_path.relative_to(ROOT))
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != before:
            raise RuntimeError("Checkpoint mutated during evaluation")
        if okay:
            row.update(status="completed", quality_passed=row["heldout"]["quality_passed"])
        atomic_json(
            summary_path,
            dict(rows=rows, completed=False, updated_at=datetime.now(timezone.utc).isoformat()),
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    done = all(row.get("status") == "completed" for row in rows)
    atomic_json(
        summary_path,
        dict(
            rows=rows,
            completed=done,
            expected_cells=3,
            training_seeds=[0],
            updated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    atomic_json(
        state_path,
        dict(
            status="completed" if done else "needs-attention",
            updated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    raise SystemExit(0 if done else 1)


if __name__ == "__main__":
    main()
