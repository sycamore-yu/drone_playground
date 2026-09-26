"""Run one explicit recipe version and both independent evaluation splits."""

import argparse
import fcntl
import hashlib
import json
import sys
from datetime import datetime, timezone

from run_p3_matrix import ROOT, atomic_json, run_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warm-start")
    args = parser.parse_args()
    folder = ROOT / "experiments/campaign-recovery"
    folder.mkdir(exist_ok=True)
    lock = (ROOT / "experiments/campaign-p3-seed0/queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    from drone_playground.composition import compose_config
    from drone_playground.runs.legacy import checkpoint_config

    config = compose_config(args.experiment)
    directory = ROOT / "experiments" / args.run_id
    if not (directory / "result.json").exists():
        if directory.exists():
            raise FileExistsError("Recover the incomplete run explicitly; do not overwrite it")
        command = [
            sys.executable,
            "-m",
            "drone_playground.cli",
            "train",
            "--experiment",
            args.experiment,
            "--run-id",
            args.run_id,
            "--device",
            "gpu",
        ]
        if args.warm_start:
            command.extend(["--warm-start", str(ROOT / args.warm_start)])
        code = run_command(command, folder / (args.run_id + ".log"))
        if code:
            raise SystemExit(code)
    result = json.loads((directory / "result.json").read_text())
    if not result.get("full_budget_completed"):
        raise RuntimeError("Existing run did not complete its budget")
    best = json.loads((directory / "checkpoints/best.json").read_text())
    checkpoint = directory / "checkpoints" / best["path"]
    meta = json.loads(checkpoint.with_suffix(".json").read_text())
    saved = checkpoint_config(meta["config"])
    if any(
        saved.get(k) != config.get(k)
        for k in ("dynamics", "task", "network", "algorithm", "objective")
    ):
        raise ValueError("The requested recipe differs from the existing checkpoint config")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    row = dict(
        run_id=args.run_id,
        task=config["task"]["name"],
        algorithm=config["algorithm"]["name"],
        dynamics=config["dynamics"]["forward"],
        actual_steps=result["actual_steps"],
        budget_completed=True,
        checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=before,
        elapsed_seconds=result["elapsed_seconds"],
        recipe=args.experiment,
    )
    for split, count in [("dev", 32), ("heldout", 128)]:
        target = directory / f"independent-{split}"
        report_path = target / "report.json"
        if not report_path.exists():
            if target.exists():
                raise FileExistsError("Incomplete evaluation requires explicit recovery")
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
                folder / f"{args.run_id}-{split}.log",
                timeout=1800,
            )
            if code:
                raise SystemExit(code)
        evidence = json.loads(report_path.read_text())
        if evidence["num_trials"] != count or not evidence["parameters_frozen"]:
            raise RuntimeError("Independent evaluation contract failed")
        row[split] = {
            k: evidence.get(k)
            for k in (
                "completed",
                "num_trials",
                "quality_passed",
                "parameter_sha256",
                "gates_passed_mean",
                "completion_time_mean_s",
                "collision_rate",
            )
        }
        row[split].update(
            rmse_m=evidence["rmse_all_mean"], report=str(report_path.relative_to(ROOT))
        )
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != before:
        raise RuntimeError("Checkpoint changed during evaluation")
    row.update(
        status="completed",
        quality_passed=row["heldout"]["quality_passed"],
        verified_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(folder / (args.run_id + ".json"), row)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
