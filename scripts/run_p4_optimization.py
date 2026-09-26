"""Evaluate two real controllers in four independent 32-trial shards each."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
from run_p3_matrix import ROOT, atomic_json, run_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use at most four independent control processes")
    folder = ROOT / f"experiments/campaign-p4-optimization-{args.version}"
    folder.mkdir(exist_ok=True)
    started = time.monotonic()
    jobs = []
    for controller in ("attitude_mpc", "sampling_mpc"):
        label = controller.replace("_", "-")
        for shard in range(4):
            run_id = f"p4-racing-{label}-heldout-{args.version}-shard{shard}"
            jobs.append(
                dict(
                    controller=controller, shard=shard, run_id=run_id, seed_start=30000 + 32 * shard
                )
            )
    state = dict(
        status="running",
        pid=os.getpid(),
        jobs=jobs,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(folder / "state.json", state)

    def execute(job):
        run = ROOT / "experiments" / job["run_id"]
        result_path = run / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            return dict(
                job, status="completed" if result.get("full_budget_completed") else "previous-error"
            )
        if run.exists():
            return dict(job, status="incomplete-existing-run")
        cuda = job["controller"] == "sampling_mpc"
        command = [
            "env",
            "JAX_PLATFORMS=cuda,cpu" if cuda else "JAX_PLATFORMS=cpu",
            sys.executable,
            str(ROOT / "scripts/evaluate_racing_control.py"),
            "--controller",
            job["controller"],
            "--episodes",
            "32",
            "--split",
            "heldout",
            "--seed-start",
            str(job["seed_start"]),
            "--run-id",
            job["run_id"],
            "--samples",
            "2000",
            "--prediction-device",
            "gpu" if cuda else "cpu",
        ]
        code = run_command(command, folder / (job["run_id"] + ".log"), timeout=3600)
        return dict(job, status="completed" if code == 0 else "execution-error", exit_code=code)

    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            outcomes.append(future.result())
            state.update(outcomes=outcomes, updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(folder / "state.json", state)
    if any(row["status"] != "completed" for row in outcomes):
        state.update(status="needs-attention")
        atomic_json(folder / "state.json", state)
        raise SystemExit(1)

    from drone_playground.evaluation.racing import merge_race_reports
    from drone_playground.runs.record import RunRecorder

    summary = []
    for controller in ("attitude_mpc", "sampling_mpc"):
        selected = sorted(
            (job for job in jobs if job["controller"] == controller), key=lambda row: row["shard"]
        )
        reports = []
        timing = []
        sources = []
        for job in selected:
            run = ROOT / "experiments" / job["run_id"]
            report_path = run / "eval/report.json"
            reports.append(json.loads(report_path.read_text()))
            steps = json.loads((run / "eval/solver-steps.json").read_text())["steps"]
            timing.extend([step["decision_seconds"] for step in steps[1:]])
            sources.append(
                dict(
                    run_id=job["run_id"],
                    seed_start=job["seed_start"],
                    episodes=32,
                    report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                )
            )
        if (
            len(
                {
                    json.dumps(report["config"]["native_disturbances"], sort_keys=True)
                    for report in reports
                }
            )
            != 1
        ):
            raise ValueError("Controller shards used different disturbance protocols")
        combined = merge_race_reports(reports, list(range(30000, 30128)))
        combined.update(
            controller=controller,
            configuration_frozen=True,
            shards=sources,
            actual_environment_devices=reports[0]["actual_environment_devices"],
            native_disturbances=reports[0]["config"]["native_disturbances"],
            nonzero_solve_status_count=sum(
                report["nonzero_solve_status_count"] for report in reports
            ),
            decision_p50_ms=float(np.median(timing) * 1000),
            decision_p95_ms=float(np.quantile(timing, 0.95) * 1000),
            deadline_miss_fraction=float(np.mean(np.asarray(timing) > 0.02)),
            timing_protocol="four independent processes on shared server; per-step timings exclude first compile; delays not injected",
            source_run_sum_seconds=sum(report["elapsed_seconds"] for report in reports),
        )
        run_id = f"p4-racing-{controller.replace('_', '-')}-heldout-{args.version}"
        rec = RunRecorder(
            ROOT,
            run_id,
            dict(
                kind="verified-shard-aggregation",
                controller=controller,
                source_runs=sources,
                expected_seeds=list(range(30000, 30128)),
            ),
            task_id="06",
        )
        try:
            rec.phase("aggregating")
            atomic_json(rec.path / "eval/report.json", combined)
            for job in selected:
                source = ROOT / "experiments" / job["run_id"] / "rollouts"
                destination = rec.path / "rollouts" / f"shard-{job['shard']}"
                shutil.copytree(source, destination)
                for file in source.rglob("*"):
                    if (
                        file.is_file()
                        and file.read_bytes()
                        != (destination / file.relative_to(source)).read_bytes()
                    ):
                        raise RuntimeError("Copied replay differs from its immutable shard source")
            for row in combined["episodes"]:
                rec.log(
                    row["case"] + 1,
                    {
                        "eval/completed": row["completed"],
                        "eval/gates_passed": row["gates_passed"],
                        "eval/collision": row["collision"],
                        "eval/completion_time_s": row["completion_time_s"] or 0,
                    },
                )
            rec.finish(
                "completed",
                full_budget_completed=True,
                engineer_passed=True,
                quality_passed=combined["quality_passed"],
                completed=combined["completed"],
                num_trials=128,
                source="128 distinct trials from four unchanged shard reports",
            )
        except BaseException as error:
            rec.finish("failed", error=repr(error))
            raise
        summary.append(
            {k: v for k, v in combined.items() if k != "episodes"}
            | {"run_id": run_id, "status": "completed"}
        )
    atomic_json(
        ROOT / f"docs/verification/p4-optimization-{args.version}.json",
        dict(rows=summary, completed=True),
    )
    state.update(
        status="completed",
        elapsed_seconds=time.monotonic() - started,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(folder / "state.json", state)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
