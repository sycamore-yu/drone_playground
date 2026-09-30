"""Validate immutable run evidence before making a phase-delivery summary.

This module reads JSON and content digests only. It never loads policy pickles,
runs an environment, changes a checkpoint, or turns a failed run into a score.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from drone_playground.runs.layout import experiment_directory, resolve_artifact


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(path: Path, expected_count: int, expected_start: int) -> dict:
    report = json.loads(path.read_text())
    rows = report["episodes"]
    if len(rows) != expected_count or report["num_trials"] != expected_count:
        raise ValueError(f"Wrong trial denominator: {path}")
    seeds = [row["seed"] for row in rows]
    if sorted(seeds) != list(range(expected_start, expected_start + expected_count)):
        raise ValueError(f"Wrong or duplicated evaluation seed: {path}")
    completed = sum(bool(row["completed"]) for row in rows)
    if completed != report["completed"]:
        raise ValueError(f"Claimed completed count differs from individual rows: {path}")
    if not math.isclose(report["completion_rate"], completed / expected_count, abs_tol=1e-12):
        raise ValueError(f"Completion rate differs from complete denominator: {path}")
    mean = sum(row["rmse_m"] for row in rows) / expected_count
    if not math.isfinite(mean) or not math.isclose(
        mean, report["rmse_all_mean"], rel_tol=1e-7, abs_tol=1e-8
    ):
        raise ValueError(f"RMSE summary differs from individual rows: {path}")
    if "racing" == report.get("task") or "controller" in report:
        if any(
            row["completed"] and (row["failed"] or row["collision"] or row["gates_passed"] < 5)
            for row in rows
        ):
            raise ValueError(f"Claimed race success has incomplete gate events: {path}")
    return report


def learning_result(root: Path, run_id: str) -> dict:
    """Return validated public evidence, retaining incomplete/error/low-quality states."""
    root = Path(root).resolve()
    run = experiment_directory(root, run_id)
    row = dict(run_id=run_id, experiment_completed=False, quality_passed=False)
    result_path = run / "result.json"
    if not result_path.exists():
        return {**row, "status": "running-or-not-started"}
    result = json.loads(result_path.read_text())
    if not result.get("full_budget_completed") or result.get("status") != "completed":
        return {
            **row,
            "status": "execution-error",
            "error": result.get("error"),
            "result": str(result_path.relative_to(root)),
            "result_sha256": digest(result_path),
        }
    selected = json.loads((run / "checkpoints/best.json").read_text())
    if selected["selection_split"] != "dev":
        raise ValueError(f"Checkpoint was not selected on development data: {run}")
    checkpoint = run / "checkpoints" / selected["path"]
    meta = json.loads(checkpoint.with_suffix(".json").read_text())
    if digest(checkpoint) != meta["sha256"]:
        raise ValueError(f"Checkpoint digest mismatch: {checkpoint}")
    config = meta["config"]
    budget = (
        config["num_timesteps"]
        if config["algorithm"] == "ppo"
        else config["policy_updates"] * config["num_envs"] * config["horizon_length"]
    )
    if result["actual_steps"] != budget:
        raise ValueError(f"Actual interaction budget differs from frozen recipe: {run}")
    row.update({name: config[name] for name in ("task", "algorithm", "dynamics", "drone", "seed")})
    row.update(
        actual_steps=result["actual_steps"],
        budget_completed=True,
        elapsed_seconds=result["elapsed_seconds"],
        selected_step=meta["step"],
        initial_policy_selected=meta["step"] == 0,
        checkpoint=str(checkpoint.relative_to(root)),
        checkpoint_sha256=meta["sha256"],
        parameter_sha256=meta["parameter_sha256"],
        result_sha256=digest(result_path),
        actual_devices=config.get("actual_devices", []),
        recipe_source=config.get("source", ""),
    )
    for split, count, start in (("dev", 32, 20000), ("heldout", 128, 30000)):
        path = run / f"independent-{split}/report.json"
        if not path.exists():
            return {**row, "status": "awaiting-evaluation", "missing_split": split}
        report = _report(path, count, start)
        for key in ("task", "dynamics", "drone"):
            if report[key] != config[key]:
                raise ValueError(f"Evaluation {key} differs from checkpoint: {path}")
        if (
            not report["parameters_frozen"]
            or report["parameter_sha256"] != meta["parameter_sha256"]
        ):
            raise ValueError(f"Evaluated policy digest differs from selected checkpoint: {path}")
        if "checkpoint" in report and resolve_artifact(report["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError(f"Evaluation references a different checkpoint: {path}")
        row[split] = {
            key: report[key]
            for key in (
                "num_trials",
                "completed",
                "completion_rate",
                "rmse_all_mean",
                "quality_passed",
            )
        }
        row[split].update(
            {
                key: report[key]
                for key in (
                    "collisions",
                    "timeouts",
                    "gates_passed_mean",
                    "completion_time_mean_s",
                    "reference_sha256",
                    "process_id",
                )
                if key in report
            }
        )
        row[split].update(report=str(path.relative_to(root)), report_sha256=digest(path))
    row.update(
        status="completed",
        experiment_completed=True,
        quality_passed=row["heldout"]["quality_passed"],
    )
    return row


def controller_result(root: Path, run_id: str) -> dict:
    """Verify an aggregated controller result against all four original shard reports."""
    root = Path(root).resolve()
    run = experiment_directory(root, run_id)
    result_path = run / "result.json"
    if not result_path.exists():
        return dict(run_id=run_id, status="running-or-not-started", experiment_completed=False)
    result = json.loads(result_path.read_text())
    if result.get("status") != "completed" or not result.get("full_budget_completed"):
        return dict(
            run_id=run_id,
            status="execution-error",
            experiment_completed=False,
            error=result.get("error"),
        )
    path = run / "eval/report.json"
    report = _report(path, 128, 30000)
    source_rows = []
    for source in report["shards"]:
        original = experiment_directory(root, source['run_id']) / 'eval/report.json'
        if digest(original) != source["report_sha256"]:
            raise ValueError(f"Original controller shard digest mismatch: {original}")
        fragment = _report(original, source["episodes"], source["seed_start"])
        if fragment["config"]["native_disturbances"] != report["native_disturbances"]:
            raise ValueError(f"Different native disturbances across controller trials: {original}")
        source_rows.extend(fragment["episodes"])
    source_rows.sort(key=lambda row: row["seed"])
    for expected, actual in zip(source_rows, report["episodes"], strict=True):
        for key in (
            "seed",
            "completed",
            "failed",
            "collision",
            "gates_passed",
            "rmse_m",
            "completion_time_s",
        ):
            if expected[key] != actual[key]:
                raise ValueError(f"Merged {key} differs from its original trial: {path}")
    row = {
        key: report[key]
        for key in (
            "controller",
            "completed",
            "num_trials",
            "completion_rate",
            "collisions",
            "timeouts",
            "completion_time_mean_s",
            "gates_passed_mean",
            "rmse_all_mean",
            "decision_p50_ms",
            "decision_p95_ms",
            "deadline_miss_fraction",
            "nonzero_solve_status_count",
            "source_run_sum_seconds",
            "native_disturbances",
            "actual_environment_devices",
            "timing_protocol",
        )
    }
    row.update(
        run_id=run_id,
        status="completed",
        experiment_completed=True,
        quality_passed=report["quality_passed"],
        report=str(path.relative_to(root)),
        report_sha256=digest(path),
        shards=report["shards"],
    )
    return row
