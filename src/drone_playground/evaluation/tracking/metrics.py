"""Tracking metrics and representative trace selection shared by every executor."""

import jax
import numpy as np


def summarize_trials(trace: dict, seeds: list[int], dt: float) -> dict:
    """Aggregate every trial, including failed episodes, exactly once."""
    active = np.asarray(trace["active"], dtype=bool)
    failed = np.any(np.asarray(trace["failed"], dtype=bool), axis=0)
    errors = np.asarray(trace["metrics"]["tracking_error"])
    steps = active.sum(axis=0)
    complete = (~failed) & (steps == active.shape[0])
    squared = np.where(active, errors**2, 0).sum(axis=0)
    rmse = np.sqrt(squared / np.maximum(steps, 1))
    returns = np.where(active, trace["reward"], 0).sum(axis=0)
    episodes = []
    for i, seed in enumerate(seeds):
        episodes.append(
            {
                "case": i,
                "seed": int(seed),
                "completed": bool(complete[i]),
                "failed": bool(failed[i]),
                "steps": int(steps[i]),
                "duration_s": float(steps[i] * dt),
                "rmse_m": float(rmse[i]),
                "return": float(returns[i]),
            }
        )
    completed_mean = float(rmse[complete].mean()) if complete.any() else None
    return {
        "num_trials": len(seeds),
        "completed": int(complete.sum()),
        "failed": int(failed.sum()),
        "completion_rate": float(complete.mean()),
        "rmse_completed_mean": completed_mean,
        "rmse_all_mean": float(rmse.mean()),
        "return_mean": float(returns.mean()),
        "episodes": episodes,
    }


def select_replays(trace: dict, report: dict, count: int = 4) -> dict:
    """Fixed cases plus the worst case; preserve all trials in the JSON report."""
    fixed = list(range(min(count, report["num_trials"])))
    worst = max(report["episodes"], key=lambda row: (row["failed"], row["rmse_m"]))["case"]
    indices = sorted(set([*fixed, worst]))
    return jax.tree.map(lambda value: np.asarray(value)[:, indices], trace)
