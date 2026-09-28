"""Independent deterministic rollouts with no autoreset or mutable normalization."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
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
        "quality_passed": bool(
            complete.sum() >= math.ceil(0.9 * len(seeds))
            and completed_mean is not None
            and completed_mean <= 0.25
        ),
        "quality_rule": "complete >= ceil(0.9*N) and completed mean position RMSE <=0.25m",
        "episodes": episodes,
    }


def tree_digest(tree) -> str:
    """Hash array content without depending on pickle or device placement."""
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class PolicyEvaluator:
    """Reuse one compiled evaluator for all snapshots of a given policy architecture."""

    def __init__(self, env, make_policy, seeds):
        from drone_playground.runtime.jax_runner import policy_rollout

        self.env, self.seeds = env, seeds
        keys = jnp.asarray([jax.random.PRNGKey(seed) for seed in seeds])
        reference_ids = jnp.arange(len(seeds), dtype=jnp.int32) % env.trajectories.shape[0]
        self._run = jax.jit(
            lambda params: policy_rollout(
                env, make_policy, params, keys, reference_ids=reference_ids, kind="tracking"
            )
        )

    def run(self, params) -> tuple[dict, dict]:
        before = tree_digest(params)
        trace = jax.tree.map(np.asarray, self._run(params))
        after = tree_digest(params)
        if before != after:
            raise RuntimeError("Evaluation changed policy or normalization parameters")
        report = summarize_trials(trace, self.seeds, self.env.dt)
        report.update(
            parameter_sha256=before,
            parameters_frozen=True,
            task=self.env.task,
            dynamics=self.env.dynamics,
            drone=self.env.drone,
            reference_seed=self.env.reference_seed,
            reference_sha256=hashlib.sha256(
                np.asarray(self.env.trajectories).tobytes()
            ).hexdigest(),
        )
        for i, episode in enumerate(report["episodes"]):
            episode["reference_seed"] = self.env.reference_seed + i % self.env.trajectories.shape[0]
        return report, trace


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # NaN results remain visible rather than being silently converted to success.
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def select_replays(trace: dict, report: dict, count: int = 4) -> dict:
    """Fixed cases plus the worst case; preserve all trials in the JSON report."""
    fixed = list(range(min(count, report["num_trials"])))
    worst = max(report["episodes"], key=lambda row: (row["failed"], row["rmse_m"]))["case"]
    indices = sorted(set(fixed + [worst]))
    return jax.tree.map(lambda value: np.asarray(value)[:, indices], trace)
