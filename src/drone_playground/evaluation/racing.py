"""Independent complete-course racing evaluation, including failed trial rows."""

from __future__ import annotations

import hashlib
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from .tracking import tree_digest


def summarize_race(trace, seeds, dt):
    active = np.asarray(trace["active"], bool)
    metrics = trace["metrics"]
    success = np.any(np.asarray(metrics["success"]) > 0, axis=0)
    collisions = np.any(np.asarray(metrics["collision"]) > 0, axis=0)
    failures = np.any(np.asarray(metrics["failure"]) > 0, axis=0)
    gates = np.max(np.asarray(metrics["gates_passed"]), axis=0).astype(int)
    completed = success & ~failures & (gates >= 5)
    timeout = ~success & ~failures
    rows = []
    for case, seed in enumerate(seeds):
        valid = active[:, case]
        errors = np.asarray(metrics["tracking_error"])[:, case][valid]
        rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0
        steps = int(valid.sum())
        rows.append(
            dict(
                case=case,
                seed=int(seed),
                completed=bool(completed[case]),
                failed=not bool(completed[case]),
                collision=bool(collisions[case]),
                time_out=bool(timeout[case]),
                gates_passed=int(gates[case]),
                steps=steps,
                duration_s=steps * dt,
                completion_time_s=steps * dt if completed[case] else None,
                rmse_m=rmse,
                **{"return": float(np.asarray(trace["reward"])[:, case][valid].sum())},
            )
        )
    n = len(rows)
    completion_times = [x["completion_time_s"] for x in rows if x["completed"]]
    return dict(
        num_trials=n,
        completed=int(completed.sum()),
        failed=int((~completed).sum()),
        completion_rate=float(completed.mean()),
        collisions=int(collisions.sum()),
        collision_rate=float(collisions.mean()),
        timeouts=int(timeout.sum()),
        gates_passed_mean=float(gates.mean()),
        rmse_all_mean=float(np.mean([x["rmse_m"] for x in rows])),
        return_mean=float(np.mean([x["return"] for x in rows])),
        completion_time_mean_s=float(np.mean(completion_times)) if completion_times else None,
        quality_passed=int(completed.sum()) >= math.ceil(0.9 * n),
        episodes=rows,
        completion_rule="LSY Level0 ordered directed gate passages [1,2,3,4,2], no failure",
        course_split="fixed Level0 geometry; independent disturbance seeds",
    )


def merge_race_reports(reports: list[dict], expected_seeds: list[int]) -> dict:
    """Combine independent shards only after exact seed/denominator validation."""
    episodes = []
    for report in reports:
        if report["num_trials"] != len(report["episodes"]):
            raise ValueError("Shard trial count differs from its episode rows")
        episodes.extend(dict(row) for row in report["episodes"])
    observed = [row["seed"] for row in episodes]
    if len(set(observed)) != len(observed) or sorted(observed) != sorted(expected_seeds):
        raise ValueError("Shards must cover every requested seed exactly once")
    episodes.sort(key=lambda row: row["seed"])
    for case, row in enumerate(episodes):
        row["case"] = case
        if row["completed"] and (row["gates_passed"] < 5 or row["failed"]):
            raise ValueError("A completed race must pass all five gates without failure")
    n = len(episodes)
    completed = sum(row["completed"] for row in episodes)
    times = [row["completion_time_s"] for row in episodes if row["completed"]]
    collisions = sum(row["collision"] for row in episodes)
    return dict(
        num_trials=n,
        completed=completed,
        failed=n - completed,
        completion_rate=completed / n,
        collisions=collisions,
        collision_rate=collisions / n,
        timeouts=sum(row["time_out"] for row in episodes),
        gates_passed_mean=float(np.mean([row["gates_passed"] for row in episodes])),
        rmse_all_mean=float(np.mean([row["rmse_m"] for row in episodes])),
        return_mean=float(np.mean([row["return"] for row in episodes])),
        completion_time_mean_s=float(np.mean(times)) if times else None,
        quality_passed=completed >= math.ceil(0.9 * n),
        episodes=episodes,
        actual_steps=sum(report["actual_steps"] for report in reports),
        course_split="fixed native LSY Level0; unique independent disturbance seeds",
        completion_rule="all directed gate passages [1,2,3,4,2], no failure",
    )


class RaceEvaluator:
    def __init__(self, env, make_policy, seeds):
        self.env, self.make_policy, self.seeds = env, make_policy, seeds
        self._run = jax.jit(self._rollout)

    def _rollout(self, params):
        env = self.env
        keys = jax.vmap(jax.random.PRNGKey)(jnp.array(self.seeds, dtype=jnp.uint32))
        states = jax.vmap(env.reset)(keys)
        policy = self.make_policy(params, deterministic=True)

        def one(carry, i):
            states, alive = carry
            action, _ = policy(states.obs, jax.random.PRNGKey(0))
            candidate = jax.vmap(env.step)(states, action)

            def select(a, b):
                mask = alive.reshape(alive.shape + (1,) * (a.ndim - alive.ndim))
                return jnp.where(mask, a, b)

            nxt = jax.tree.map(select, candidate, states)
            x = nxt.pipeline_state.sim_data.states
            row = dict(
                pos=x.pos[:, 0, 0],
                quat=x.quat[:, 0, 0],
                obs=states.obs,
                actions=action,
                reward=jnp.where(alive, nxt.reward, 0.0),
                time=jnp.full(alive.shape, (i + 1) * env.dt),
                metrics=nxt.metrics,
                active=alive,
                failed=nxt.metrics["failure"] > 0,
            )
            return (nxt, alive & ~nxt.done.astype(bool)), row

        _, trace = jax.lax.scan(
            one, (states, jnp.ones(len(self.seeds), bool)), jnp.arange(env.episode_length)
        )
        return trace

    def run(self, params):
        before = tree_digest(params)
        tic = time.monotonic()
        trace = jax.tree.map(np.asarray, self._run(params))
        report = summarize_race(trace, self.seeds, self.env.dt)
        after = tree_digest(params)
        if after != before:
            raise RuntimeError("Evaluation changed frozen parameters")
        report.update(
            parameter_sha256=before,
            parameters_frozen=True,
            task="racing",
            dynamics=self.env.dynamics,
            drone=self.env.drone,
            elapsed_seconds=time.monotonic() - tic,
            reference_seed=self.env.reference_seed,
            reference_sha256=hashlib.sha256(
                np.asarray(self.env.trajectories).tobytes()
            ).hexdigest(),
        )
        return report, trace
