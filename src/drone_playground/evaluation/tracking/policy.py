"""Independent deterministic rollouts with no autoreset or mutable normalization."""

from __future__ import annotations

from drone_playground.evaluation.tracking.metrics import select_replays
from drone_playground.evaluation.tracking.metrics import summarize_trials

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.artifacts.reporting import tree_digest




class PolicyEvaluator:
    """Reuse one compiled evaluator for all snapshots of a given policy architecture."""

    def __init__(self, env, make_policy, seeds):
        from drone_playground.runtime.jax_runner import policy_rollout

        self.env, self.make_policy, self.seeds = env, make_policy, seeds
        keys = jnp.asarray([jax.random.PRNGKey(seed) for seed in seeds])
        reference_ids = (
            jnp.arange(len(seeds), dtype=jnp.int32) % env.trajectories.shape[0]
        )
        self._run = jax.jit(
            lambda params: policy_rollout(
                env,
                make_policy,
                params,
                keys,
                reference_ids=reference_ids,
                kind="tracking",
            )
        )

    def run(self, params) -> tuple[dict, dict]:
        before = tree_digest(params)
        trace = jax.tree.map(np.asarray, self._run(params))
        after = tree_digest(params)
        if before != after:
            raise RuntimeError(
                "Evaluation changed policy or normalization parameters"
            )
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
            episode["reference_seed"] = (
                self.env.reference_seed + i % self.env.trajectories.shape[0]
            )
        from drone_playground.runtime.timing import measure_policy

        initial = self.env.reset(jax.random.PRNGKey(self.seeds[0]))
        report.update(
            measure_policy(
                self.env, self.make_policy, params, initial.obs, initial
            )
        )
        report["sensor_timing"] = getattr(self.env, "sensor_timing", None)
        return report, trace
