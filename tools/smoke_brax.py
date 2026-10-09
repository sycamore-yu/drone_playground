"""Execute native Brax PPO/APG smoke tests without a second project Trainer.

Example:
    SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu PYTHONPATH=src python tools/smoke_brax.py

Install the optional Brax extra in an isolated environment. This checks the
training API, not convergence or method quality.
"""

import argparse
import json
import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import jax
import numpy as np
from brax.training.agents.apg import train as native_apg
from brax.training.agents.ppo import train as native_ppo

from drone_playground.learning.brax_adapter import BraxTrainingEnv
from drone_playground.simulation.environment import make_environment


def main():
    """Train once with the selected unmodified Brax algorithm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "apg"), default="ppo")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_envs < 2 or args.num_envs % 2:
        raise ValueError("Use at least two worlds and a multiple of two for this smoke recipe")
    original = make_environment(
        num_envs=args.num_envs,
        device="cpu",
        task="tracking",
        duration=0.08,
        physics_hz=100,
        method_hz=50,
    )
    try:
        environment = BraxTrainingEnv(original)
        checkpoints = []
        if args.algorithm == "ppo":

            def capture(step, maker, params):
                """Capture only policy network parameters for update verification."""
                checkpoints.append(jax.tree.map(lambda value: np.asarray(value).copy(), params[1]))

            _, params, metrics = native_ppo.train(
                environment=environment,
                num_timesteps=args.timesteps,
                num_envs=args.num_envs,
                episode_length=4,
                unroll_length=2,
                batch_size=args.num_envs,
                num_minibatches=2,
                num_updates_per_batch=1,
                num_evals=1,
                num_eval_envs=args.num_envs,
                run_evals=False,
                normalize_observations=False,
                wrap_env=False,
                use_pmap_on_reset=True,
                seed=args.seed,
                policy_params_fn=capture,
            )
            if len(checkpoints) < 2:
                raise AssertionError("Brax PPO did not expose both initial and trained policies")
            change = sum(
                np.sum((new - old) ** 2)
                for old, new in zip(
                    jax.tree.leaves(checkpoints[0]),
                    jax.tree.leaves(checkpoints[-1]),
                    strict=True,
                )
            )
            if not np.isfinite(change) or change <= 0:
                raise AssertionError("Native PPO did not update Actor parameters")
        else:
            _, params, metrics = native_apg.train(
                environment=environment,
                policy_updates=2,
                num_envs=args.num_envs,
                horizon_length=2,
                episode_length=4,
                num_evals=2,
                num_eval_envs=args.num_envs,
                wrap_env=False,
                seed=args.seed,
            )
        scalar_metrics = {
            name: float(value)
            for name, value in jax.device_get(metrics).items()
            if np.asarray(value).size == 1
        }
        if not scalar_metrics or not all(np.isfinite(list(scalar_metrics.values()))):
            raise AssertionError("Native Brax trainer did not produce finite training metrics")
        if not jax.tree.leaves(params):
            raise AssertionError("Native Brax trainer did not return network parameters")
        print(
            json.dumps(
                {
                    "result": "passed",
                    "runner": f"brax.training.agents.{args.algorithm}.train",
                    "backend": environment.backend,
                    "num_worlds": args.num_envs,
                    "expected_interactions": (
                        args.timesteps if args.algorithm == "ppo" else 2 * args.num_envs * 2
                    ),
                    "seed": args.seed,
                    "metrics": scalar_metrics,
                    "quality_claim": False,
                },
                sort_keys=True,
            )
        )
    finally:
        original.sim.close()


if __name__ == "__main__":
    main()
