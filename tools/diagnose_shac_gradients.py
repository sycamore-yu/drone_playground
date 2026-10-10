"""Compare Racing SHAC physical and terminal Critic gradients on one training state.

This is a one-time causal diagnostic, not a frozen-policy benchmark. It reads
only the archived same-run training checkpoint. Gradients are measured on a
fresh, common training-diagnostic batch with SHAC's original rollout horizon,
holding actor, target Critic, initial states, and RNG fixed.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from drone_playground.learning.apg import rollout_objective
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.policy import read_checkpoint
from drone_playground.simulation.records import atomic_json


def norm(tree):
    """Return a global L2 norm without materializing parameter gradients on host."""
    return jnp.sqrt(sum(jnp.vdot(g, g) for g in jax.tree.leaves(tree)))


def compare(checkpoint: Path, *, worlds=16, horizon=32) -> dict:
    """Isolate the terminal-value term without replacing historical parameters."""
    config_file = checkpoint.parents[1] / "config.yaml"
    config = OmegaConf.load(config_file)
    if config.task.name != "racing" or config.learning.algorithm != "shac":
        raise ValueError("Expected a Racing/SHAC native training checkpoint")
    if worlds <= 0 or horizon <= 0:
        raise ValueError("World count and gradient horizon must be positive")
    payload, checkpoint_metadata = read_checkpoint(checkpoint, "training")
    archived = serialization.msgpack_restore(payload)
    if checkpoint_metadata["config"]["experiment"]["task"]["name"] != "racing":
        raise ValueError("Archived checkpoint does not describe Racing")
    for key in ("params", "critic_params", "target_critic_params"):
        if key not in archived:
            raise ValueError(f"Missing archived SHAC weight tree: {key}")
    options = OmegaConf.to_container(config.learning.options, resolve=True)

    # Read only actor/critic variables from the older archive. The old full
    # EnvState predates procedural scenes and must NOT be resumed or upgraded.
    # This is a diagnostic on new initial states, not historical training.
    config.simulation.num_envs = worlds
    environment = create_environment(config)
    options["horizon"] = horizon
    diagnostic = Trainer(
        environment,
        kind="state",
        algorithm="shac",
        seed=777_123,
        config=options,
        actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
        method=create_method(config),
        loss=config.learning.loss,
    )
    initial = diagnostic.initialize()
    trees = {}
    for key in ("params", "critic_params", "target_critic_params"):
        reference = getattr(initial, key)
        recorded = jax.tree.map(jnp.asarray, archived[key])
        if jax.tree.structure(reference) != jax.tree.structure(recorded) or any(
            a.shape != b.shape or a.dtype != b.dtype
            for a, b in zip(jax.tree.leaves(reference), jax.tree.leaves(recorded), strict=True)
        ):
            raise ValueError(f"Archived {key} structure differs from current SHAC network")
        trees[key] = recorded
    initial = initial.replace(**trees)

    def gradients(weight):
        diagnostic.config["bootstrap_weight"] = weight

        def objective(params):
            cost, (_, _, terms) = rollout_objective(diagnostic, initial, params, bootstrap=True)
            return cost, terms

        return jax.value_and_grad(objective, has_aux=True)(initial.params)

    (full_loss, full_terms), total_grad = gradients(1.0)
    (physical_loss, physical_terms), physical_grad = gradients(0.0)
    residual = jax.tree.map(lambda a, b: a - b, total_grad, physical_grad)
    n_full = norm(total_grad)
    n_phys = norm(physical_grad)
    n_residual = norm(residual)
    inner = sum(
        jnp.vdot(a, b)
        for a, b in zip(jax.tree.leaves(physical_grad), jax.tree.leaves(residual), strict=True)
    )
    cosine = inner / jnp.maximum(n_phys * n_residual, 1e-12)
    return {
        "checkpoint": str(checkpoint),
        "test_seed_partition_used": False,
        "fresh_training_seed": 777_123,
        "worlds": worlds,
        "horizon": horizon,
        "loss_full": float(full_loss),
        "loss_without_terminal": float(physical_loss),
        "full_gradient_norm": float(n_full),
        "physical_gradient_norm": float(n_phys),
        "terminal_gradient_norm": float(n_residual),
        "terminal_to_physical_norm_ratio": float(n_residual / jnp.maximum(n_phys, 1e-12)),
        "physical_terminal_cosine_similarity": float(cosine),
        "critic_terminal_value_abs": float(full_terms["bootstrap_terminal_value_abs"]),
        "critic_terminal_vs_step_cost": float(full_terms["bootstrap_terminal_vs_step_cost"]),
        "gates_advanced": float(physical_terms["gates_advanced"]),
        "collisions_during_rollout": float(physical_terms["collision_fraction"]),
        "training_episode_randomness": "same fresh initial state and PRNG for both objectives",
    }


def main():
    """Persist reusable gradient evidence without modifying any checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=32)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("Refusing to overwrite earlier gradient diagnosis")
    result = compare(
        arguments.checkpoint.resolve(strict=True),
        worlds=arguments.worlds,
        horizon=arguments.horizon,
    )
    if not np.isfinite(list(v for v in result.values() if isinstance(v, float))).all():
        raise ValueError("Gradient diagnostics produced nonfinite values")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
