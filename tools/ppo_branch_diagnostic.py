"""Run a bounded, paired continuation from an intact PPO training checkpoint."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint, ppo
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method, evaluate
from drone_playground.simulation.records import atomic_json


def tree_hash(tree):
    """Identify the complete restored training state, including keys and counters."""
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    """Check behavior likelihoods, replay the source, and continue actual updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--scenes", nargs="+", default=["S01", "D01", "D02", "D03"])
    parser.add_argument("--actor-lr-scale", type=float, default=1.0)
    parser.add_argument("--freeze-critic", action="store_true")
    parser.add_argument("--refresh-memory", action="store_true")
    args = parser.parse_args()
    if args.updates < 0 or args.interval <= 0 or args.actor_lr_scale <= 0:
        raise ValueError("Updates must be nonnegative; LR scale and interval must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    config = OmegaConf.create(metadata["config"]["experiment"])
    config.simulation.device = "gpu"
    resolved = OmegaConf.to_container(config, resolve=True)
    options = dict(resolved["learning"]["options"])
    options["lr"] *= args.actor_lr_scale
    trainer = Trainer(
        create_environment(config),
        kind=config.method.actor.kind,
        algorithm="ppo",
        seed=config.seed,
        config=options,
        actor_config=resolved["method"]["actor"],
        method=create_method(config),
        loss=config.learning.loss,
        backward_model=config.learning.backward_model,
        backward_options=resolved["learning"]["backward_options"],
    )
    state, _ = checkpoint.load_model_state(args.checkpoint, trainer.initialize())
    origin_update = int(state.updates)
    origin_hash = tree_hash(state)
    budget = config.learning.max_interactions
    identity = {
        "checkpoint": str(args.checkpoint),
        "source_provenance": metadata["provenance"],
        "source_config": resolved,
        "restored_state_sha256": origin_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "status": "running",
    }
    atomic_json(args.output / "experiment.json", identity)
    print("restored", origin_update, origin_hash, flush=True)

    @jax.jit
    def check_rollout(before, progress):
        collected, batch, terms = ppo.collect_rollout(trainer, before, sampling_progress=progress)
        memory, (raw, _) = ppo.recurrent_policy(
            trainer, before.params, batch.observations, before.recurrent_memory, batch.done
        )
        probability = trainer.method.log_probability(
            batch.pre_tanh, raw, before.log_std, batch.observations
        )
        return {
            "log_probability_max_error": jnp.max(jnp.abs(probability - batch.log_prob)),
            "recurrent_memory_max_error": jnp.max(jnp.abs(memory - collected.recurrent_memory)),
            "reward_finite": jnp.all(jnp.isfinite(batch.rewards)),
        }

    checks = {k: float(v) for k, v in check_rollout(state, state.interactions / budget).items()}
    assert checks["log_probability_max_error"] < 1e-4, checks
    assert checks["recurrent_memory_max_error"] < 1e-5, checks
    assert checks["reward_finite"], checks
    assert tree_hash(state) == origin_hash
    atomic_json(args.output / "rollout-checks.json", checks)
    print("rollout_checks", checks, flush=True)

    @jax.jit
    def branch_update(before, progress):
        if args.refresh_memory:
            collected, batch, terms = ppo.collect_rollout(
                trainer, before, sampling_progress=progress
            )
            with patch.object(ppo, "collect_rollout", return_value=(collected, batch, terms)):
                after, metrics = ppo.update(trainer, before, sampling_progress=progress)
            memory, _ = ppo.recurrent_policy(
                trainer, after.params, batch.observations, before.recurrent_memory, batch.done
            )
            after = after.replace(recurrent_memory=memory)
        else:
            after, metrics = ppo.update(trainer, before, sampling_progress=progress)
        if args.freeze_critic:
            after = after.replace(
                critic_params=before.critic_params,
                optimizer_state={**after.optimizer_state, "critic": before.optimizer_state["critic"]},
            )
        return after, metrics

    evaluation = OmegaConf.create(resolved)
    evaluation.mode = "checkpoint_eval"
    evaluation.benchmark.protocol = "diagnostic"
    evaluation.benchmark.episodes = 8
    evaluation.benchmark.scenes = args.scenes
    evaluation.benchmark.seed_base = 1_000_000
    evaluation.benchmark.record_replay = False
    evaluation.benchmark.record_trajectories = False
    environments = {}
    started = time.monotonic()
    try:
        for offset in range(args.updates + 1):
            if offset % args.interval == 0 or offset == args.updates:
                report = evaluate(
                    evaluation,
                    args.output / f"update-{int(state.updates):06d}",
                    actor=trainer.actor,
                    parameters=state.params,
                    environments=environments,
                )
                scores = {k: v["successes"] for k, v in report["reports"].items()}
                print("evaluation", int(state.updates), scores, flush=True)
                if offset == 0:
                    with (args.checkpoint / "screening" / "episodes.csv").open() as stream:
                        source_rows = list(csv.DictReader(stream))
                    with (args.output / f"update-{origin_update:06d}" / "episodes.csv").open() as stream:
                        replay_rows = list(csv.DictReader(stream))
                    fields = ("scene", "geometry_sha256", "initialization_seed", "world_index")
                    expected = {
                        tuple(row[k] for k in fields): row["event"]
                        for row in source_rows if row["scene"] in args.scenes
                    }
                    actual = {tuple(row[k] for k in fields): row["event"] for row in replay_rows}
                    assert expected == actual, (expected, actual)
            if offset == args.updates:
                break
            updates, interactions = int(state.updates), int(state.interactions)
            state, metrics = branch_update(
                state.replace(updates=0, interactions=0),
                jnp.asarray(min(interactions / budget, 1.0), jnp.float32),
            )
            state = state.replace(
                updates=updates + 1,
                interactions=interactions + trainer.config["horizon"] * trainer.env.num_envs,
            )
            if (offset + 1) % 10 == 0:
                record = {k: float(v) for k, v in jax.device_get(metrics).items()}
                record.update(update=int(state.updates), wall_seconds=time.monotonic() - started)
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print("update", int(state.updates), "reward", record["reward"], flush=True)
        checkpoint.save_state(
            args.output / "final.training.zip",
            state,
            config={"experiment": resolved, "learning": trainer.resolved_config},
            provenance={"diagnostic": True, "source_checkpoint": str(args.checkpoint)},
        )
        identity.update(status="completed", final_update=int(state.updates))
        atomic_json(args.output / "experiment.json", identity)
    finally:
        trainer.env.sim.close()
        for env in environments.values():
            if hasattr(env, "sim"):
                env.sim.close()


if __name__ == "__main__":
    main()
