"""Run a bounded, paired continuation from an intact PPO training checkpoint."""

import argparse
import copy
import csv
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint, ppo
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method, evaluate
from drone_playground.simulation.records import atomic_json, source_identity


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


def reuse_source_checks(directory, source, config, branch_config):
    """Reuse source identity/replay, and rollout checks when sampling is unchanged."""
    identity = json.loads((directory / "experiment.json").read_text())
    checks = json.loads((directory / "rollout-checks.json").read_text())
    previous = identity["branch_config"]
    current = copy.deepcopy(branch_config)
    for candidate in (previous, current):
        for key in ("minibatches", "gae_lambda"):
            candidate["learning"]["options"].pop(key, None)
    old_sampling = previous.get("simulation", {}).get("procedural", {})
    new_sampling = current.get("simulation", {}).get("procedural", {})
    changed_sampling = (
        old_sampling.get("start_sampling") in {"distance_curriculum", "distance_mixture"}
        and new_sampling.get("start_sampling") == "stratified"
    )
    if changed_sampling:
        old_sampling["start_sampling"] = "stratified"
    if (
        identity.get("status") != "completed"
        or Path(identity["checkpoint"]).resolve() != source.resolve()
        or identity["source_config"] != config
        or previous != current
        or checks["reward_finite"] != 1
        or not checks["log_probability_max_error"] < 1e-4
        or not checks["recurrent_memory_max_error"] < 1e-5
    ):
        raise ValueError("Existing source verification does not match this checkpoint/rollout")
    return identity["restored_state_sha256"], None if changed_sampling else checks


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
    parser.add_argument("--clearance-aggregation", choices=("time_average", "minimum"))
    parser.add_argument("--deterministic-update", action="store_true")
    parser.add_argument("--rehearsal", type=Path)
    parser.add_argument("--rehearsal-weight", type=float, default=0.1)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--minibatches", type=int)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--reuse-source-checks", type=Path)
    parser.add_argument("--start-sampling", choices=("stratified",))
    parser.add_argument("--critic-fit-multiplier", type=int, default=1)
    args = parser.parse_args()
    if args.updates < 0 or args.interval <= 0 or args.actor_lr_scale <= 0:
        raise ValueError("Updates must be nonnegative; LR scale and interval must be positive")
    if args.critic_fit_multiplier < 1:
        raise ValueError("Critic fit multiplier must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    config = OmegaConf.create(metadata["config"]["experiment"])
    config.simulation.device = "gpu"
    for key in ("horizon", "minibatches", "gae_lambda"):
        if (value := getattr(args, key)) is not None:
            config.learning.options[key] = value
    if args.start_sampling is not None:
        config.simulation.procedural.start_sampling = args.start_sampling
    if args.clearance_aggregation is not None:
        if config.task.options.get("reward_profile") != "goal_reaching":
            raise ValueError("Clearance aggregation requires the goal-reaching Task reward")
        OmegaConf.update(
            config,
            "task.options.reward_options.clearance_aggregation",
            args.clearance_aggregation,
            force_add=True,
        )
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
    reused_checks = None
    if args.reuse_source_checks is not None:
        origin_hash, reused_checks = reuse_source_checks(
            args.reuse_source_checks,
            args.checkpoint,
            metadata["config"]["experiment"],
            resolved,
        )
    else:
        origin_hash = tree_hash(state)
    rehearsal_loss, rehearsal_contract = None, None
    if args.rehearsal is not None:
        from ppo_rehearsal_diagnostic import install_rehearsal

        rehearsal_loss, rehearsal_contract = install_rehearsal(
            trainer, state, args.rehearsal, args.rehearsal_weight
        )
    budget = config.learning.max_interactions
    identity = {
        "checkpoint": str(args.checkpoint),
        "runtime_provenance": {
            **source_identity(),
            "jax_version": jax.__version__,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
        },
        "source_provenance": metadata["provenance"],
        "source_config": metadata["config"]["experiment"],
        "branch_config": resolved,
        "restored_state_sha256": origin_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "status": "running",
        "rehearsal": rehearsal_contract,
        "source_checks_reused_from": (
            str(args.reuse_source_checks) if args.reuse_source_checks is not None else None
        ),
        "rollout_checks_reused": reused_checks is not None,
    }
    atomic_json(args.output / "experiment.json", identity)
    print("restored", origin_update, origin_hash, flush=True)

    @jax.jit
    def check_rollout(before, progress):
        collected, batch, _ = ppo.collect_rollout(trainer, before, sampling_progress=progress)
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

    checks = reused_checks
    if checks is None:
        checks = {k: float(v) for k, v in check_rollout(state, state.interactions / budget).items()}
        if args.reuse_source_checks is None:
            assert tree_hash(state) == origin_hash
    assert checks["log_probability_max_error"] < 1e-4, checks
    assert checks["recurrent_memory_max_error"] < 1e-5, checks
    assert checks["reward_finite"], checks
    atomic_json(args.output / "rollout-checks.json", checks)
    print("rollout_checks", checks, flush=True)
    if args.critic_fit_multiplier != 1:
        if trainer.config["shared_visual_encoder"] or args.freeze_critic:
            raise ValueError("Extra critic fits require a separate, unfrozen critic")
        fit_critic = trainer._fit_critic

        def extra_critic_fits(state, observations, targets, epochs, *, old_values=None):
            return fit_critic(
                state,
                observations,
                targets,
                epochs * args.critic_fit_multiplier,
                old_values=old_values,
            )

        trainer._fit_critic = extra_critic_fits

    @partial(
        jax.jit,
        compiler_options=(
            {"xla_gpu_exclude_nondeterministic_ops": True, "xla_gpu_autotune_level": 0}
            if args.deterministic_update
            else None
        ),
    )
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
                optimizer_state={
                    **after.optimizer_state,
                    "critic": before.optimizer_state["critic"],
                },
            )
        if rehearsal_loss is not None:
            metrics["rehearsal_kl_probe"] = rehearsal_loss(
                {"params": after.params, "log_std": after.log_std}, after.rng
            )
        if args.critic_fit_multiplier != 1:
            metrics["critic_optimizer_steps"] = (
                after.optimizer_state["critic"][1][0].count
                - before.optimizer_state["critic"][1][0].count
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
            if (offset % args.interval == 0 or offset == args.updates) and not (
                offset == 0 and args.reuse_source_checks is not None
            ):
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
                    with (
                        args.output / f"update-{origin_update:06d}" / "episodes.csv"
                    ).open() as stream:
                        replay_rows = list(csv.DictReader(stream))
                    fields = ("scene", "geometry_sha256", "initialization_seed", "world_index")
                    expected = {
                        tuple(row[k] for k in fields): row["event"]
                        for row in source_rows
                        if row["scene"] in args.scenes
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
            record = {k: float(v) for k, v in jax.device_get(metrics).items()}
            if args.critic_fit_multiplier != 1:
                expected = (
                    trainer.config["ppo_epochs"]
                    * trainer.config["minibatches"]
                    * args.critic_fit_multiplier
                )
                assert record["critic_optimizer_steps"] == expected, record
            if args.start_sampling is not None:
                scene = state.env_state.scene_state
                distances = np.linalg.norm(np.asarray(scene.start - scene.goal), axis=-1)
                record.update(
                    active_start_distance_min_metres=float(distances.min()),
                    active_start_distance_max_metres=float(distances.max()),
                    active_start_distance_over90m_worlds=int((distances >= 90).sum()),
                )
            record.update(update=int(state.updates), wall_seconds=time.monotonic() - started)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if (offset + 1) % 10 == 0:
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
