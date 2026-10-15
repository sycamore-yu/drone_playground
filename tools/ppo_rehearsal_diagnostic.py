"""Collect training-only self-generated sequences and test policy-cloning regularization.

This bounded diagnostic is PPO plus policy cloning, not SIL or the complete CLEAR
algorithm. Historical data never enter the PPO ratio, GAE or critic fitting.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint, ppo
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.networks import reset_actor_memory
from drone_playground.simulation.records import atomic_json, source_identity
from drone_playground.simulation.tasks import Event


def digest(path):
    """Identify the exact immutable corpus used by a continuation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian_kl(teacher_mean, teacher_log_std, mean, log_std):
    """KL(teacher || current); the shared bijective tanh Jacobian cancels."""
    return jnp.sum(
        log_std
        - teacher_log_std
        + 0.5
        * (
            jnp.exp(2 * (teacher_log_std - log_std))
            + (teacher_mean - mean) ** 2 * jnp.exp(-2 * log_std)
            - 1
        ),
        axis=-1,
    )


def install_rehearsal(trainer, state, directory, weight):
    """Add only the historical-policy KL gradient to executed actor updates."""
    from ppo_branch_diagnostic import tree_hash

    directory = Path(directory)
    metadata = json.loads((directory / "corpus.json").read_text())
    if (
        trainer.algorithm != "ppo"
        or not np.isfinite(weight)
        or weight < 0
        or metadata["scene"] != "train/procedural"
        or metadata["source_state_sha256"] != tree_hash(state)
        or metadata["actor"] != trainer.actor.specification
        or metadata["corpus_sha256"] != digest(directory / "corpus.npz")
    ):
        raise ValueError("Rehearsal must match this PPO state and a verified training corpus")
    with np.load(directory / "corpus.npz", allow_pickle=False) as archive:
        data = {key: jnp.asarray(archive[key]) for key in archive.files}
    observation = {key[5:]: value for key, value in data.items() if key.startswith("obs__")}
    count = data["initial_memory"].shape[0]
    if count < 4 or data["teacher_mean"].shape[:2] != (64, count):
        raise ValueError("Expected at least four complete 64-step teacher sequences")

    def loss(variables, rng):
        indices = jax.random.randint(jax.random.fold_in(rng, 730018), (4,), 0, count)
        obs = jax.tree.map(lambda value: value[:, indices], observation)
        _, (mean, _) = ppo.recurrent_policy(
            trainer,
            variables["params"],
            obs,
            data["initial_memory"][indices],
            data["done"][:, indices],
        )
        return gaussian_kl(
            data["teacher_mean"][:, indices],
            data["teacher_log_std"],
            mean,
            variables["log_std"],
        )[32:].mean()

    original_apply = trainer._apply_actor

    def apply_actor(current, gradients):
        variables = {"params": current.params, "log_std": current.log_std}
        regularizer = jax.grad(loss)(variables, current.rng)
        combined = jax.tree.map(lambda a, b: a + weight * b, gradients, regularizer)
        return original_apply(current, combined)

    if weight:
        trainer._apply_actor = apply_actor
    return loss, {
        "weight": weight,
        "corpus": str(directory.resolve()),
        "corpus_sha256": metadata["corpus_sha256"],
        "metadata_sha256": digest(directory / "corpus.json"),
        "source_active_interactions": metadata["active_interactions"],
        "sequences_per_minibatch": 4,
        "burn_in_steps": 32,
        "loss_steps": 32,
        "scope": "PPO plus historical-policy KL; logged PPO actor loss excludes this extra term",
    }


def collect(source, output):
    """Run one fresh procedural batch with frozen PPO parameters and independent RNG."""
    from ppo_branch_diagnostic import tree_hash

    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source / "metadata.json").read_text())
    config = OmegaConf.create(metadata["config"]["experiment"])
    if (
        config.scene.name != "train/procedural"
        or config.learning.algorithm != "ppo"
        or config.method.actor.kind != "depth"
    ):
        raise ValueError("This bounded corpus pilot requires procedural Depth PPO")
    trainer = Trainer(
        create_environment(config),
        kind=config.method.actor.kind,
        algorithm="ppo",
        seed=config.seed,
        config=OmegaConf.to_container(config.learning.options),
        actor_config=OmegaConf.to_container(config.method.actor),
        method=create_method(config),
        loss=config.learning.loss,
        backward_model=config.learning.backward_model,
        backward_options=OmegaConf.to_container(config.learning.backward_options),
    )
    state, _ = checkpoint.load_model_state(source, trainer.initialize())
    source_hash = tree_hash(state)
    progress = int(state.interactions) / config.learning.max_interactions
    reset_key = jax.random.fold_in(state.rng, 730015)
    initial = trainer.env.reset(reset_key, sampling_progress=progress)
    carry = (
        initial,
        trainer.actor.initialize_memory(trainer.env.num_envs),
        jax.random.fold_in(state.rng, 730016),
    )

    @jax.jit
    def chunk(carry):
        def step(carry, _):
            before, memory, rng = carry
            rng, action_key = jax.random.split(rng)
            obs = trainer.env.observe(before)
            raw, new_memory, _ = trainer.actor.apply(state.params, obs, memory)
            action, _, _ = trainer.method.sample(action_key, raw, state.log_std)
            after = trainer.method.execute(trainer.env, before, action)
            records = {
                **{f"obs__{key}": obs[key] for key in ("state", "depth", "depth_mask")},
                "memory": memory,
                "teacher_mean": raw,
                "active": ~before.done,
                "done": after.done,
                "event": after.task.event,
                "time": after.time,
                "position": after.physics.states.pos[:, 0],
                "action": action,
            }
            return (after, reset_actor_memory(new_memory, after.done), rng), records

        return jax.lax.scan(step, carry, None, length=25)

    started = time.monotonic()
    chunks = []
    max_chunks = int(np.ceil(trainer.env.task.duration / trainer.env.dt / 25)) + 1
    for index in range(max_chunks):
        carry, records = chunk(carry)
        chunks.append(jax.device_get(records))
        if index % 20 == 0:
            print("collection", index * 25, "completed", int(carry[0].done.sum()), flush=True)
        if np.asarray(carry[0].done).all():
            break
    assert np.asarray(carry[0].done).all()
    traces = {key: np.concatenate([row[key] for row in chunks]) for key in chunks[0]}
    lengths = traces["active"].sum(axis=0).astype(int)
    events = np.asarray(carry[0].task.event)
    starts, worlds, episodes = [], [], []
    rng = np.random.default_rng(730017)
    for world, length in enumerate(lengths):
        event = Event(int(events[world])).name
        episodes.append({"world": world, "steps": int(length), "event": event})
        if event != "SUCCESS" or length < 64:
            continue
        selected = rng.choice(length - 63, size=min(4, length - 63), replace=False)
        worlds.extend([world] * len(selected))
        starts.extend(map(int, selected))
    assert len(set(worlds)) >= 8, "Insufficient complete training successes for this diagnostic"
    data = {
        key: np.stack(
            [value[start : start + 64, world] for world, start in zip(worlds, starts, strict=True)],
            axis=1,
        )
        for key, value in traces.items()
        if key.startswith("obs__") or key in ("teacher_mean", "done")
    }
    data.update(
        initial_memory=np.stack(
            [traces["memory"][start, world] for world, start in zip(worlds, starts, strict=True)]
        ),
        teacher_log_std=np.asarray(state.log_std),
        source_world=np.asarray(worlds),
        source_start=np.asarray(starts),
    )
    obs = {key[5:]: jnp.asarray(value) for key, value in data.items() if key.startswith("obs__")}
    _, (recomputed, _) = jax.jit(
        lambda params: ppo.recurrent_policy(
            trainer, params, obs, jnp.asarray(data["initial_memory"]), jnp.asarray(data["done"])
        )
    )(state.params)
    max_error = float(jnp.max(jnp.abs(recomputed - data["teacher_mean"])))
    assert max_error < 1e-4, max_error
    assert tree_hash(state) == source_hash
    np.savez_compressed(output / "corpus.npz", **data)
    np.savez_compressed(
        output / "collection-traces.npz",
        **{
            key: value
            for key, value in traces.items()
            if key in ("active", "event", "time", "position", "action")
        },
    )
    result = {
        "source": str(source.resolve()),
        "source_state_sha256": source_hash,
        "source_config": metadata["config"]["experiment"],
        "source_identity": source_identity(),
        "actor": trainer.actor.specification,
        "scene": "train/procedural",
        "initial_state_sha256": tree_hash(initial),
        "reset_key": np.asarray(jax.random.key_data(reset_key)).tolist(),
        "script_sha256": digest(Path(__file__)),
        "sampling_progress": progress,
        "episodes": episodes,
        "eligible_successes": len(set(worlds)),
        "active_interactions": int(lengths.sum()),
        "allocated_simulation_slots": len(traces["active"]) * trainer.env.num_envs,
        "collection_wall_seconds": time.monotonic() - started,
        "sequences": len(worlds),
        "source_policy_recompute_max_error": max_error,
        "corpus_sha256": digest(output / "corpus.npz"),
        "traces_sha256": digest(output / "collection-traces.npz"),
        "limitation": (
            "Stored teacher memory plus32-step burn-in approximates changed-policy history"
        ),
    }
    atomic_json(output / "corpus.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "active_interactions",
                    "eligible_successes",
                    "sequences",
                    "source_policy_recompute_max_error",
                )
            }
        ),
        flush=True,
    )
    trainer.env.sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.checkpoint, args.output)
