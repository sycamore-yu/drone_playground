"""Measure frozen-policy return dependence on carried memory at identical states."""

import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.networks import reset_actor_memory
from drone_playground.simulation.records import atomic_json
from drone_playground.simulation.tasks import Event


def main():
    """Run paired complete returns while preserving all original training evidence."""
    root = Path(__file__).resolve().parent
    output = root / "depth-critic-memory"
    output.mkdir(exist_ok=False)
    source = root / "depth-batch64-retention-pair/model_000300"
    verified = json.loads((root / "depth-h512-lambda1-minibatches4/experiment.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    assert verified["status"] == "completed" and Path(verified["checkpoint"]) == source
    assert verified["source_config"] == metadata["config"]["experiment"]
    config = OmegaConf.create(metadata["config"]["experiment"])
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
    env, objective = trainer.env, trainer.objective
    assert env.num_envs == 64 and config.task.options.finite_horizon
    age = np.asarray(state.env_state.time)
    norms = np.linalg.norm(np.asarray(state.recurrent_memory), axis=-1)
    eligible = np.flatnonzero((age >= 1) & (norms > 1e-6) & ~np.asarray(state.env_state.done))
    assert len(eligible) >= 8, len(eligible)
    ranked = eligible[np.argsort(age[eligible])]
    indices = ranked[np.linspace(0, len(ranked) - 1, 8, dtype=int)]

    def repeat(value):
        if getattr(value, "shape", ()) and value.shape[0] == 64:
            return jnp.repeat(value[indices], 8, axis=0)
        return value

    initial = jax.tree.map(repeat, state.env_state)
    history = jax.tree.map(repeat, state.objective_state)
    actual_memory = repeat(state.recurrent_memory)
    zero_memory = jnp.zeros_like(actual_memory)
    observation = trainer._observe(initial)
    predictions = np.asarray(trainer.critic.apply(state.critic_params, observation)).reshape(8, 8)
    assert np.max(np.abs(predictions - predictions[:, :1])) < 1e-5
    actual_raw = trainer.actor.apply(state.params, observation, actual_memory)[0]
    zero_raw = trainer.actor.apply(state.params, observation, zero_memory)[0]
    raw_difference = np.asarray(jnp.linalg.norm(actual_raw - zero_raw, axis=-1)).reshape(8, 8)

    @jax.jit
    def chunk(carry):
        def step(carry, _):
            before, memory, past, rng = carry
            rng, action_key = jax.random.split(rng)
            raw, memory, _ = trainer.actor.apply(state.params, env.observe(before), memory)
            action, _, _ = trainer.method.sample(action_key, raw, state.log_std)
            after = trainer.method.execute(env, before, action)
            cost, past, _ = objective.cost(before, after, action, past)
            data = {
                "reward": -cost,
                "active": ~before.done,
                "event": after.task.event,
                "position": after.physics.states.pos[:, 0],
                "action": action,
            }
            return (after, reset_actor_memory(memory, after.done), past, rng), data

        return jax.lax.scan(step, carry, None, length=25)

    result = {
        "status": "running",
        "source": str(source),
        "source_state_identity_reused": verified["restored_state_sha256"],
        "runtime_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root.parents[2], text=True
        ).strip(),
        "source_world_indices": indices.tolist(),
        "source_ages_seconds": age[indices].tolist(),
        "source_memory_norms": norms[indices].tolist(),
        "initial_critic_predictions": predictions[:, 0].tolist(),
        "initial_raw_action_difference_norm": raw_difference[:, 0].tolist(),
        "exploration_std": np.asarray(jnp.exp(state.log_std)).tolist(),
        "runs": [],
    }
    atomic_json(output / "report.json", result)
    returns, events = {"actual": [], "zero": []}, {"actual": [], "zero": []}
    try:
        for replica in range(2):
            key = jax.random.fold_in(state.rng, 740000 + replica)
            for name, memory in (("actual", actual_memory), ("zero", zero_memory)):
                carry = initial, memory, history, key
                blocks = []
                started = time.monotonic()
                for index in range(int(np.ceil(env.task.duration / env.dt / 25))):
                    carry, data = chunk(carry)
                    blocks.append(jax.device_get(data))
                    if index % 20 == 0:
                        print(name, replica, index * 25, int(carry[0].done.sum()), flush=True)
                    if np.asarray(carry[0].done).all():
                        break
                assert np.asarray(carry[0].terminated).all()
                raw = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
                assert np.isfinite(raw["reward"]).all()
                assert np.all(raw["reward"][~raw["active"]] == 0)
                scores = raw["reward"].sum(axis=0, dtype=np.float64).reshape(8, 8)
                outcomes = np.asarray(carry[0].task.event).reshape(8, 8)
                returns[name].append(scores)
                events[name].append(outcomes)
                np.savez_compressed(output / f"{name}-{replica}.npz", **raw, returns=scores)
                result["runs"].append(
                    {
                        "memory": name,
                        "replica_batch": replica,
                        "action_key": np.asarray(jax.random.key_data(key)).tolist(),
                        "active_interactions": int(raw["active"].sum()),
                        "allocated_slots": raw["active"].size,
                        "wall_seconds": time.monotonic() - started,
                    }
                )
                atomic_json(output / "report.json", result)
        returns = {k: np.concatenate(v, axis=1) for k, v in returns.items()}
        events = {k: np.concatenate(v, axis=1) for k, v in events.items()}
        rng = np.random.default_rng(740002)
        rows = []
        for index in range(8):
            difference = returns["actual"][index] - returns["zero"][index]
            samples = difference[rng.integers(0, 16, size=(10000, 16))].mean(axis=1)
            interval = np.quantile(samples, [0.025, 0.975])
            rows.append(
                {
                    "world": int(indices[index]),
                    "returns": {k: v[index].tolist() for k, v in returns.items()},
                    "events": {
                        k: [Event(int(e)).name for e in v[index]] for k, v in events.items()
                    },
                    "paired_mean_difference": float(difference.mean()),
                    "descriptive_paired_95_interval": interval.tolist(),
                    "material_difference": bool(
                        abs(difference.mean()) > 0.2 and (interval[0] > 0 or interval[1] < 0)
                    ),
                }
            )
        result.update(
            status="completed",
            states=rows,
            supports_memory_intervention=sum(r["material_difference"] for r in rows) >= 2,
        )
        atomic_json(output / "report.json", result)
        print("completed", [(r["world"], r["paired_mean_difference"]) for r in rows], flush=True)
    finally:
        env.sim.close()


if __name__ == "__main__":
    main()
