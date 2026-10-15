"""Estimate full stochastic returns at repeated training states for two frozen policies."""

import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint
from drone_playground.learning.objective import EpisodeObjective
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.networks import reset_actor_memory
from drone_playground.simulation.records import atomic_json, source_identity
from drone_playground.simulation.tasks import Event

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[2] / "tools"))
from ppo_branch_diagnostic import tree_hash  # noqa: E402


def prepare():
    """Restore frozen learners and exactly repeat the registered training states."""
    source = root / "depth-batch64-retention-pair/model_000300"
    metadata = json.loads((source / "metadata.json").read_text())
    config = OmegaConf.create(metadata["config"]["experiment"])
    assert config.scene.name == "train/procedural" and config.learning.options.gamma == 1
    assert config.task.options.finite_horizon and config.learning.loss is None
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
    template = trainer.initialize()
    before, _ = checkpoint.load_model_state(source, template)
    after_path = root / "depth-clearance-control/final.training.zip"
    after, after_metadata = checkpoint.load_state(after_path, template)
    assert after_metadata["config"]["experiment"] == metadata["config"]["experiment"]
    assert int(before.updates) == 300 and int(after.updates) == 480
    env = create_environment(config, num_envs=128)
    progress = int(before.interactions) / config.learning.max_interactions
    reset_key = jax.random.fold_in(before.rng, 730020)
    sampled = env.reset(reset_key, sampling_progress=progress)

    def repeat_worlds(value):
        """Replicate the first eight complete worlds within the existing128-world shape."""
        if getattr(value, "shape", ()) and value.shape[0] == 128:
            return jnp.repeat(value[:8], 16, axis=0)
        return value

    initial = jax.tree.map(repeat_worlds, sampled)
    for value in jax.tree.leaves(initial):
        if getattr(value, "shape", ()) and value.shape[0] == 128:
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                value = jax.random.key_data(value)
            grouped = np.asarray(value).reshape(8, 16, *value.shape[1:])
            assert np.array_equal(grouped, np.repeat(grouped[:, :1], 16, axis=1))
    return config, trainer, env, initial, [("u300", before, source), ("u480", after, after_path)]


def main():
    """Collect complete frozen-policy Monte Carlo returns without overwriting evidence."""
    output = root / "depth-critic-calibration"
    output.mkdir(exist_ok=False)
    config, trainer, env, initial, models = prepare()
    before = models[0][1]
    objective = EpisodeObjective(env, trainer.config)
    progress = int(before.interactions) / config.learning.max_interactions
    reset_key = jax.random.fold_in(before.rng, 730020)
    action_key = jax.random.fold_in(before.rng, 730021)

    @jax.jit
    def chunk(carry, params, log_std):
        """Use the actual PPO action distribution and shared training objective."""

        def step(carry, _):
            current, memory, history, rng = carry
            rng, key = jax.random.split(rng)
            raw, memory, _ = trainer.actor.apply(params, env.observe(current), memory)
            action, _, _ = trainer.method.sample(key, raw, log_std)
            following = trainer.method.execute(env, current, action)
            cost, history, _ = objective.cost(current, following, action, history)
            data = {
                "reward": -cost,
                "active": ~current.done,
                "event": following.task.event,
                "time": following.time,
                "position": following.physics.states.pos[:, 0],
                "action": action,
            }
            return (following, reset_actor_memory(memory, following.done), history, rng), data

        return jax.lax.scan(step, carry, None, length=25)

    result = {
        "runtime_provenance": source_identity(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": OmegaConf.to_container(config, resolve=True),
        "initial_state_sha256": tree_hash(initial),
        "sampling_progress": progress,
        "reset_key": np.asarray(jax.random.key_data(reset_key)).tolist(),
        "action_key": np.asarray(jax.random.key_data(action_key)).tolist(),
        "scope": "Frozen train-state Monte Carlo calibration; no learning or formal acceptance",
        "status": "running",
        "models": [],
    }
    atomic_json(output / "report.json", result)
    try:
        for name, state, path in models:
            state_hash = tree_hash(state)
            observation = {
                **env.observe(initial),
                "privileged_state": env.observe_privileged(initial),
            }
            prediction = np.asarray(trainer.critic.apply(state.critic_params, observation)).reshape(
                8, 16
            )
            assert np.max(np.abs(prediction - prediction[:, :1])) < 1e-5
            carry = (
                initial,
                trainer.actor.initialize_memory(128),
                objective.initialize_state(128),
                action_key,
            )
            started = time.monotonic()
            chunks = []
            for index in range(int(np.ceil(env.task.duration / env.dt / 25))):
                carry, data = chunk(carry, state.params, state.log_std)
                chunks.append(jax.device_get(data))
                if index % 20 == 0:
                    print(name, index * 25, "completed", int(carry[0].done.sum()), flush=True)
                if np.asarray(carry[0].done).all():
                    break
            assert (
                np.asarray(carry[0].terminated).all() and not np.asarray(carry[0].truncated).any()
            )
            assert (
                tree_hash(state) == state_hash
                and tree_hash(initial) == result["initial_state_sha256"]
            )
            data = {key: np.concatenate([item[key] for item in chunks]) for key in chunks[0]}
            assert np.isfinite(data["reward"]).all() and np.all(
                data["reward"][~data["active"]] == 0
            )
            returns = data["reward"].sum(axis=0, dtype=np.float64).reshape(8, 16)
            events = np.asarray(carry[0].task.event).reshape(8, 16)
            np.savez_compressed(
                output / f"{name}.npz", **data, initial_prediction=prediction, returns=returns
            )
            rng = np.random.default_rng(730022)
            rows = []
            for group in range(8):
                bootstrap = returns[group, rng.integers(0, 16, (10000, 16))].mean(axis=1)
                rows.append(
                    {
                        "state": group,
                        "prediction": float(prediction[group, 0]),
                        "returns": returns[group].tolist(),
                        "mean_return": float(returns[group].mean()),
                        "return_standard_error": float(returns[group].std(ddof=1) / 4),
                        "descriptive_bootstrap_95_interval": np.quantile(
                            bootstrap, [0.025, 0.975]
                        ).tolist(),
                        "prediction_minus_mc_mean": float(
                            prediction[group, 0] - returns[group].mean()
                        ),
                        "events": [Event(int(event)).name for event in events[group]],
                    }
                )
            result["models"].append(
                {
                    "name": name,
                    "checkpoint": str(path),
                    "state_sha256": state_hash,
                    "raw_sha256": hashlib.sha256((output / f"{name}.npz").read_bytes()).hexdigest(),
                    "active_interactions": int(data["active"].sum()),
                    "allocated_simulation_slots": data["active"].size,
                    "wall_seconds": time.monotonic() - started,
                    "states": rows,
                }
            )
            atomic_json(output / "report.json", result)
            print(
                name,
                "biases",
                [round(row["prediction_minus_mc_mean"], 3) for row in rows],
                flush=True,
            )
        result["status"] = "completed"
        atomic_json(output / "report.json", result)
    finally:
        env.sim.close()
        trainer.env.sim.close()


if __name__ == "__main__":
    main()
