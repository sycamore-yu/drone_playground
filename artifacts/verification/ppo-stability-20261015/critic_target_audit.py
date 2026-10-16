"""Compare actual GAE targets against stored complete returns on the same trajectories."""

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from critic_mc_calibration import prepare, root, tree_hash

from drone_playground.learning.objective import EpisodeObjective
from drone_playground.learning.ppo import generalized_advantage_estimate
from drone_playground.simulation.records import atomic_json, source_identity

prior = root / "depth-critic-calibration"
output = root / "depth-critic-targets"
output.mkdir(exist_ok=False)
calibration = json.loads((prior / "report.json").read_text())
assert calibration["status"] == "completed"
config, trainer, env, initial, models = prepare()
assert tree_hash(initial) == calibration["initial_state_sha256"]
objective = EpisodeObjective(env, trainer.config)


@jax.jit
def replay(carry, actions, critic_params):
    """Replay recorded actions and evaluate the unchanged critic at every boundary."""

    def step(carry, action):
        before, history = carry
        obs = {**env.observe(before), "privileged_state": env.observe_privileged(before)}
        value = trainer.critic.apply(critic_params, obs)
        after = trainer.method.execute(env, before, action)
        cost, history, _ = objective.cost(before, after, action, history)
        return (after, history), {
            "value": value,
            "reward": -cost,
            "event": after.task.event,
            "position": after.physics.states.pos[:, 0],
            "terminated": after.terminated,
            "done": after.done,
        }

    return jax.lax.scan(step, carry, actions)


result = {
    "runtime_provenance": source_identity(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "setup_script_sha256": hashlib.sha256(
        (root / "critic_mc_calibration.py").read_bytes()
    ).hexdigest(),
    "calibration_report_sha256": hashlib.sha256((prior / "report.json").read_bytes()).hexdigest(),
    "initial_state_sha256": tree_hash(initial),
    "horizons": [32, 128, 512],
    "lambdas": [0.95, 1.0],
    "scope": "Fixed recorded on-policy trajectories; estimator comparison, no parameter updates",
    "status": "running",
    "models": [],
}
atomic_json(output / "report.json", result)
try:
    for (name, state, _), original in zip(models, calibration["models"], strict=True):
        assert tree_hash(state) == original["state_sha256"]
        raw_path = prior / f"{name}.npz"
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == original["raw_sha256"]
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        assert raw["action"].shape[0] >= 512
        carry = initial, objective.initialize_state(128)
        blocks = []
        started = time.monotonic()
        for start in range(0, 512, 32):
            carry, block = replay(
                carry, jnp.asarray(raw["action"][start : start + 32]), state.critic_params
            )
            blocks.append(jax.device_get(block))
        replayed = {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}
        position_error = float(np.max(np.abs(replayed["position"] - raw["position"][:512])))
        reward_error = float(np.max(np.abs(replayed["reward"] - raw["reward"][:512])))
        assert position_error < 1e-4, position_error
        assert reward_error < 1e-6, reward_error
        np.testing.assert_array_equal(replayed["event"], raw["event"][:512])
        last_obs = {**env.observe(carry[0]), "privileged_state": env.observe_privileged(carry[0])}
        last_value = np.asarray(trainer.critic.apply(state.critic_params, last_obs))
        values = np.concatenate([replayed["value"], last_value[None]])
        np.testing.assert_allclose(values[0].reshape(8, 16), raw["initial_prediction"], atol=1e-5)
        targets = []
        saved = {}
        for horizon in result["horizons"]:
            for lam in result["lambdas"]:
                rewards = raw["reward"][:horizon]
                terminated = replayed["terminated"][:horizon]
                done = replayed["done"][:horizon]
                advantage, target = generalized_advantage_estimate(
                    jnp.asarray(rewards),
                    jnp.asarray(values[:horizon]),
                    jnp.asarray(values[1 : horizon + 1]),
                    jnp.asarray(terminated),
                    jnp.asarray(done),
                    gamma=1.0,
                    gae_lambda=lam,
                )
                independent = np.zeros(128, dtype=np.float64)
                for step in reversed(range(horizon)):
                    delta = rewards[step] + values[step + 1] * ~terminated[step] - values[step]
                    independent = delta + lam * ~done[step] * independent
                target0 = np.asarray(target[0]).reshape(8, 16)
                np.testing.assert_allclose(target0.ravel(), independent + values[0], atol=1e-4)
                errors = target0 - raw["returns"]
                saved[f"target_h{horizon}_lambda{lam}"] = target0
                targets.append(
                    {
                        "horizon": horizon,
                        "lambda": lam,
                        "per_state_mean_target": target0.mean(axis=1).tolist(),
                        "per_state_mean_error": errors.mean(axis=1).tolist(),
                        "sample_rmse": float(np.sqrt(np.mean(errors**2))),
                        "mean_absolute_state_bias": float(np.abs(errors.mean(axis=1)).mean()),
                    }
                )
        np.savez_compressed(output / f"{name}.npz", **replayed, values=values, **saved)
        assert tree_hash(state) == original["state_sha256"]
        result["models"].append(
            {
                "name": name,
                "max_replay_position_error_m": position_error,
                "max_replay_reward_error": reward_error,
                "exact_event_replay": True,
                "allocated_replay_slots": 512 * 128,
                "wall_seconds": time.monotonic() - started,
                "raw_sha256": hashlib.sha256((output / f"{name}.npz").read_bytes()).hexdigest(),
                "targets": targets,
            }
        )
        atomic_json(output / "report.json", result)
        print(
            name,
            [(t["horizon"], t["lambda"], round(t["mean_absolute_state_bias"], 3)) for t in targets],
            flush=True,
        )
    result["status"] = "completed"
    atomic_json(output / "report.json", result)
finally:
    env.sim.close()
    trainer.env.sim.close()
