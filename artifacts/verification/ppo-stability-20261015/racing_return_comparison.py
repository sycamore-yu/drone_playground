"""Compare realized returns of failed and successful frozen Racing policies."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.networks import reset_actor_memory
from drone_playground.simulation.policy import load_policy
from drone_playground.simulation.tasks import Event

root = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent / "racing-return-comparison"
output.mkdir(exist_ok=False)
config = OmegaConf.load(root / "results/racing_common_reward/shac_s0/config.yaml")
paths = [
    root / "results/racing_common_reward/ppo_s0/checkpoints/step-002900/policy.zip",
    root / "results/racing_common_reward/shac_s0/checkpoints/step-000700/policy.zip",
]
actor, _, _ = load_policy(paths[0])
env = create_environment(config, num_envs=8)
method = create_method(config)
initial = env.reset(jax.random.key(1_100_000))
initial_hash = hashlib.sha256()
for leaf in jax.tree.leaves(initial):
    if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        leaf = jax.random.key_data(leaf)
    initial_hash.update(np.asarray(leaf).tobytes())


@jax.jit
def chunk(carry, parameters):
    def step(carry, _):
        before, memory = carry
        raw, memory, _ = actor.apply(parameters, env.observe(before), memory)
        action = method.deterministic(raw)
        after = method.execute(env, before, action)
        cost, _ = env.task_reward_components(before, after, action)
        records = {
            "reward": -cost, "active": ~before.done, "event": after.task.event,
            "time": after.time, "gates": after.task.gates,
            "position": after.physics.states.pos[:, 0],
            "velocity": after.physics.states.vel[:, 0], "action": action,
        }
        return (after, reset_actor_memory(memory, after.done)), records

    return jax.lax.scan(step, carry, None, length=25)


results = []
try:
    for path in paths:
        loaded, parameters, metadata = load_policy(path)
        assert loaded.specification == actor.specification
        saved = metadata["config"]["experiment"]
        assert saved["task"] == OmegaConf.to_container(config.task, resolve=True)
        assert saved["method"] == OmegaConf.to_container(config.method, resolve=True)
        name = path.parents[2].name
        carry = (initial, actor.initialize_memory(env.num_envs))
        blocks = []
        for _ in range(int(np.ceil(env.task.duration / env.dt / 25)) + 1):
            carry, block = chunk(carry, parameters)
            blocks.append(jax.device_get(block))
            if bool(jnp.all(carry[0].done)):
                break
        assert bool(jnp.all(carry[0].done))
        data = {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}
        assert np.isfinite(data["reward"]).all()
        assert np.all(data["reward"][~data["active"]] == 0)
        target = output / f"{name}.npz"
        np.savez_compressed(target, **data)
        rows = []
        for world in range(env.num_envs):
            active = data["active"][:, world]
            reward = data["reward"][active, world]
            rows.append({
                "world": world, "event": Event(int(data["event"][active, world][-1])).name,
                "seconds": float(data["time"][active, world][-1]),
                "gates": int(data["gates"][active, world][-1]),
                "returns": {str(gamma): float(np.dot(gamma ** np.arange(len(reward)), reward))
                            for gamma in (0.99, 0.995, 0.999, 1.0)},
            })
        result = {"policy": str(path), "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "initial_state_sha256": initial_hash.hexdigest(), "seed": 1_100_000,
                  "raw_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "episodes": rows}
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / "report.json").write_text(json.dumps({
        "scope": "paired deterministic frozen-policy returns; diagnostic cases, not acceptance",
        "config": OmegaConf.to_container(config, resolve=True), "results": results,
    }, indent=2) + "\n")
finally:
    env.sim.close()
