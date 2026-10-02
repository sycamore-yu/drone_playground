"""Finite-horizon BPTT using the Brax APG policy and trajectory-gradient mechanism.

This update differentiates the mean rollout reward, with no value bootstrap.
It adds explicit per-window snapshots and complete continuation state to the
shared task interface. It is identified as BPTT, not a new policy-gradient method.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.io import model
from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.agents.apg.networks import make_inference_fn
from flax import struct

from drone_playground.learning.wrappers import wrap_for_training
from drone_playground.networks.factory import network_factory


@struct.dataclass
class TrainingState:
    policy: object
    optimizer: object
    normalizer: object
    environment: object
    key: object
    updates: object


def save_state(path, state, config):
    path = Path(path)
    typed = []

    def host(p, x):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            typed.append(jax.tree_util.keystr(p))
            return np.asarray(jax.random.key_data(x))
        return np.asarray(x)

    payload = jax.tree_util.tree_map_with_path(host, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    model.save_params(str(temporary), payload)
    temporary.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                config=config,
                typed_key_paths=typed,
                updates=int(state.updates),
                kind="bptt-full-training-state",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


def load_state(path, config):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("BPTT state digest mismatch")
    ignored = {
        "num_evals",
        "max_wall_seconds",
        "resume",
        "components",
        "actual_devices",
    }
    for field, value in meta["config"].items():
        if field not in ignored and config.get(field) != value:
            raise ValueError(f"BPTT continuation differs on {field}")
    if "components" in config:
        old, new = meta["config"]["components"], config["components"]
        for group in (
            "env",
            "method",
            "algorithm",
            "network",
            "objective",
            "runtime",
        ):
            if old[group] != new[group]:
                raise ValueError(
                    f"BPTT continuation differs on component {group}"
                )
    paths = set(meta["typed_key_paths"])
    return jax.tree_util.tree_map_with_path(
        lambda p, x: (
            jax.random.wrap_key_data(jnp.asarray(x))
            if jax.tree_util.keystr(p) in paths
            else jnp.asarray(x)
        ),
        model.load_params(str(path)),
    )


def train(
    environment,
    config,
    policy_params_fn=lambda *_: None,
    progress_fn=lambda *_: None,
    state_directory=None,
    restore_state=None,
):
    count, horizon, updates = (
        int(config[k]) for k in ("num_envs", "horizon_length", "policy_updates")
    )
    if min(count, horizon, updates) < 1:
        raise ValueError("BPTT counts must be positive")
    preprocess = (
        running_statistics.normalize
        if config.get("normalize_observations")
        else types.identity_observation_preprocessor
    )
    net = network_factory(config)(
        environment.observation_size,
        environment.action_size,
        preprocess_observations_fn=preprocess,
    )
    make_policy = make_inference_fn(net)
    lr = config.get("learning_rate", 0.005)
    if config.get("use_schedule", True):
        lr = optax.exponential_decay(lr, 1, config.get("schedule_decay", 0.997))
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.get("max_grad_norm", 1.0)),
        optax.clip(1.0),
        optax.adam(lr, b1=0.7, b2=0.95),
    )
    env = wrap_for_training(environment, environment.episode_length)
    key, pk, ek = jax.random.split(jax.random.PRNGKey(config.get("seed", 0)), 3)
    params = net.policy_network.init(pk)
    normalizer = running_statistics.init_state(
        specs.Array((environment.observation_size,), jnp.float32)
    )
    if config.get("warm_start"):
        from drone_playground.artifacts.checkpoints import load_policy

        _, previous, meta = load_policy(config["warm_start"])
        if meta["config"]["algorithm"]["name"] not in ("apg", "bptt", "shac"):
            raise ValueError(
                "BPTT warm start requires the same APG-family policy parameterization"
            )
        if meta["observation_size"] != environment.observation_size:
            raise ValueError("Warm-start observation contract differs")
        normalizer, params = jax.tree.map(jnp.asarray, previous)
        net.policy_network.apply(
            normalizer, params, jnp.zeros((environment.observation_size,))
        )
    state = TrainingState(
        params,
        optimizer.init(params),
        normalizer,
        jax.jit(env.reset)(jax.random.split(ek, count)),
        key,
        jnp.int32(0),
    )
    if restore_state:
        state = load_state(restore_state, config)
    initial = jax.tree.map(np.asarray, state.policy)

    def objective(params, normalizer, current, key):
        policy = make_policy((normalizer, params))

        def step(carry, _):
            current, key = carry
            key, sample = jax.random.split(key)
            action = policy(current.obs, sample)[0]
            nxt = env.step(current, action)
            return (nxt, key), (nxt.reward, current.obs)

        (end, key), (rewards, obs) = jax.lax.scan(
            step, (current, key), None, length=horizon
        )
        return -jnp.mean(rewards), (end, key, obs)

    @jax.jit
    def update(state):
        start = jax.tree.map(jax.lax.stop_gradient, state.environment)
        key = state.key
        if config.get("resample_window_initials", False):
            key, reset_key = jax.random.split(key)
            start = env.reset(jax.random.split(reset_key, count))
        (loss, (end, key, obs)), grad = jax.value_and_grad(
            objective, has_aux=True
        )(
            state.policy,
            state.normalizer,
            start,
            key,
        )
        change, os = optimizer.update(grad, state.optimizer, state.policy)
        return state.replace(
            policy=optax.apply_updates(state.policy, change),
            optimizer=os,
            normalizer=running_statistics.update(state.normalizer, obs),
            environment=jax.tree.map(jax.lax.stop_gradient, end),
            key=key,
            updates=state.updates + 1,
        ), {
            "training/actor_loss": loss,
            "training/actor_grad_norm": optax.global_norm(grad),
        }

    milestones = {
        int(v)
        for v in np.linspace(0, updates, max(config.get("num_evals", 9), 2))
    }
    start, net_seconds, compile_seconds = time.monotonic(), 0.0, 0.0
    policy_params_fn(
        int(state.updates) * count * horizon,
        make_policy,
        (state.normalizer, state.policy),
    )
    metrics = {}
    for iteration in range(int(state.updates) + 1, updates + 1):
        tic = time.monotonic()
        state, raw = update(state)
        metrics = {key: float(value) for key, value in raw.items()}
        elapsed = time.monotonic() - tic
        if iteration == int(np.asarray(state.updates)) and compile_seconds == 0:
            compile_seconds = elapsed
        else:
            net_seconds += elapsed
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(
                f"Non-finite BPTT update {iteration}: {metrics}"
            )
        if iteration % 10 == 0 or iteration in milestones:
            progress_fn(
                iteration * count * horizon,
                {
                    **metrics,
                    "training/updates": iteration,
                    "training/net_update_seconds": net_seconds,
                    "training/compile_and_first_update_s": compile_seconds,
                },
            )
        if iteration in milestones:
            if state_directory:
                save_state(
                    Path(state_directory) / f"update-{iteration:07d}.pkl",
                    state,
                    config,
                )
            policy_params_fn(
                iteration * count * horizon,
                make_policy,
                (state.normalizer, state.policy),
            )
        if time.monotonic() - start > config.get("max_wall_seconds", 3600):
            if state_directory:
                save_state(
                    Path(state_directory) / "budget-exhausted.pkl",
                    state,
                    config,
                )
            raise TimeoutError("BPTT wall-clock budget reached")
    delta = sum(
        np.square(np.asarray(a) - np.asarray(b)).sum()
        for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(state.policy))
    )
    return (
        make_policy,
        (state.normalizer, state.policy),
        {
            **metrics,
            "actual_steps": int(state.updates) * count * horizon,
            "actor_parameter_delta_l2": float(np.sqrt(delta)),
            "compile_and_first_update_seconds": compile_seconds,
            "net_update_seconds": net_seconds,
            "window_resets": int(state.updates) * count
            if config.get("resample_window_initials", False)
            else 0,
        },
    )
