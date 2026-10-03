"""Short-horizon actor-critic using Crazyflow derivatives and Brax networks.

The objective follows SHAC (Xu et al., 2022): differentiate rewards and terminal
value with respect to actions through a short physical rollout. Target-critic
weights stay fixed during the actor update, while its input derivative remains
in the graph. This is an independent JAX implementation, not copied Torch code.
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
from brax.training import networks, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.apg import networks as apg_networks
from flax import linen, struct

from drone_playground.learning.wrappers import wrap_for_training


def bootstrap_value(value_fn, weights, observation, terminated):
    """Evaluate a frozen critic, retaining only the observation derivative."""
    weights = jax.tree.map(jax.lax.stop_gradient, weights)
    safe_obs = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.where(terminated, 0.0, value_fn(weights, safe_obs))


def lambda_returns(rewards, next_values, done, terminated, discount, lam):
    """Finite-window lambda returns; timeouts bootstrap before the fresh reset."""
    boot = jnp.where(terminated, 0.0, next_values)

    def backward(carry, row):
        reward, value, boundary = row
        continuation = jnp.where(boundary, value, (1 - lam) * value + lam * carry)
        target = reward + discount * continuation
        return target, target

    _, target = jax.lax.scan(backward, boot[-1], (rewards, boot, done), reverse=True)
    return jax.lax.stop_gradient(target)


def segment_objective(rewards, bootstrap, done, discount):
    """Sum each episode fragment once, including the end of the current window."""
    boundary = done.at[-1].set(True)

    def forward(weight, row):
        reward, value, ended, close = row
        result = weight * reward + jnp.where(close, weight * discount * value, 0.0)
        weight = jnp.where(ended, 1.0, weight * discount)
        return weight, result

    _, values = jax.lax.scan(
        forward, jnp.ones_like(rewards[0]), (rewards, bootstrap, done, boundary)
    )
    return -jnp.mean(values)


@struct.dataclass
class TrainingState:
    policy: object
    critic: object
    target_critic: object
    actor_optimizer: object
    critic_optimizer: object
    normalizer: object
    environment: object
    key: jax.Array
    updates: jax.Array


def save_training_state(path: Path, state: TrainingState, config: dict) -> None:
    """Store all continuation state and identify typed PRNG leaves explicitly."""
    typed_paths = []

    def to_host(keypath, leaf):
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            typed_paths.append(jax.tree_util.keystr(keypath))
            return np.asarray(jax.random.key_data(leaf))
        return np.asarray(leaf)

    host = jax.tree_util.tree_map_with_path(to_host, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    model.save_params(str(temporary), host)
    temporary.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "shac-full-training-state",
                "config": config,
                "typed_key_paths": typed_paths,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "updates": int(state.updates),
            },
            indent=2,
        )
        + "\n"
    )


def load_training_state(path: Path) -> tuple[TrainingState, dict]:
    """Load a trusted local continuation snapshot and restore typed RNG keys."""
    meta = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("SHAC training-state digest mismatch")
    keypaths = set(meta["typed_key_paths"])
    state = model.load_params(str(path))
    state = jax.tree_util.tree_map_with_path(
        lambda p, x: (
            jax.random.wrap_key_data(jnp.asarray(x))
            if jax.tree_util.keystr(p) in keypaths
            else jnp.asarray(x)
        ),
        state,
    )
    return state, meta


def train(
    environment,
    config: dict,
    policy_params_fn=lambda *_: None,
    progress_fn=lambda *_: None,
    state_directory: Path | None = None,
    restore_state: Path | None = None,
):
    """Run the declared SHAC update budget with periodic external evaluation."""
    count = int(config["num_envs"])
    horizon = int(config["horizon_length"])
    updates = int(config["policy_updates"])
    if min(count, horizon, updates) < 1:
        raise ValueError("SHAC requires positive environment, horizon and update counts")
    gamma, lam = config.get("discounting", 0.99), config.get("td_lambda", 0.95)
    if not (0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError("discounting and td_lambda must be in [0,1]")
    alpha = config.get("target_critic_alpha", 0.4)
    value_iterations = int(config.get("critic_updates", 8))
    if not 0 <= alpha <= 1 or value_iterations < 1:
        raise ValueError("Invalid target critic averaging or update count")
    preprocess = (
        running_statistics.normalize
        if config.get("normalize_observations", False)
        else types.identity_observation_preprocessor
    )
    sizes = tuple(config.get("hidden_sizes", [64, 64]))
    from drone_playground.networks.factory import network_factory

    actor = network_factory({**config, "algorithm": config.get("algorithm", "shac")})(
        environment.observation_size,
        environment.action_size,
        preprocess_observations_fn=preprocess,
    )
    critic = (
        actor.value_network
        if config.get("sensor_layout")
        else networks.make_value_network(
            environment.observation_size,
            preprocess_observations_fn=preprocess,
            hidden_layer_sizes=sizes,
            activation=linen.elu,
        )
    )
    make_policy = apg_networks.make_inference_fn(actor)
    env = wrap_for_training(environment, environment.episode_length)
    lr = config.get("learning_rate", 0.003)
    if config.get("use_schedule", True):
        lr = optax.exponential_decay(lr, 1, config.get("schedule_decay", 0.997))
    actor_opt = optax.chain(
        optax.clip_by_global_norm(config.get("max_grad_norm", 1.0)),
        optax.adam(
            lr,
            b1=config.get("actor_adam_b1", 0.7),
            b2=config.get("actor_adam_b2", 0.95),
        ),
    )
    critic_opt = optax.chain(
        optax.clip_by_global_norm(config.get("critic_max_grad_norm", 10.0)),
        optax.adam(config.get("critic_learning_rate", 0.002)),
    )
    key, ak, ck, ek = jax.random.split(jax.random.PRNGKey(config.get("seed", 0)), 4)
    policy, value = actor.policy_network.init(ak), critic.init(ck)
    normalizer = running_statistics.init_state(
        specs.Array((environment.observation_size,), jnp.float32)
    )
    if config.get("warm_start"):
        from drone_playground.artifacts.checkpoints import load_policy
        from drone_playground.learning.brax_configuration import native_training_config

        if restore_state is not None:
            raise ValueError("Select either actor warm start or full-state continuation")
        _, previous, metadata = load_policy(config["warm_start"])
        source = native_training_config(metadata["config"])
        if source["algorithm"] not in ("apg", "bptt", "shac"):
            raise ValueError("SHAC warm start requires an APG-family actor")
        if (
            metadata["observation_size"] != environment.observation_size
            or metadata["action_size"] != environment.action_size
        ):
            raise ValueError("Warm-start observation/action dimensions differ")
        for field in (
            "task",
            "dynamics",
            "drone",
            "hidden_sizes",
            "layer_norm",
            "normalize_observations",
            "sensor_layout",
        ):
            if source.get(field) != config.get(field):
                raise ValueError(f"SHAC warm-start configuration differs: {field}")
        if config.get("components"):
            old, new = metadata["config"]["env"], config["components"]["env"]
            if (
                old["sensor"] != new["sensor"]
                or old["task"]["observation"] != new["task"]["observation"]
            ):
                raise ValueError("SHAC warm-start input semantics differ")
        normalizer, policy = jax.tree.map(jnp.asarray, previous)
        actor.policy_network.apply(normalizer, policy, jnp.zeros(environment.observation_size))
    state = TrainingState(
        policy,
        value,
        value,
        actor_opt.init(policy),
        critic_opt.init(value),
        normalizer,
        jax.jit(env.reset)(jax.random.split(ek, count)),
        key,
        jnp.int32(0),
    )
    if restore_state is not None:
        restored, meta = load_training_state(restore_state)
        for name in (
            "task",
            "dynamics",
            "drone",
            "num_envs",
            "horizon_length",
            "hidden_sizes",
            "normalize_observations",
            "learning_rate",
            "critic_learning_rate",
            "resample_window_initials",
        ):
            if meta["config"].get(name) != config.get(name):
                raise ValueError(f"Restore configuration differs: {name}")
        state = restored
    initial_policy = jax.tree.map(np.asarray, state.policy)
    initial_critic = jax.tree.map(np.asarray, state.critic)

    def objective(params, normalizer, target, start, key):
        policy_fn = make_policy((normalizer, params))

        def value_fn(w, obs):
            return critic.apply(normalizer, w, obs)

        def step(carry, _):
            current, key = carry
            key, sample = jax.random.split(key)
            action = policy_fn(current.obs, sample)[0]
            nxt = env.step(current, action)
            terminal = nxt.info["terminated"].astype(bool)
            value = bootstrap_value(value_fn, target, nxt.info["terminal_observation"], terminal)
            row = (
                current.obs,
                nxt.reward,
                value,
                nxt.done.astype(bool),
                terminal,
                nxt.info["terminal_observation"],
            )
            return (nxt, key), row

        (end, key), rows = jax.lax.scan(step, (start, key), None, length=horizon)
        obs, rewards, values, done, terminal, last_obs = rows
        loss = segment_objective(rewards, values, done, gamma)
        return loss, (end, key, obs, rewards, values, done, terminal, last_obs)

    @jax.jit
    def update(state):
        # Each update is a fresh differentiation window, preserving actual state.
        start = jax.tree.map(jax.lax.stop_gradient, state.environment)
        key = state.key
        if config.get("resample_window_initials", False):
            key, reset_key = jax.random.split(key)
            start = env.reset(jax.random.split(reset_key, count))
        (loss, aux), grad = jax.value_and_grad(objective, has_aux=True)(
            state.policy, state.normalizer, state.target_critic, start, key
        )
        end, key, obs, rewards, values, done, terminal, last_obs = jax.tree.map(
            jax.lax.stop_gradient, aux
        )
        targets = lambda_returns(rewards, values, done, terminal, gamma, lam)
        actor_change, actor_os = actor_opt.update(grad, state.actor_optimizer, state.policy)
        policy = optax.apply_updates(state.policy, actor_change)
        flat_obs = obs.reshape((-1, environment.observation_size))
        flat_target = targets.reshape(-1)

        def fit(carry, _):
            weights, os = carry

            def value_loss(p):
                prediction = critic.apply(state.normalizer, p, flat_obs)
                return jnp.mean(jnp.square(prediction - flat_target))

            vl, vg = jax.value_and_grad(value_loss)(weights)
            delta, os = critic_opt.update(vg, os, weights)
            return (optax.apply_updates(weights, delta), os), (
                vl,
                optax.global_norm(vg),
            )

        (value, critic_os), (vl, vg) = jax.lax.scan(
            fit,
            (state.critic, state.critic_optimizer),
            None,
            length=value_iterations,
        )
        target = jax.tree.map(
            lambda old, new: alpha * old + (1 - alpha) * new,
            state.target_critic,
            value,
        )
        normalizer = running_statistics.update(state.normalizer, obs)
        return state.replace(
            policy=policy,
            critic=value,
            target_critic=target,
            actor_optimizer=actor_os,
            critic_optimizer=critic_os,
            normalizer=normalizer,
            environment=end,
            key=key,
            updates=state.updates + 1,
        ), {
            "training/actor_loss": loss,
            "training/critic_loss": vl[-1],
            "training/actor_grad_norm": optax.global_norm(grad),
            "training/critic_grad_norm": vg[-1],
            "training/reward": jnp.mean(rewards),
            "training/bootstrap_value": jnp.mean(values),
            "training/terminated_fraction": jnp.mean(terminal),
            "training/value_target_mean": jnp.mean(targets),
        }

    start_clock = time.monotonic()
    initial_step = int(state.updates) * count * horizon
    policy_params_fn(initial_step, make_policy, (state.normalizer, state.policy))
    milestones = set(int(x) for x in np.linspace(0, updates, max(config.get("num_evals", 9), 2)))
    total_update_seconds = 0.0
    compile_seconds = 0.0
    metrics = {}
    for iteration in range(int(state.updates) + 1, updates + 1):
        tic = time.monotonic()
        state, raw_metrics = update(state)
        # Synchronize before reporting actual work and timings.
        metrics = {k: float(v) for k, v in raw_metrics.items()}
        duration = time.monotonic() - tic
        if iteration == 1:
            compile_seconds = duration
        else:
            total_update_seconds += duration
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"Non-finite SHAC update {iteration}: {metrics}")
        step = iteration * count * horizon
        if iteration in milestones or iteration % 10 == 0:
            progress_fn(
                step,
                {
                    **metrics,
                    "training/updates": iteration,
                    "training/compile_and_first_update_s": compile_seconds,
                    "training/net_update_seconds": total_update_seconds,
                },
            )
        if iteration in milestones:
            if state_directory is not None:
                save_training_state(
                    Path(state_directory) / f"update-{iteration:07d}.pkl",
                    state,
                    config,
                )
            policy_params_fn(step, make_policy, (state.normalizer, state.policy))
        if time.monotonic() - start_clock > config.get("max_wall_seconds", 3600):
            if state_directory is not None:
                save_training_state(
                    Path(state_directory) / "budget-exhausted.pkl",
                    state,
                    config,
                )
            raise TimeoutError("SHAC wall-clock budget reached")

    def delta(old, new):
        return float(
            np.sqrt(
                sum(
                    float(np.square(np.asarray(a) - np.asarray(b)).sum())
                    for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new))
                )
            )
        )

    result = {
        **metrics,
        "actual_steps": int(state.updates) * count * horizon,
        "actor_parameter_delta_l2": delta(initial_policy, state.policy),
        "critic_parameter_delta_l2": delta(initial_critic, state.critic),
        "compile_and_first_update_seconds": compile_seconds,
        "net_update_seconds": total_update_seconds,
        "updates": int(state.updates),
        "window_resets": int(state.updates) * count
        if config.get("resample_window_initials", False)
        else 0,
    }
    return make_policy, (state.normalizer, state.policy), result
