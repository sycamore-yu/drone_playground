"""JAX port of D.VA's detached-observation short-horizon policy gradient.

The upstream D.VA implementation extends SHAC by detaching the visual
observation before the actor while retaining the differentiable
action -> controller -> dynamics -> reward path. P5 uses the same rule for
both D435 depth and MID360 range observations. The sensor encoder remains
fully trainable with respect to its parameters; only the observation's state
derivative is stopped.

The critic consumes the proprioceptive subset of the same observation
container. Target-critic weights are frozen during the actor update while the
terminal observation derivative is retained, matching the D.VA/SHAC bootstrap
semantics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import types
from brax.training.acme import running_statistics, specs
from flax import struct

from drone_playground.learning.algorithms.shac import (
    bootstrap_value,
    lambda_returns,
    segment_objective,
)
from drone_playground.learning.wrappers import wrap_for_training
from drone_playground.networks.factory import network_factory


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


def detached_policy_action(
    networks,
    normalizer,
    policy,
    observation,
    key,
    *,
    deterministic: bool = False,
):
    """Apply the actor with D.VA's detached-observation semantics."""
    detached = jax.lax.stop_gradient(observation)
    logits = networks.policy_network.apply(normalizer, policy, detached)
    distribution = networks.parametric_action_distribution
    if deterministic:
        return distribution.mode(logits)
    raw = distribution.sample_no_postprocessing(logits, key)
    return distribution.postprocess(raw)


def save_training_state(path, state, config):
    """Persist the DVA continuation tree through the shared store."""
    from drone_playground.artifacts.training_state import save_learner_state

    return save_learner_state(path, state, config, kind="dva-full-training-state")


def load_training_state(path):
    """Read DVA state; its trainer retains algorithm-specific restore checks."""
    from drone_playground.artifacts.training_state import load_learner_state

    return load_learner_state(path, kind="dva-full-training-state")


def _linear_schedule(initial: float, updates: int):
    """Match upstream D.VA's linear learning-rate decay toward 1e-5."""
    return optax.linear_schedule(
        init_value=initial,
        end_value=1e-5,
        transition_steps=max(int(updates), 1),
    )


def restore_contract(config: dict) -> dict:
    """Keep all training/environment semantics; exclude only run destinations."""
    config = json.loads(json.dumps(config))
    for key in (
        "resume",
        "max_wall_seconds",
        "num_evals",
        "publish_live",
        "actual_devices",
    ):
        config.pop(key, None)
    components = config.get("components", {})
    for key in (
        "run_id",
        "evaluation",
        "replay",
        "checkpoint",
        "source",
        "provenance",
    ):
        components.pop(key, None)
    for key in ("resume", "max_wall_seconds", "num_evals", "publish_live"):
        components.get("training", {}).pop(key, None)
    return config


def terminal_critic_observation(state, layout):
    """Use the physical terminal state even when the wrapper already reset."""
    proprio = state.info["terminal_proprioception"]
    if proprio.shape[-1] != layout.proprioception_size:
        raise ValueError("Terminal proprioception differs from the declared sensor layout")
    return jnp.pad(proprio, [(0, 0)] * (proprio.ndim - 1) + [(0, layout.sensor_size)])


def critic_observation_from_pipeline(environment, pipeline_state, layout):
    """Build the value input from physical state without traversing the sensor.

    D.VA explicitly detaches perception. Reusing the terminal observation for
    bootstrap would still trace the depth/LiDAR ray caster even though the
    critic ignores those channels. Some ray-intersection branches have
    undefined derivatives at topology changes, so a zero cotangent can still
    contaminate reverse mode with NaNs. The critic is declared proprio-only:
    construct exactly that differentiable subset from the physical state and
    append constant zeros for the unused sensor block.
    """
    data = pipeline_state
    states = data.sim_data.states
    pos = states.pos[:, 0, 0]
    quat = states.quat[:, 0, 0]
    vel = states.vel[:, 0, 0]
    ang_vel = states.ang_vel[:, 0, 0]
    goal = environment.bank.goal[data.scenario_id]
    proprio = jnp.concatenate([pos, quat, vel, ang_vel, goal - pos, data.previous_action], axis=-1)
    if proprio.shape[-1] != layout.proprioception_size:
        raise ValueError(
            f"critic proprioception has {proprio.shape[-1]} fields; "
            f"layout declares {layout.proprioception_size}"
        )
    sensor = jnp.zeros((*proprio.shape[:-1], layout.sensor_size), dtype=proprio.dtype)
    return jnp.concatenate([proprio, sensor], axis=-1)


def train(
    environment,
    config: dict,
    policy_params_fn=lambda *_: None,
    progress_fn=lambda *_: None,
    state_directory: Path | None = None,
    restore_state: Path | None = None,
):
    """Train one detached-perception D.VA unit."""
    count = int(config["num_envs"])
    horizon = int(config["horizon_length"])
    updates = int(config["policy_updates"])
    if min(count, horizon, updates) < 1:
        raise ValueError("D.VA requires positive environment, horizon and update counts")
    if not config.get("sensor_layout"):
        raise ValueError("D.VA requires a declared P5 perception sensor layout")

    gamma = float(config.get("discounting", 0.99))
    lam = float(config.get("td_lambda", 0.95))
    alpha = float(config.get("target_critic_alpha", 0.2))
    value_iterations = int(config.get("critic_updates", 16))
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError("D.VA discounting and td_lambda must be in [0,1]")
    if not 0.0 <= alpha <= 1.0 or value_iterations < 1:
        raise ValueError("Invalid D.VA target critic averaging or update count")

    preprocess = (
        running_statistics.normalize
        if config.get("normalize_observations", False)
        else types.identity_observation_preprocessor
    )
    networks = network_factory(config)(
        (environment.observation_size,),
        environment.action_size,
        preprocess_observations_fn=preprocess,
    )
    from drone_playground.networks.perception import SensorLayout

    layout = SensorLayout.from_dict(config["sensor_layout"])
    from brax.training.agents.ppo import networks as ppo_networks

    make_policy = ppo_networks.make_inference_fn(networks)
    env = wrap_for_training(environment, environment.episode_length)

    actor_lr = _linear_schedule(float(config.get("learning_rate", 0.002)), updates)
    critic_lr = _linear_schedule(float(config.get("critic_learning_rate", 0.0002)), updates)
    beta1, beta2 = tuple(config.get("betas", (0.7, 0.95)))
    actor_opt = optax.chain(
        optax.clip_by_global_norm(float(config.get("max_grad_norm", 1.0))),
        optax.adam(actor_lr, b1=float(beta1), b2=float(beta2)),
    )
    critic_opt = optax.chain(
        optax.clip_by_global_norm(float(config.get("critic_max_grad_norm", 10.0))),
        optax.adam(
            lambda step: critic_lr(step // value_iterations),
            b1=float(beta1),
            b2=float(beta2),
        ),
    )

    key, actor_key, critic_key, env_key = jax.random.split(
        jax.random.PRNGKey(int(config.get("seed", 0))), 4
    )
    policy = networks.policy_network.init(actor_key)
    critic = networks.value_network.init(critic_key)
    state = TrainingState(
        policy=policy,
        critic=critic,
        target_critic=critic,
        actor_optimizer=actor_opt.init(policy),
        critic_optimizer=critic_opt.init(critic),
        normalizer=running_statistics.init_state(
            specs.Array((environment.observation_size,), jnp.float32)
        ),
        environment=jax.jit(env.reset)(jax.random.split(env_key, count)),
        key=key,
        updates=jnp.int32(0),
    )

    if restore_state is not None:
        restored, meta = load_training_state(restore_state)
        previous, current = restore_contract(meta["config"]), restore_contract(config)
        for name in sorted(previous.keys() | current.keys()):
            if previous.get(name) != current.get(name):
                raise ValueError(f"D.VA restore configuration differs: {name}")
        if int(restored.updates) >= updates:
            raise ValueError("D.VA checkpoint has already completed the declared budget")
        state = restored

    initial_policy = jax.tree.map(np.asarray, state.policy)
    initial_critic = jax.tree.map(np.asarray, state.critic)

    def value_fn(normalizer, weights, observation):
        return networks.value_network.apply(normalizer, weights, observation)

    def objective(params, normalizer, target, start, key):
        def one_step(carry, _):
            current, key = carry
            key, sample = jax.random.split(key)
            action = detached_policy_action(
                networks,
                normalizer,
                params,
                current.obs,
                sample,
                deterministic=False,
            )
            nxt = env.step(current, action)
            terminal = nxt.info["terminated"].astype(bool)
            critic_observation = terminal_critic_observation(nxt, layout)
            value = bootstrap_value(
                lambda w, obs: value_fn(normalizer, w, obs),
                target,
                critic_observation,
                terminal,
            )
            row = (
                jax.lax.stop_gradient(current.obs),
                nxt.reward,
                value,
                nxt.done.astype(bool),
                terminal,
            )
            return (nxt, key), row

        (end, key), rows = jax.lax.scan(one_step, (start, key), None, length=horizon)
        observations, rewards, values, done, terminal = rows
        loss = segment_objective(rewards, values, done, gamma)
        return loss, (end, key, observations, rewards, values, done, terminal)

    @jax.jit
    def update(state):
        start = jax.tree.map(jax.lax.stop_gradient, state.environment)
        (actor_loss, aux), actor_grad = jax.value_and_grad(objective, has_aux=True)(
            state.policy, state.normalizer, state.target_critic, start, state.key
        )
        end, key, observations, rewards, values, done, terminal = jax.tree.map(
            jax.lax.stop_gradient, aux
        )
        targets = lambda_returns(rewards, values, done, terminal, gamma, lam)
        actor_delta, actor_os = actor_opt.update(actor_grad, state.actor_optimizer, state.policy)
        policy = optax.apply_updates(state.policy, actor_delta)

        flat_obs = observations.reshape((-1, environment.observation_size))
        flat_target = targets.reshape(-1)

        def fit(carry, _):
            weights, optimizer_state = carry

            def critic_loss(p):
                prediction = networks.value_network.apply(state.normalizer, p, flat_obs)
                return jnp.mean(jnp.square(prediction - flat_target))

            loss, grad = jax.value_and_grad(critic_loss)(weights)
            delta, optimizer_state = critic_opt.update(grad, optimizer_state, weights)
            return (optax.apply_updates(weights, delta), optimizer_state), (
                loss,
                optax.global_norm(grad),
            )

        (critic, critic_os), (critic_losses, critic_grad_norms) = jax.lax.scan(
            fit,
            (state.critic, state.critic_optimizer),
            None,
            length=value_iterations,
        )
        target = jax.tree.map(
            lambda old, new: alpha * old + (1.0 - alpha) * new,
            state.target_critic,
            critic,
        )
        normalizer = running_statistics.update(state.normalizer, observations)
        return state.replace(
            policy=policy,
            critic=critic,
            target_critic=target,
            actor_optimizer=actor_os,
            critic_optimizer=critic_os,
            normalizer=normalizer,
            environment=end,
            key=key,
            updates=state.updates + 1,
        ), {
            "training/actor_loss": actor_loss,
            "training/critic_loss": critic_losses[-1],
            "training/actor_grad_norm": optax.global_norm(actor_grad),
            "training/critic_grad_norm": critic_grad_norms[-1],
            "training/reward": jnp.mean(rewards),
            "training/bootstrap_value": jnp.mean(values),
            "training/terminated_fraction": jnp.mean(terminal),
            "training/value_target_mean": jnp.mean(targets),
        }

    start_clock = time.monotonic()
    initial_updates = int(state.updates)
    initial_step = initial_updates * count * horizon
    policy_params_fn(initial_step, make_policy, (state.normalizer, state.policy))
    milestones = set(
        int(x) for x in np.linspace(0, updates, max(int(config.get("num_evals", 9)), 2))
    )
    total_update_seconds = 0.0
    compile_seconds = 0.0
    metrics = {}
    for iteration in range(initial_updates + 1, updates + 1):
        tic = time.monotonic()
        state, raw_metrics = update(state)
        metrics = {key: float(value) for key, value in raw_metrics.items()}
        duration = time.monotonic() - tic
        if iteration == initial_updates + 1:
            compile_seconds = duration
        else:
            total_update_seconds += duration
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Non-finite D.VA update {iteration}: {metrics}")
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
        if time.monotonic() - start_clock > float(config.get("max_wall_seconds", 3600)):
            if state_directory is not None:
                save_training_state(
                    Path(state_directory) / "budget-exhausted.pkl",
                    state,
                    config,
                )
            raise TimeoutError("D.VA wall-clock budget reached")

    def delta(old, new):
        return float(
            np.sqrt(
                sum(
                    float(np.square(np.asarray(a) - np.asarray(b)).sum())
                    for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True)
                )
            )
        )

    return (
        make_policy,
        (state.normalizer, state.policy),
        {
            **metrics,
            "actual_steps": int(state.updates) * count * horizon,
            "actor_parameter_delta_l2": delta(initial_policy, state.policy),
            "critic_parameter_delta_l2": delta(initial_critic, state.critic),
            "compile_and_first_update_seconds": compile_seconds,
            "net_update_seconds": total_update_seconds,
            "updates": int(state.updates),
            "method": "D.VA-JAX detached-observation port",
            "sensor_gradient": "stopped at actor observation",
        },
    )
