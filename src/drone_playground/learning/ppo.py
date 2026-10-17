"""Stochastic tanh-Gaussian PPO with complete recurrent-sequence recomputation."""

import math

import jax
import jax.numpy as jnp
import optax
from flax import struct

from drone_playground.simulation.networks import reset_actor_memory
from drone_playground.simulation.tasks import Event


def shared_critic_parameters(actor_parameters, critic_head):
    """Bind VisFly's value head to the one actor-owned visual/state encoder."""
    scope = dict(critic_head["params"]["visfly_value"])
    scope["VisFlyFeatures_0"] = actor_parameters["params"]["visfly_policy"]["VisFlyFeatures_0"]
    return {"params": {"visfly_value": scope}}


def policy_variables(state, *, shared_visual_encoder=False):
    """Return each parameter once for the joint actor/value optimizer."""
    variables = {"params": state.params, "log_std": state.log_std}
    if shared_visual_encoder:
        head = {
            key: value
            for key, value in state.critic_params["params"]["visfly_value"].items()
            if key != "VisFlyFeatures_0"
        }
        variables["critic_head"] = {"params": {"visfly_value": head}}
    return variables


@struct.dataclass
class Rollout:
    """Carry sampled transitions and recurrent memory for PPO updates."""

    observations: dict
    pre_tanh: object
    log_prob: jax.Array
    rewards: jax.Array
    values: jax.Array
    next_values: jax.Array
    terminated: jax.Array
    done: jax.Array


def generalized_advantage_estimate(
    rewards, values, next_values, terminated, done, *, gamma=0.99, gae_lambda=0.95
):
    """Return GAE and value targets for [T,B] arrays.

    next_values must use final observations BEFORE resetting environments.
    A true termination suppresses bootstrap; either kind of done stops the trace.
    The rollout boundary bootstraps normally and starts with a zero future trace.
    """
    deltas = rewards + gamma * jnp.where(terminated, 0, next_values) - values

    def backward(advantage, inputs):
        delta, reset = inputs
        advantage = delta + gamma * gae_lambda * jnp.where(reset, 0, advantage)
        return advantage, advantage

    _, advantages = jax.lax.scan(backward, jnp.zeros_like(values[0]), (deltas, done), reverse=True)
    return advantages, advantages + values


def normalize_advantages(advantages):
    """Population standard deviation normalization, including constant batches."""
    return (advantages - advantages.mean()) / jnp.maximum(advantages.std(), 1e-8)


def exceeds_kl_limit(approx_kl, target_kl):
    """Stop further actor minibatch updates when KL exceeds 1.5 times target."""
    return jnp.logical_and(target_kl > 0, approx_kl > 1.5 * target_kl)


def advantage_progress_correlation(advantages, progress):
    """Measure whether better-than-expected actions coincide with real progress.

    Only training rollout transitions are used; frozen benchmark outcomes are
    deliberately not fed into this score. A zero-variance batch reports zero
    instead of producing an undefined correlation.
    """
    x, y = advantages - advantages.mean(), progress - progress.mean()
    covariance = jnp.mean(x * y)
    return jnp.where(
        (jnp.mean(x**2) > 1e-12) & (jnp.mean(y**2) > 1e-12),
        covariance / jnp.maximum(x.std() * y.std(), 1e-8),
        0.0,
    )


def completed_navigation_metrics(done, terms):
    """Measure actual completed-episode outcomes instead of per-step event density."""
    completed = jnp.sum(done)
    denominator = jnp.maximum(completed, 1)
    results = {
        "navigation_completed_episodes": completed.astype(jnp.float32),
        "navigation_completed_success_rate": (
            jnp.sum(terms["navigation_success_fraction"]) / denominator
        ),
        "navigation_completed_collision_rate": (
            jnp.sum(terms["navigation_collision_fraction"]) / denominator
        ),
        "navigation_completed_out_of_bounds_rate": (
            jnp.sum(terms["navigation_out_of_bounds_fraction"]) / denominator
        ),
    }
    if "navigation_timeout_fraction" in terms:
        results["navigation_completed_timeout_rate"] = (
            jnp.sum(terms["navigation_timeout_fraction"]) / denominator
        )
        results["navigation_completed_numerical_failure_rate"] = (
            jnp.sum(terms["navigation_numerical_failure_fraction"]) / denominator
        )
        successes = jnp.sum(terms["navigation_success_fraction"])
        results["navigation_completed_successes"] = successes
        results["navigation_success_mean_flight_seconds"] = jnp.sum(
            terms["navigation_success_flight_seconds"]
        ) / jnp.maximum(successes, 1)
        results["navigation_completed_min_clearance_mean_metres"] = (
            jnp.sum(terms["navigation_episode_min_clearance_metres"]) / denominator
        )
    return results


def collect_rollout(trainer, state, *, sampling_progress=None):
    """Collect stochastic transitions and retain pre-reset critic observations."""

    def step(carry, _):
        env_state, memory, history, rng, _ = carry
        rng, action_key, reset_key = jax.random.split(rng, 3)
        observation = trainer._observe(env_state)
        raw, memory, _ = trainer.actor.apply(state.params, observation, memory)
        action, pre_tanh, log_prob = trainer.method.sample(action_key, raw, state.log_std)
        after = trainer._step(env_state, action, differentiable=False)
        cost, history, terms = trainer.objective.cost(env_state, after, action, history)
        # A single sampled world is enough for an opt-in Dashboard preview.
        # These values are observational, never part of the loss or checkpoint.
        terms["_preview_position"] = after.physics.states.pos[0, 0]
        terms["_preview_quaternion"] = after.physics.states.quat[0, 0]
        terms["_preview_time"] = after.time[0]
        terms["episode_success"] = ((after.task.event == Event.SUCCESS) & ~env_state.done).astype(
            jnp.float32
        )
        if trainer.env.task.name == "navigation":
            before_position = env_state.physics.states.pos[:, 0]
            after_position = after.physics.states.pos[:, 0]
            goal = trainer.env.goal_for(env_state)
            distances = jnp.linalg.norm(goal - before_position, axis=-1)
            remaining = jnp.linalg.norm(goal - after_position, axis=-1)
            progress = jnp.where(env_state.done, 0.0, distances - remaining)
            terms = {
                **terms,
                "navigation_progress_metres": progress,
                "navigation_remaining_metres": remaining,
                "navigation_collision_fraction": (
                    (after.task.event == Event.COLLISION) & ~env_state.done
                ).astype(jnp.float32),
                "navigation_out_of_bounds_fraction": (
                    (after.task.event == Event.OUT_OF_BOUNDS) & ~env_state.done
                ).astype(jnp.float32),
                "navigation_success_fraction": (
                    (after.task.event == Event.SUCCESS) & ~env_state.done
                ).astype(jnp.float32),
                "navigation_timeout_fraction": (
                    (after.task.event == Event.TIMEOUT) & ~env_state.done
                ).astype(jnp.float32),
                "navigation_numerical_failure_fraction": (
                    (after.task.event == Event.NUMERICAL_FAILURE) & ~env_state.done
                ).astype(jnp.float32),
                "navigation_success_flight_seconds": jnp.where(
                    (after.task.event == Event.SUCCESS) & ~env_state.done,
                    after.time,
                    0.0,
                ),
                "navigation_episode_min_clearance_metres": jnp.where(
                    after.done & ~env_state.done,
                    after.task.min_clearance,
                    0.0,
                ),
            }
        value = trainer.critic.apply(state.critic_params, observation)
        next_value = trainer.critic.apply(state.critic_params, trainer._observe(after))
        data = Rollout(
            observation, pre_tanh, log_prob, -cost, value, next_value, after.terminated, after.done
        )
        env_state, memory, history = trainer._reset(
            reset_key, after, memory, history, sampling_progress=sampling_progress
        )
        if trainer.distance_sampling:
            terms.update(trainer.reset_distance_counts(env_state, after.done))
        return (env_state, memory, history, rng, after.scene_state), (data, terms)

    carry, (rollout, terms) = jax.lax.scan(
        step,
        (
            state.env_state,
            state.recurrent_memory,
            state.objective_state,
            state.rng,
            state.env_state.scene_state,
        ),
        None,
        length=trainer.config["horizon"],
    )
    env_state, memory, history, rng, last_scene = carry
    state = state.replace(
        env_state=env_state, recurrent_memory=memory, objective_state=history, rng=rng
    )
    return (
        jax.lax.stop_gradient(state),
        jax.lax.stop_gradient(rollout),
        {**terms, "_preview_scene": jax.lax.stop_gradient(last_scene)},
    )


def recurrent_policy(trainer, params, observations, initial_memory, done):
    """Recompute each sequence under current weights; clear memory after done."""

    def step(memory, inputs):
        observation, reset = inputs
        raw, memory, auxiliary = trainer.actor.apply(params, observation, memory)
        return reset_actor_memory(memory, reset), (raw, auxiliary)

    # Recompute PointNet activations instead of retaining them for every timestep.
    scan_step = jax.checkpoint(step, prevent_cse=False) if trainer.kind == "lidar" else step
    return jax.lax.scan(scan_step, initial_memory, (observations, done))


def monte_carlo_tail(trainer, state, active):
    """Finish copied episodes under the behavior policy without changing the live stream."""
    limit = math.ceil(trainer.env.task.duration / trainer.env.dt) + 1
    env_state = state.env_state.replace(
        task=state.env_state.task.replace(
            event=jnp.where(active, state.env_state.task.event, Event.TIMEOUT)
        )
    )
    initial = (
        env_state,
        state.recurrent_memory,
        state.objective_state,
        jax.random.fold_in(state.rng, 800001),
        active,
        jnp.zeros_like(active, dtype=jnp.float32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )

    def condition(carry):
        return jnp.any(carry[4]) & (carry[6] < limit)

    def step(carry):
        current, memory, history, rng, running, returns, steps, interactions = carry
        rng, key = jax.random.split(rng)
        raw, memory, _ = trainer.actor.apply(state.params, trainer._observe(current), memory)
        action, _, _ = trainer.method.sample(key, raw, state.log_std)
        after = trainer._step(current, action, differentiable=False)
        cost, history, _ = trainer.objective.cost(current, after, action, history)
        return (
            after,
            reset_actor_memory(memory, after.done),
            history,
            rng,
            running & ~after.done,
            returns - jnp.where(running, cost, 0),
            steps + 1,
            interactions + running.sum(dtype=jnp.int32),
        )

    final = jax.lax.while_loop(condition, step, initial)
    metrics = {
        "monte_carlo_tail_interactions": final[7],
        "monte_carlo_tail_allocated_slots": final[6] * trainer.env.num_envs,
        "monte_carlo_tail_unfinished": final[4].sum(),
        "monte_carlo_tail_episodes": active.sum(),
        "monte_carlo_tail_successes": (active & (final[0].task.event == Event.SUCCESS)).sum(),
    }
    return jax.lax.stop_gradient((final[5], metrics))


def update(trainer, state, *, sampling_progress=None):
    """Collect recurrent samples and update actor and critic parameters with PPO."""
    initial_memory = state.recurrent_memory
    state, rollout, terms = collect_rollout(trainer, state, sampling_progress=sampling_progress)
    state, episode_metrics = trainer.episode_statistics(state, rollout.rewards, rollout.done)
    tail_metrics = {}
    if trainer.config["ppo_monte_carlo_returns"]:
        active = ~rollout.done[-1]
        returns, tail_metrics = monte_carlo_tail(trainer, state, active)
        tail_metrics["monte_carlo_tail_value_sample_rmse"] = jnp.sqrt(
            jnp.sum(jnp.where(active, (returns - rollout.next_values[-1]) ** 2, 0))
            / jnp.maximum(active.sum(), 1)
        )
        rollout = rollout.replace(next_values=rollout.next_values.at[-1].set(returns))
    advantages, targets = generalized_advantage_estimate(
        rollout.rewards,
        rollout.values,
        rollout.next_values,
        rollout.terminated,
        rollout.done,
        gamma=trainer.config["gamma"],
        gae_lambda=trainer.config["gae_lambda"],
    )
    value_target_std = targets.std()
    value_explained_variance = 1 - jnp.var(targets - rollout.values) / jnp.maximum(
        jnp.var(targets), 1e-8
    )
    advantage_std = advantages.std()
    advantage_positive_fraction = jnp.mean(advantages > 0.0)
    advantage_mean = advantages.mean()
    advantage_target_correlation = (
        advantage_progress_correlation(advantages, terms["navigation_progress_metres"])
        if "navigation_progress_metres" in terms
        else jnp.zeros_like(advantage_std)
    )
    advantages = normalize_advantages(advantages)
    batches = trainer.config["minibatches"]
    shared = trainer.config["shared_visual_encoder"]

    def epoch(carry, _):
        state, halted = carry
        rng, permutation_key = jax.random.split(state.rng)
        sequences = jax.random.permutation(permutation_key, trainer.env.num_envs).reshape(
            batches, -1
        )
        state = state.replace(rng=rng)

        def minibatch(carry, indices):
            state, halted = carry
            data = jax.tree.map(lambda x: x[:, indices], rollout)
            advantage = advantages[:, indices]
            memory = jax.tree.map(lambda leaf: leaf[indices], initial_memory)
            rng, entropy_key = jax.random.split(state.rng)
            state = state.replace(rng=rng)

            def objective(variables):
                _, (mean, auxiliary) = recurrent_policy(
                    trainer, variables["params"], data.observations, memory, data.done
                )
                log_prob = trainer.method.log_probability(
                    data.pre_tanh, mean, variables["log_std"], data.observations
                )
                ratio = jnp.exp(log_prob - data.log_prob)
                clipped = jnp.clip(
                    ratio, 1 - trainer.config["clip_epsilon"], 1 + trainer.config["clip_epsilon"]
                )
                policy_loss = -jnp.mean(jnp.minimum(ratio * advantage, clipped * advantage))
                # Reparameterized entropy estimate includes the tanh transform's Jacobian.
                entropy = trainer.method.entropy(entropy_key, mean, variables["log_std"])
                auxiliary_loss = jnp.mean(
                    trainer.objective.auxiliary_cost(auxiliary, data.observations)
                )
                loss = policy_loss - trainer.config["entropy_weight"] * entropy + auxiliary_loss
                value_loss = jnp.zeros_like(loss)
                if shared:
                    critic_parameters = shared_critic_parameters(
                        variables["params"], variables["critic_head"]
                    )
                    flattened = jax.tree.map(
                        lambda value: value.reshape((-1, *value.shape[2:])), data.observations
                    )
                    values = trainer.critic.apply(critic_parameters, flattened)
                    value_loss = 0.5 * jnp.mean(
                        (values - jax.lax.stop_gradient(targets[:, indices].reshape(-1))) ** 2
                    )
                    loss = loss + value_loss
                return loss, {
                    "policy_loss": policy_loss,
                    "entropy": entropy,
                    "auxiliary_loss": auxiliary_loss,
                    "approx_kl": jnp.mean((ratio - 1) - (log_prob - data.log_prob)),
                    "clip_fraction": jnp.mean(jnp.abs(ratio - 1) > trainer.config["clip_epsilon"]),
                    "joint_value_loss": value_loss,
                }

            (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
                policy_variables(state, shared_visual_encoder=shared)
            )
            skip_actor = halted | exceeds_kl_limit(
                metrics["approx_kl"], trainer.config["ppo_target_kl"]
            )
            state = jax.lax.cond(
                skip_actor,
                lambda current: current,
                lambda current: trainer._apply_actor(current, gradients),
                state,
            )
            if shared:
                critic_metrics = {
                    "critic_loss": metrics["joint_value_loss"],
                    "critic_grad_norm": optax.tree.norm(gradients["critic_head"]),
                }
            else:
                state, critic_metrics = trainer._fit_critic(
                    state, data.observations, targets[:, indices], 1, old_values=data.values
                )
            return (state, skip_actor), {
                **metrics,
                **critic_metrics,
                "actor_loss": loss - metrics["joint_value_loss"],
                "actor_grad_norm": optax.tree.norm(gradients),
                "kl_actor_update_skipped": skip_actor.astype(jnp.float32),
            }

        return jax.lax.scan(minibatch, (state, halted), sequences)

    (state, _), metrics = jax.lax.scan(
        epoch, (state, jnp.asarray(False)), None, length=trainer.config["ppo_epochs"]
    )
    metrics = jax.tree.map(jnp.mean, metrics)
    metrics.update(tail_metrics)
    preview = {
        key: terms.pop(key)
        for key in ("_preview_position", "_preview_quaternion", "_preview_time", "_preview_scene")
    }
    metrics.update(jax.tree.map(jnp.mean, terms))
    if trainer.distance_sampling:
        for label in trainer.distance_group_labels:
            name = f"reset_distance_{label}_count"
            metrics[name] = jnp.sum(terms[name])
        metrics["distance_curriculum_phase"] = (
            jnp.minimum(jnp.floor(sampling_progress * 3), 2)
            if sampling_progress is not None
            else -1.0
        )
    metrics["reward"] = rollout.rewards.mean()
    metrics["advantage_std"] = advantage_std
    metrics["advantage_positive_fraction"] = advantage_positive_fraction
    metrics["advantage_mean_pre_normalization"] = advantage_mean
    metrics["advantage_progress_correlation"] = advantage_target_correlation
    metrics["value_target_std"] = value_target_std
    metrics["value_prediction_mean"] = rollout.values.mean()
    metrics["value_prediction_std"] = rollout.values.std()
    metrics["value_target_mean"] = targets.mean()
    metrics["critic_return_rmse_preupdate"] = jnp.sqrt(jnp.mean((targets - rollout.values) ** 2))
    metrics["critic_explained_variance_preupdate"] = value_explained_variance
    metrics["saturated_action_fraction"] = jnp.mean(jnp.abs(jnp.tanh(rollout.pre_tanh)) > 0.95)
    metrics["exploration_std"] = jnp.exp(state.log_std).mean()
    if "depth_mask" in rollout.observations:
        metrics["depth_valid_fraction"] = rollout.observations["depth_mask"].mean()
    if "mask" in rollout.observations:
        metrics["lidar_valid_fraction"] = rollout.observations["mask"].mean()
    metrics["done_fraction"] = rollout.done.mean()
    if "navigation_success_fraction" in terms:
        metrics.update(completed_navigation_metrics(rollout.done, terms))
    metrics["loss"] = metrics["actor_loss"] + metrics["critic_loss"]
    return trainer._finish(
        state,
        {
            **metrics,
            **preview,
            **episode_metrics,
            "episode_successes": terms["episode_success"].sum(),
        },
    )
