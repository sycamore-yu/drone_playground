"""Stochastic tanh-Gaussian PPO with complete recurrent-sequence recomputation."""

import jax
import jax.numpy as jnp
import optax
from flax import struct

from drone_playground.simulation.networks import sample_tanh_gaussian, tanh_gaussian_log_prob


@struct.dataclass
class Rollout:
    """Carry sampled transitions and recurrent memory for PPO updates."""

    observations: dict
    pre_tanh: jax.Array
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


def collect_rollout(trainer, state):
    """Collect stochastic transitions and retain pre-reset critic observations."""

    def step(carry, _):
        env_state, memory, history, rng = carry
        rng, action_key, reset_key = jax.random.split(rng, 3)
        observation = trainer._observe(env_state)
        raw, memory, _ = trainer.actor.apply(state.params, observation, memory)
        action, pre_tanh, log_prob = sample_tanh_gaussian(action_key, raw, state.log_std)
        after = trainer._step(env_state, action, differentiable=False)
        cost, history, terms = trainer._cost(env_state, after, action, history)
        value = trainer.critic.apply(state.critic_params, observation)
        next_value = trainer.critic.apply(state.critic_params, trainer._observe(after))
        data = Rollout(
            observation, pre_tanh, log_prob, -cost, value, next_value, after.terminated, after.done
        )
        env_state, memory, history = trainer._reset(reset_key, after, memory, history)
        return (env_state, memory, history, rng), (data, terms)

    carry, (rollout, terms) = jax.lax.scan(
        step,
        (state.env_state, state.recurrent_memory, state.objective_state, state.rng),
        None,
        length=trainer.config["horizon"],
    )
    env_state, memory, history, rng = carry
    state = state.replace(
        env_state=env_state, recurrent_memory=memory, objective_state=history, rng=rng
    )
    return jax.lax.stop_gradient(state), jax.lax.stop_gradient(rollout), terms


def recurrent_policy(trainer, params, observations, initial_memory, done):
    """Recompute each sequence under current weights; clear memory after done."""

    def step(memory, inputs):
        observation, reset = inputs
        raw, memory, auxiliary = trainer.actor.apply(params, observation, memory)
        return jnp.where(reset[:, None], 0, memory), (raw, auxiliary)

    return jax.lax.scan(step, initial_memory, (observations, done))


def update(trainer, state):
    """Collect recurrent samples and update actor and critic parameters with PPO."""
    initial_memory = state.recurrent_memory
    state, rollout, terms = collect_rollout(trainer, state)
    advantages, targets = generalized_advantage_estimate(
        rollout.rewards,
        rollout.values,
        rollout.next_values,
        rollout.terminated,
        rollout.done,
        gamma=trainer.config["gamma"],
        gae_lambda=trainer.config["gae_lambda"],
    )
    advantages = normalize_advantages(advantages)
    batches = trainer.config["minibatches"]

    def epoch(state, _):
        rng, permutation_key = jax.random.split(state.rng)
        sequences = jax.random.permutation(permutation_key, trainer.env.num_envs).reshape(
            batches, -1
        )
        state = state.replace(rng=rng)

        def minibatch(state, indices):
            data = jax.tree.map(lambda x: x[:, indices], rollout)
            advantage = advantages[:, indices]
            memory = initial_memory[indices]
            rng, entropy_key = jax.random.split(state.rng)
            state = state.replace(rng=rng)

            def objective(variables):
                _, (mean, auxiliary) = recurrent_policy(
                    trainer, variables["params"], data.observations, memory, data.done
                )
                log_prob = tanh_gaussian_log_prob(data.pre_tanh, mean, variables["log_std"])
                ratio = jnp.exp(log_prob - data.log_prob)
                clipped = jnp.clip(
                    ratio, 1 - trainer.config["clip_epsilon"], 1 + trainer.config["clip_epsilon"]
                )
                policy_loss = -jnp.mean(jnp.minimum(ratio * advantage, clipped * advantage))
                # Reparameterized entropy estimate includes the tanh transform's Jacobian.
                _, _, entropy_log_prob = sample_tanh_gaussian(
                    entropy_key, mean, variables["log_std"]
                )
                entropy = -jnp.mean(entropy_log_prob)
                auxiliary_loss = jnp.mean(trainer._auxiliary_cost(auxiliary, data.observations))
                loss = policy_loss - trainer.config["entropy_weight"] * entropy + auxiliary_loss
                return loss, {
                    "policy_loss": policy_loss,
                    "entropy": entropy,
                    "auxiliary_loss": auxiliary_loss,
                    "approx_kl": jnp.mean((ratio - 1) - (log_prob - data.log_prob)),
                    "clip_fraction": jnp.mean(jnp.abs(ratio - 1) > trainer.config["clip_epsilon"]),
                }

            (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
                {"params": state.params, "log_std": state.log_std}
            )
            state = trainer._apply_actor(state, gradients)
            state, critic_metrics = trainer._fit_critic(
                state, data.observations, targets[:, indices], 1
            )
            return state, {
                **metrics,
                **critic_metrics,
                "actor_loss": loss,
                "actor_grad_norm": optax.tree.norm(gradients),
            }

        return jax.lax.scan(minibatch, state, sequences)

    state, metrics = jax.lax.scan(epoch, state, None, length=trainer.config["ppo_epochs"])
    metrics = jax.tree.map(jnp.mean, metrics)
    metrics.update(jax.tree.map(jnp.mean, terms))
    metrics["reward"] = rollout.rewards.mean()
    metrics["done_fraction"] = rollout.done.mean()
    metrics["loss"] = metrics["actor_loss"] + metrics["critic_loss"]
    return trainer._finish(state, metrics)
