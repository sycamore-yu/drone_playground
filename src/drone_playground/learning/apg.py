"""Rematerialized analytical policy gradients through official Crazyflow steps."""

import jax
import jax.numpy as jnp
import optax

from drone_playground.learning.ppo import Rollout


def rollout_objective(trainer, state, params, *, bootstrap=False):
    """Differentiable segmented rollout, optionally bootstrapped by a target critic.

    Target critic parameters are fixed, but its input-state Jacobian stays live.
    Termination suppresses bootstrap; truncation uses the final pre-reset state.
    Reset boundaries restart the within-segment discount and recurrent history.
    """

    def step(carry, index):
        env_state, memory, history, rng, discount = carry
        rng, reset_key = jax.random.split(rng)
        observation = trainer._observe(env_state)
        raw, memory, auxiliary = trainer.actor.apply(params, observation, memory)
        action = jnp.tanh(raw)
        after = trainer._step(env_state, action, differentiable=True)
        cost, history, terms = trainer._cost(env_state, after, action, history)
        auxiliary_cost = trainer._auxiliary_cost(auxiliary, observation)
        boundary = after.done | (index == trainer.config["horizon"] - 1)
        if bootstrap:
            value = trainer.critic.apply(state.target_critic_params, observation)
            next_value = trainer.critic.apply(state.target_critic_params, trainer._observe(after))
            contribution = discount * (
                cost
                - trainer.config["gamma"] * jnp.where(boundary & ~after.terminated, next_value, 0)
            )
        else:
            value = next_value = jnp.zeros_like(cost)
            contribution = cost
        data = Rollout(
            observation,
            raw,
            jnp.zeros_like(cost),
            -cost,
            value,
            next_value,
            after.terminated,
            after.done,
        )
        env_state, memory, history = trainer._reset(reset_key, after, memory, history)
        discount = jnp.where(after.done, 1.0, discount * trainer.config["gamma"])
        return (env_state, memory, history, rng, discount), (
            contribution + auxiliary_cost,
            data,
            terms,
            auxiliary_cost,
        )

    carry, (costs, rollout, terms, auxiliary) = jax.lax.scan(
        jax.checkpoint(step),
        (
            state.env_state,
            state.recurrent_memory,
            state.objective_state,
            state.rng,
            jnp.ones(trainer.env.num_envs),
        ),
        jnp.arange(trainer.config["horizon"]),
    )
    env_state, memory, history, rng, _ = carry
    state = state.replace(
        env_state=env_state, recurrent_memory=memory, objective_state=history, rng=rng
    )
    metrics = {
        **jax.tree.map(jnp.mean, terms),
        "reward": rollout.rewards.mean(),
        "done_fraction": rollout.done.mean(),
        "auxiliary_loss": auxiliary.mean(),
    }
    return costs.mean(), (state, rollout, metrics)


def update(trainer, state):
    """Apply one differentiable actor update to the recoverable training state."""

    def objective(params):
        return rollout_objective(trainer, state, params)

    (loss, (state, _, metrics)), gradients = jax.value_and_grad(objective, has_aux=True)(
        state.params
    )
    state = trainer._apply_actor(state, gradients)
    return trainer._finish(
        state,
        {
            **metrics,
            "loss": loss,
            "actor_loss": loss,
            "actor_grad_norm": optax.global_norm(gradients),
        },
    )
