"""Rematerialized analytical policy gradients through official Crazyflow steps."""

import jax
import jax.numpy as jnp
import optax

from drone_playground.learning.ppo import Rollout
from drone_playground.simulation.tasks import Event


def rollout_objective(trainer, state, params, *, bootstrap=False, sampling_progress=None):
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
        action = trainer.method.deterministic(raw)
        after = trainer._step(env_state, action, differentiable=True)
        cost, history, terms = trainer.objective.cost(env_state, after, action, history)
        terms = {
            **terms,
            "gates_advanced": (after.task.gates - env_state.task.gates).astype(jnp.float32),
            "missed_gate_fraction": (
                (after.task.event == Event.MISSED_GATE) & ~env_state.done
            ).astype(jnp.float32),
            "collision_fraction": ((after.task.event == Event.COLLISION) & ~env_state.done).astype(
                jnp.float32
            ),
            "out_of_bounds_fraction": (
                (after.task.event == Event.OUT_OF_BOUNDS) & ~env_state.done
            ).astype(jnp.float32),
        }
        auxiliary_cost = trainer.objective.auxiliary_cost(auxiliary, observation)
        boundary = after.done | (index == trainer.config["horizon"] - 1)
        if bootstrap:
            value = trainer.critic.apply(state.target_critic_params, observation)
            next_observation = trainer._observe(after)
            next_value = trainer.critic.apply(state.target_critic_params, next_observation)
            contribution = discount * (
                cost
                - trainer.config["bootstrap_weight"]
                * trainer.config["gamma"]
                * jnp.where(boundary & ~after.terminated, next_value, 0)
            )
            if trainer.config["report_terminal_critic_grad"]:
                if trainer.env.task.name != "racing":
                    raise ValueError("Terminal Critic gradient trace is Racing-only")

                def grad_value():
                    def evaluate(obs):
                        return trainer.critic.apply(
                            state.target_critic_params,
                            {**next_observation, "state": obs},
                        ).sum()

                    return jnp.linalg.norm(jax.grad(evaluate)(next_observation["state"]), axis=-1)

                grad_norm = jax.lax.cond(
                    jnp.any(boundary & ~after.terminated),
                    grad_value,
                    lambda: jnp.zeros_like(cost),
                )
                terms["terminal_critic_input_gradient_norm"] = jnp.where(
                    boundary & ~after.terminated, grad_norm, 0.0
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
        env_state, memory, history = trainer._reset(
            reset_key, after, memory, history, sampling_progress=sampling_progress
        )
        if trainer.distance_sampling:
            distance = jnp.linalg.norm(
                env_state.scene_state.start - env_state.scene_state.goal, axis=-1
            )
            for metres in (20, 50, 80):
                terms[f"reset_distance_{metres}m_count"] = (
                    after.done & (jnp.abs(distance - metres) < 0.01)
                ).astype(jnp.float32)
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
    if trainer.distance_sampling:
        for metres in (20, 50, 80):
            name = f"reset_distance_{metres}m_count"
            metrics[name] = jnp.sum(terms[name])
        metrics["distance_curriculum_phase"] = (
            jnp.minimum(jnp.floor(sampling_progress * 3), 2)
            if sampling_progress is not None
            else -1.0
        )
    if bootstrap:
        metrics["bootstrap_terminal_value_abs"] = jnp.abs(rollout.next_values[-1]).mean()
        metrics["bootstrap_terminal_vs_step_cost"] = (
            trainer.config["gamma"]
            * jnp.abs(rollout.next_values[-1]).mean()
            / jnp.maximum(jnp.abs(rollout.rewards).mean(), 1e-6)
        )
    return costs.mean(), (state, rollout, metrics)


def update(trainer, state, *, sampling_progress=None):
    """Apply one differentiable actor update to the recoverable training state."""

    def objective(params):
        return rollout_objective(trainer, state, params, sampling_progress=sampling_progress)

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
            "actor_grad_norm": optax.tree.norm(gradients),
        },
    )
