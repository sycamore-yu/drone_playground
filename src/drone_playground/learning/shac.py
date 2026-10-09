"""Short-horizon actor-critic with differentiable terminal-value bootstrap."""

import jax
import optax

from drone_playground.learning.apg import rollout_objective
from drone_playground.learning.ppo import generalized_advantage_estimate


def update_target(old, new, alpha):
    """Polyak averaging: alpha weights the OLD target variables."""
    return jax.tree.map(lambda old, new: alpha * old + (1 - alpha) * new, old, new)


def update(trainer, state):
    """Update the short-horizon actor and bootstrap critic with SHAC."""

    def objective(params):
        return rollout_objective(trainer, state, params, bootstrap=True)

    (loss, (state, rollout, metrics)), gradients = jax.value_and_grad(objective, has_aux=True)(
        state.params
    )
    state = trainer._apply_actor(state, gradients)
    rollout = jax.lax.stop_gradient(rollout)
    _, targets = generalized_advantage_estimate(
        rollout.rewards,
        rollout.values,
        rollout.next_values,
        rollout.terminated,
        rollout.done,
        gamma=trainer.config["gamma"],
        gae_lambda=trainer.config["gae_lambda"],
    )
    state, critic_metrics = trainer._fit_critic(
        state, rollout.observations, targets, trainer.config["critic_epochs"]
    )
    state = state.replace(
        target_critic_params=update_target(
            state.target_critic_params, state.critic_params, trainer.config["target_alpha"]
        )
    )
    return trainer._finish(
        state,
        {
            **metrics,
            **critic_metrics,
            "loss": loss,
            "actor_loss": loss,
            "actor_grad_norm": optax.global_norm(gradients),
        },
    )
