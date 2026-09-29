"""Shared batched closed-loop rollout, preserving each task's trace projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rollout(env, policy, initial, length, project):
    """Run fixed-shape trajectories; completed worlds retain their terminal state.

    ``project`` maps the old/new state and applied action into task-specific
    records. It cannot choose reset, step order or episode survival.
    """
    batch = initial.done.shape[0]

    def advance(carry, index):
        current, alive = carry
        action, _ = policy(current.obs, jax.random.PRNGKey(0))
        candidate = jax.vmap(env.step)(current, action)

        def freeze(old, new):
            mask = alive.reshape(alive.shape + (1,) * (new.ndim - alive.ndim))
            return jnp.where(mask, new, old)

        next_state = jax.tree.map(freeze, current, candidate)
        ended = (~alive) | next_state.done.astype(bool)
        record = project(current, next_state, action, alive, ended, index)
        return (next_state, ~ended), record

    def inactive(carry, index):
        current, alive = carry
        action = jnp.zeros((batch, env.action_size), dtype=initial.obs.dtype)
        record = project(current, current, action, alive, ~alive, index)
        return carry, record

    def one(carry, index):
        # Keep static archive shapes without evaluating policy/sensors/physics after
        # the entire batch has ended. Exporters retain only the real active prefix.
        return jax.lax.cond(jnp.any(carry[1]), advance, inactive, carry, index)

    return jax.lax.scan(one, (initial, jnp.ones(batch, bool)), jnp.arange(length))[1]


def recurrent_scan(step, initial, length, *, rematerialize=False):
    """Differentiable recurrent unfolding with explicit method and physical state."""
    if rematerialize:
        step = jax.checkpoint(step, prevent_cse=False)
    return jax.lax.scan(step, initial, jnp.arange(length))


def policy_rollout(env, make_policy, parameters, keys, *, reference_ids=None, kind="tracking"):
    initial = (
        jax.vmap(env.reset)(keys)
        if reference_ids is None
        else jax.vmap(env.reset)(keys, reference_ids)
    )
    policy = make_policy(parameters, deterministic=True)

    def project(old, new, action, alive, ended, index):
        physical = new.pipeline_state.sim_data.states
        row = dict(
            pos=physical.pos[:, 0, 0],
            quat=physical.quat[:, 0, 0],
            obs=old.obs if kind == "racing" else new.obs,
            time=jnp.full(alive.shape, (index + 1) * env.dt),
            actions=action,
            reward=jnp.where(alive, new.reward, 0.0) if kind == "racing" else new.reward,
            metrics=new.metrics,
            active=alive,
            failed=new.metrics["failure"] > 0 if kind == "racing" else ended,
        )
        if kind == "navigation":
            row.update(done=new.done, outcome=new.info["outcome"])
        if "applied_action" in new.info:
            row["requested_actions"] = action
            row["actions"] = new.info["applied_action"]
        return row

    return rollout(env, policy, initial, env.episode_length, project)
