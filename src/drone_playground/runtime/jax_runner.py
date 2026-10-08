"""Shared batched closed-loop rollout, preserving each task's trace projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rollout(env, policy, initial, length, project, *, early_exit=False):
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

    start = (initial, jnp.ones(batch, bool))
    if not early_exit:
        return jax.lax.scan(one, start, jnp.arange(length))[1]

    # Evaluation only: a dynamic while-loop really stops launching per-tick
    # kernels once all worlds have ended. A fixed scan still launches thousands
    # of inactive iterations for a 300 s navigation deadline. Preserve the exact
    # old padded trace in one vectorized operation, without advancing physics.
    template = inactive((initial, jnp.zeros(batch, bool)), jnp.int32(0))[1]
    buffers = jax.tree.map(lambda value: jnp.zeros((length, *value.shape), value.dtype), template)

    def condition(carry):
        current, index, _ = carry
        return (index < length) & jnp.any(current[1])

    def body(carry):
        current, index, archive = carry
        current, row = advance(current, index)
        archive = jax.tree.map(lambda storage, value: storage.at[index].set(value), archive, row)
        return current, index + 1, archive

    final, stop, archive = jax.lax.while_loop(condition, body, (start, jnp.int32(0), buffers))
    padding = jax.vmap(lambda index: inactive(final, index)[1])(jnp.arange(length))
    return jax.tree.map(
        lambda prefix, tail: jnp.where(
            (jnp.arange(length) < stop).reshape((length,) + (1,) * (prefix.ndim - 1)),
            prefix,
            tail,
        ),
        archive,
        padding,
    )


def recurrent_scan(step, initial, length, *, rematerialize=False):
    """Differentiable recurrent unfolding with explicit method and physical state."""
    if rematerialize:
        step = jax.checkpoint(step, prevent_cse=False)
    return jax.lax.scan(step, initial, jnp.arange(length))


def policy_rollout(
    env,
    make_policy,
    parameters,
    keys,
    *,
    reference_ids=None,
    initial_states=None,
    kind="tracking",
):
    """Roll out a JAX policy over fixed environment initial conditions."""
    if initial_states is not None:
        initial = jax.vmap(env.reset)(keys, reference_ids, initial_states)
    else:
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
            row.update(
                done=new.done,
                outcome=new.info["outcome"],
                observation_pos=old.pipeline_state.sim_data.states.pos[:, 0, 0],
                sensor_capture_time=old.pipeline_state.sensor_time[:, -1],
                observation_time=jnp.full(alive.shape, index * env.dt),
            )
        if "applied_action" in new.info:
            row["requested_actions"] = action
            row["actions"] = new.info["applied_action"]
        return row

    return rollout(
        env,
        policy,
        initial,
        env.episode_length,
        project,
        early_exit=kind == "navigation",
    )
