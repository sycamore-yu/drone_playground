"""A full 300s horizon must not re-simulate an already terminated batch."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.runtime.jax_runner import rollout


def test_terminal_batch_preserves_terminal_state_without_more_physics_calls():
    calls = []

    class State(NamedTuple):
        obs: jax.Array
        done: jax.Array

    class Env:
        action_size = 1

        @staticmethod
        def step(state, action):
            jax.debug.callback(lambda x: calls.append(float(x)), state.obs[0])
            observation = state.obs + 1
            return State(observation, (observation[0] >= 2).astype(jnp.float32))

    def policy(obs, key):
        return jnp.ones_like(obs), {}

    def project(old, new, action, alive, ended, index):
        return dict(pos=new.obs, active=alive, action=action, done=ended)

    initial = State(jnp.zeros((1, 1)), jnp.zeros(1))
    trace = jax.jit(lambda: rollout(Env(), policy, initial, 6, project))()
    jax.block_until_ready(trace)
    assert len(calls) == 2
    np.testing.assert_array_equal(trace["pos"][:, 0, 0], [1, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(trace["active"][:, 0], [True, True, False, False, False, False])
    np.testing.assert_array_equal(trace["action"][:2, 0, 0], [1, 1])
