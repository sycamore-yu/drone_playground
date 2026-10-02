"""Small deterministic environments used by algorithm tests."""

import jax.numpy as jnp
from brax.envs.base import State


class SmoothTask:
    observation_size = 2
    action_size = 1
    episode_length = 20

    def reset(self, key):
        del key
        return State(
            jnp.zeros(1),
            jnp.zeros(2),
            jnp.float32(0),
            jnp.float32(0),
            {},
            {"terminated": jnp.float32(0)},
        )

    def step(self, state, action):
        position = state.pipeline_state + 0.1 * action
        return state.replace(
            pipeline_state=position,
            obs=jnp.r_[position, 1.0 - position],
            reward=-jnp.square(1.0 - position[0]),
        )
