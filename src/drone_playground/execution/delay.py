"""Explicit command delay; all queue state belongs to its environment instance."""

import jax.numpy as jnp
from brax.envs.base import Wrapper


class ActionDelay(Wrapper):
    reset_info_fields = ("command_queue",)

    def __init__(self, env, steps: int):
        super().__init__(env)
        if steps < 1:
            raise ValueError("ActionDelay requires a positive delay")
        self.delay_steps = int(steps)

    def reset(self, rng, *args, **kwargs):
        state = self.env.reset(rng, *args, **kwargs)
        queue = jnp.broadcast_to(self.env.hover_action, (self.delay_steps, self.env.action_size))
        return state.replace(
            info={**state.info, "command_queue": queue, "applied_action": self.env.hover_action}
        )

    def step(self, state, action):
        queue = state.info["command_queue"]
        result = self.env.step(state, queue[0])
        updated = jnp.concatenate((queue[1:], action[None]), axis=0)
        return result.replace(
            info={**result.info, "command_queue": updated, "applied_action": queue[0]}
        )

    def step_physical(self, state, physical):
        action = 2 * (physical - self.env.low) / (self.env.high - self.env.low) - 1
        return self.step(state, action)
