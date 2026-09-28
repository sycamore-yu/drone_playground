"""Training-only batching, episode truncation and fresh auto-reset."""

import jax
import jax.numpy as jnp
from brax.envs.base import State, Wrapper
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper


class FreshAutoReset(Wrapper):
    """Brax same-step reset with fresh initial states and saved terminal observations.

    EpisodeWrapper owns truncation and metrics. Unlike Brax's cached-reset wrapper,
    a completed trial consumes its own new random key. Resets disconnect gradients.
    """

    def __init__(self, env, reset_info_fields=()):
        super().__init__(env)
        self.reset_info_fields = tuple(reset_info_fields)

    def reset(self, rng: jax.Array) -> State:
        children = jax.vmap(jax.random.split)(rng)
        state = self.env.reset(children[:, 1])
        return state.replace(
            info={
                **state.info,
                "reset_key": children[:, 0],
                "terminal_observation": state.obs,
                "time_out": jnp.zeros_like(state.done),
            }
        )

    def step(self, state: State, action: jax.Array) -> State:
        info = {**state.info, "steps": jnp.where(state.done, 0, state.info["steps"])}
        state = state.replace(done=jnp.zeros_like(state.done), info=info)
        terminal = self.env.step(state, action)
        done = terminal.done.astype(bool)
        keys = jax.vmap(jax.random.split)(terminal.info["reset_key"])

        def choose(fresh, current):
            mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
            return jnp.where(mask, jax.lax.stop_gradient(fresh), current)

        def reset_done(_):
            fresh = self.env.reset(keys[:, 1])
            data = jax.tree.map(choose, fresh.pipeline_state, terminal.pipeline_state)
            obs = choose(fresh.obs, terminal.obs)
            memory = {
                key: choose(fresh.info[key], terminal.info[key]) for key in self.reset_info_fields
            }
            return data, obs, memory

        data, obs, memory = jax.lax.cond(
            jnp.any(done),
            reset_done,
            lambda _: (
                terminal.pipeline_state,
                terminal.obs,
                {key: terminal.info[key] for key in self.reset_info_fields},
            ),
            operand=None,
        )
        info = {
            **terminal.info,
            **memory,
            "terminal_observation": terminal.obs,
            "time_out": terminal.info["truncation"],
            "reset_key": jnp.where(done[:, None], keys[:, 0], terminal.info["reset_key"]),
        }
        return terminal.replace(pipeline_state=data, obs=obs, info=info)


def wrap_for_training(
    env, episode_length: int, action_repeat: int = 1, randomization_fn=None
) -> FreshAutoReset:
    """Compose native Brax batching/statistics with the task's fresh-reset contract."""
    if randomization_fn is not None:
        raise ValueError("Randomization belongs to this Crazyflow task's reset, not a Brax System")
    if action_repeat != 1:
        raise ValueError("Action repeat is fixed at one; task frequency owns physical substeps")
    return FreshAutoReset(
        EpisodeWrapper(VmapWrapper(env), episode_length, action_repeat),
        reset_info_fields=getattr(env, "reset_info_fields", ()),
    )
