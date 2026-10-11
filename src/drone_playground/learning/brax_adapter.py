"""Brax training boundary for the existing batched Crazyflow environment.

This module is optional. It does not add a second physics or task implementation.
Its pipeline_state is the original environment episode state, including all
physical, sensor, command-delay and task history.
"""

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State


def _batch_seed(keys, count):
    """Combine the external per-world reset keys into one Crazyflow batch seed."""
    data = jax.random.key_data(keys)
    if data.ndim == 1:
        return jax.random.wrap_key_data(data, impl=jax.random.key_impl(keys))
    if data.shape[0] != count:
        raise ValueError("Brax reset key batch must match Crazyflow num_envs")
    combined = jnp.bitwise_xor.reduce(data, axis=0)
    return jax.random.wrap_key_data(combined, impl=jax.random.key_impl(keys))


class BraxTrainingEnv(Env):
    """Expose an already-batched Crazyflow environment to native Brax learners.

    Brax's default VmapWrapper must not wrap this instance. For a single-device
    native PPO comparison, use wrap_env=False and use_pmap_on_reset=True. This
    class performs the episode reset itself and keeps the terminal observation.
    """

    def __init__(self, environment):
        """Bind one underlying environment with no replicated physical state."""
        self.environment = environment
        if environment.sensor is not None:
            raise ValueError("Native Brax baseline initially supports state-only observations")
        self._observation_size = int(
            jax.eval_shape(
                lambda key: environment.observe(environment.reset(key))["state"],
                jax.random.PRNGKey(0),
            ).shape[-1]
        )

    @property
    def observation_size(self):
        """Return the dimension of the baseline state observation."""
        return self._observation_size

    @property
    def action_size(self):
        """Use the configured physical action contract, not a task-name lookup."""
        return self.environment.action_size

    @property
    def backend(self):
        """Identify the real physical backend."""
        return "crazyflow"

    def reset(self, rng):
        """Initialize the complete Crazyflow batch and the Brax State fields."""
        key = _batch_seed(rng, self.environment.num_envs)
        episode = self.environment.reset(key)
        observation = self.environment.observe(episode)["state"]
        zeros = jnp.zeros(self.environment.num_envs, observation.dtype)
        return State(
            pipeline_state=episode,
            obs=observation,
            reward=zeros,
            done=zeros,
            metrics={"task_cost": zeros},
            info={
                "reset_key": key,
                "truncation": zeros,
                "terminated": zeros,
                "terminal_observation": observation,
                "steps": jnp.zeros(self.environment.num_envs, dtype=jnp.int32),
                "episode_done": zeros,
                "episode_metrics": {
                    "sum_reward": zeros,
                    "length": zeros,
                    "task_cost": zeros,
                },
            },
        )

    def step(self, state, action):
        """Evaluate a physical transition, then reset only completed worlds."""
        before = state.pipeline_state
        after = self.environment.step(before, action)
        task_cost = self.environment.cost(before, after, action)
        terminal_observation = self.environment.observe(after)["state"]
        terminated = after.terminated
        truncated = after.truncated
        done = after.done
        reward = -task_cost
        previous_done = state.info["episode_done"].astype(bool)
        accum = state.info["episode_metrics"]
        episode_metrics = {
            "sum_reward": jnp.where(previous_done, 0.0, accum["sum_reward"]) + reward,
            "length": jnp.where(previous_done, 0.0, accum["length"]) + 1.0,
            "task_cost": jnp.where(previous_done, 0.0, accum["task_cost"]) + task_cost,
        }
        steps = jnp.where(previous_done, 0, state.info["steps"]) + 1
        reset_key, fresh_key = jax.random.split(state.info["reset_key"])
        episode = jax.lax.cond(
            jnp.any(done),
            lambda: self.environment.reset(fresh_key, after, mask=done),
            lambda: after,
        )
        fresh_observation = self.environment.observe(episode)["state"]
        return state.replace(
            pipeline_state=episode,
            obs=fresh_observation,
            reward=reward,
            done=done.astype(state.done.dtype),
            metrics={"task_cost": task_cost},
            info={
                "reset_key": reset_key,
                "truncation": truncated.astype(state.done.dtype),
                "terminated": terminated.astype(state.done.dtype),
                "terminal_observation": terminal_observation,
                "steps": steps,
                "episode_done": done.astype(state.done.dtype),
                "episode_metrics": episode_metrics,
            },
        )
