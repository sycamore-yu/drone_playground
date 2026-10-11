"""Optional Brax interface checks against the real Crazyflow environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

brax = pytest.importorskip("brax")
from brax.envs.base import Env, State  # noqa: E402

from drone_playground.learning.brax_adapter import BraxTrainingEnv  # noqa: E402
from drone_playground.simulation.environment import make_environment  # noqa: E402
from drone_playground.simulation.tasks import Event  # noqa: E402


def test_brax_adapter_preserves_batched_crazyflow_and_fresh_reset():
    """One Brax State carries all worlds and resets only completed episodes."""
    original = make_environment(
        task="tracking", num_envs=2, device="cpu", physics_hz=100, method_hz=50, duration=0.04
    )
    try:
        env = BraxTrainingEnv(original)
        assert isinstance(env, Env)
        keys = jax.random.split(jax.random.PRNGKey(7), 2)
        state = env.reset(keys)
        assert isinstance(state, State)
        assert state.pipeline_state.physics.states.pos.shape == (2, 1, 3)
        assert state.obs.shape == (2, env.observation_size)
        assert state.reward.shape == (2,)
        assert env.backend == "crazyflow"
        action = jnp.zeros((2, env.action_size))
        first = env.step(state, action)
        assert not np.any(np.asarray(first.done))
        second = env.step(first, action)
        assert np.all(np.asarray(second.done))
        assert np.array_equal(
            np.asarray(second.done, bool),
            np.asarray(second.info["truncation"], bool)
            | np.asarray(second.info["terminated"], bool),
        )
        np.testing.assert_allclose(second.info["episode_metrics"]["length"], [2, 2])
        assert np.all(np.asarray(second.pipeline_state.time) == 0)
        assert second.info["terminal_observation"].shape == (2, 28)
        third = env.step(second, action)
        np.testing.assert_allclose(third.info["episode_metrics"]["length"], [1, 1])
        np.testing.assert_array_equal(third.info["episode_done"], [0, 0])
    finally:
        original.sim.close()


def test_brax_adapter_resets_only_ended_worlds():
    """Keep the other physical world, sensor timing and episode counter untouched."""
    original = make_environment(
        task="tracking", num_envs=2, device="cpu", physics_hz=100, method_hz=50, duration=0.10
    )
    try:
        adapter = BraxTrainingEnv(original)
        initial = adapter.reset(jax.random.split(jax.random.PRNGKey(4), 2))
        one_finished = initial.pipeline_state.replace(
            task=initial.pipeline_state.task.replace(
                event=jnp.array([Event.TIMEOUT, Event.RUNNING], dtype=jnp.int32)
            )
        )
        after = adapter.step(initial.replace(pipeline_state=one_finished), jnp.zeros((2, 4)))
        np.testing.assert_array_equal(after.done, [1, 0])
        np.testing.assert_allclose(after.pipeline_state.time, [0, original.dt], atol=1e-8)
        np.testing.assert_array_equal(after.info["truncation"], [1, 0])
    finally:
        original.sim.close()
