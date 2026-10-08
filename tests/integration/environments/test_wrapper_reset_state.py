"""Reset complete wrapper state without changing unfinished batch members."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs.base import Env, State

from drone_playground.control.delay import ActionDelay
from drone_playground.learning.wrappers import wrap_for_training


class CountingEnv(Env):
    reset_info_fields = ("effects_key", "disturbance_key", "physical_parameters")
    action_size = 1
    observation_size = 1
    backend = "test"
    dt = 0.02
    hover_action = jnp.zeros(1)

    def __init__(self, structured=False):
        """Initialize the test fixture with the requested observation structure."""
        self.structured = structured

    def reset(self, rng):
        effects, disturbance, parameters = jax.random.split(rng, 3)
        value = jnp.zeros(1)
        obs = {"state": value, "sensor": jnp.zeros((2, 3))} if self.structured else value
        return State(
            value,
            obs,
            jnp.float32(0),
            jnp.float32(0),
            {},
            {
                "effects_key": effects,
                "disturbance_key": disturbance,
                "physical_parameters": jax.random.uniform(parameters, (2,)),
            },
        )

    def step(self, state, action):
        value = state.pipeline_state + action
        obs = {"state": value, "sensor": jnp.full((2, 3), value[0])} if self.structured else value
        return state.replace(pipeline_state=value, obs=obs, done=(value[0] > 0).astype(jnp.float32))


def test_fixed_delay_preserves_inner_reset_contract():
    env = ActionDelay(CountingEnv(), 1)
    assert set(CountingEnv.reset_info_fields) <= set(env.reset_info_fields)


@pytest.mark.parametrize("structured", [False, True])
def test_fixed_delay_resets_all_owned_fields_only_for_finished_member(structured):
    inner = CountingEnv(structured)
    env = wrap_for_training(ActionDelay(inner, 1), episode_length=20)
    original = env.reset(jax.random.split(jax.random.PRNGKey(7), 2))
    action = jnp.array([[1.0], [0.0]])
    state = jax.jit(env.step)(original, action)
    state = jax.jit(env.step)(state, action)
    np.testing.assert_array_equal(state.done, [1, 0])
    np.testing.assert_array_equal(state.pipeline_state, [[0], [0]])
    np.testing.assert_array_equal(state.info["command_queue"][0], 0)
    for key in CountingEnv.reset_info_fields:
        assert not np.array_equal(state.info[key][0], original.info[key][0]), key
        np.testing.assert_array_equal(state.info[key][1], original.info[key][1])
    if structured:
        np.testing.assert_array_equal(state.obs["sensor"][0], 0)
        np.testing.assert_array_equal(state.info["terminal_observation"]["sensor"][0], 1)
    else:
        np.testing.assert_array_equal(state.info["terminal_observation"][0], [1])
