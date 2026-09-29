"""Public final-acceptance conditions, real timing and terminal recording."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs.base import State

from drone_playground.composition import compose_method, validate_config


@pytest.mark.parametrize("method", ["paper/super", "paper/ego_planner"])
def test_navigation_v2_limits_reach_the_worker_request(method, tmp_path):
    from drone_playground.integrations.native_planner import NativePlanner

    cfg = compose_method(method, "navigation/static", ["evaluation=navigation_v2"])
    validate_config(cfg)
    assert cfg["env"]["task"]["duration"] == 300.0
    assert cfg["method"]["limits"]["max_velocity_mps"] == 20.0
    planner = object.__new__(NativePlanner)
    planner.directory = tmp_path
    requests = []
    planner.request = lambda payload, timeout: (
        requests.append(payload) or {"launch_xml": "<launch/>"}
    )
    planner.start({}, [98, 0, 3], limits=cfg["method"]["limits"])
    assert requests[0]["limits"]["max_velocity_mps"] == 20.0


class Integrator:
    """A one-dimensional physical system with a 2ms actuator clock."""

    dt = 0.02
    substeps = 10
    action_size = 1
    hover_action = jnp.array([0.0])
    low, high = jnp.array([-1.0]), jnp.array([1.0])

    def reset(self, key):
        del key
        return State(None, jnp.zeros(1), jnp.float32(0), jnp.float32(0), {}, {})

    def physical_action(self, action):
        return action

    def step_schedule(self, state, commands):
        obs = state.obs + jnp.sum(commands, axis=0) * self.dt / self.substeps
        return state.replace(obs=obs)


def test_fractional_delay_uses_physics_clock_and_propagates_past_action_gradient():
    from drone_playground.execution.delay import RandomActionDelay

    env = RandomActionDelay(Integrator(), [30.0, 30.0])
    state = env.reset(jax.random.PRNGKey(0))
    first = env.step(state, jnp.array([1.0]))
    np.testing.assert_allclose(first.obs, [0.0])
    second = env.step(first, jnp.array([0.0]))
    np.testing.assert_allclose(second.obs, [0.01], atol=1e-7)
    np.testing.assert_allclose(second.info["delay_effective_ms"], 30.0, atol=1e-5)

    def response(action):
        a = env.step(state, jnp.array([action]))
        return env.step(a, jnp.zeros(1)).obs[0]

    assert float(jax.jit(jax.grad(response))(1.0)) == pytest.approx(0.01)


def test_random_delays_are_reproducible_independent_and_stay_in_declared_range():
    from drone_playground.execution.delay import RandomActionDelay

    env = RandomActionDelay(Integrator(), [25.0, 50.0])
    keys = jax.random.split(jax.random.PRNGKey(42), 1024)
    one = jax.jit(jax.vmap(env.reset))(keys)
    two = jax.jit(jax.vmap(env.reset))(keys)
    np.testing.assert_array_equal(one.info["delay_requested_ms"], two.info["delay_requested_ms"])
    requested = np.asarray(one.info["delay_requested_ms"])
    effective = np.asarray(one.info["delay_effective_ms"])
    assert 25.0 <= requested.min() < 26.0 and 49.0 < requested.max() <= 50.0
    assert np.unique(requested).size > 1000
    assert effective.min() >= 25.0 and effective.max() <= 50.0001
    assert np.all(effective + 1e-4 >= requested)
    assert np.max(effective - requested) <= 2.0001


def test_export_trims_each_episode_after_terminal_transition(tmp_path):
    from drone_playground.visualization.rscope_io import trim_episode

    trace = {
        "pos": np.arange(24).reshape(4, 2, 3),
        "time": np.broadcast_to(np.arange(4)[:, None], (4, 2)),
        "active": np.array([[1, 1], [1, 1], [0, 1], [0, 0]], bool),
        "metrics": {"success": np.zeros((4, 2))},
    }
    first, second = trim_episode(trace, 0), trim_episode(trace, 1)
    assert first["time"].shape == (2, 1) and second["time"].shape == (3, 1)
    np.testing.assert_array_equal(first["pos"][-1, 0], trace["pos"][1, 0])


@pytest.mark.parametrize(
    "task", ["hovering", "tracking", "racing", "navigation/static", "navigation/dynamic"]
)
def test_bptt_is_a_real_trainable_composed_method(task):
    cfg = compose_method("learning/bptt", task)
    validate_config(cfg)
    assert cfg["algorithm"]["name"] == "bptt"
    assert cfg["runtime"]["action_delay_ms"] == [25.0, 50.0]


def test_autoreset_resamples_delay_and_clears_only_finished_world():
    from drone_playground.execution.delay import RandomActionDelay
    from drone_playground.learning.env_adapter import wrap_for_training

    class DoneIntegrator(Integrator):
        def step_schedule(self, state, commands):
            result = super().step_schedule(state, commands)
            return result.replace(done=(result.obs[0] > 0.005).astype(jnp.float32))

    env = wrap_for_training(RandomActionDelay(DoneIntegrator(), [25.0, 50.0]), 20)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    delay = np.asarray(state.info["delay_requested_ms"]).copy()
    for _ in range(4):
        state = env.step(state, jnp.array([[1.0], [0.0]]))
        if bool(state.done[0]):
            break
    assert bool(state.done[0]) and not bool(state.done[1])
    assert float(state.info["delay_requested_ms"][0]) != float(delay[0])
    assert float(state.info["delay_requested_ms"][1]) == float(delay[1])
    np.testing.assert_array_equal(state.info["delay_command_queue"][0], 0.0)
