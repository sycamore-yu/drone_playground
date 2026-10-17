"""Asynchronous observations must inform the policy how stale their sensor is."""

import jax
import numpy as np

from drone_playground.simulation.environment import make_environment


def setup_sensor(frequency, method_hz, latency=0.0):
    """Run real Crazyflow with a small optical snapshot and public OA preprocessing."""
    return make_environment(
        task="navigation",
        scene="S01",
        num_envs=1,
        device="cpu",
        physics_hz=100,
        method_hz=method_hz,
        sensor="depth",
        sensor_config={
            "width": 4,
            "height": 4,
            "max_range": 10.0,
            "frequency_hz": frequency,
            "latency": latency,
        },
        observation_config={"depth": {"motion_compensate": True, "representation": "metric"}},
        action={"level": "acceleration", "frame": "body"},
    )


def test_multi_rate_sensor_age_enters_actor_state_even_without_transport_delay():
    """At 20 Hz policy/10 Hz camera, the intermediate observation is 50 ms old."""
    env = setup_sensor(10.0, 20)
    initial = env.reset(jax.random.key(7))
    observed = env.observe(initial)
    assert observed["state"].shape[-1] == 11
    assert float(observed["sensor_age_s"][0]) == 0.0
    later = env.step(initial, np.zeros((1, 3), dtype=np.float32))
    obs = env.observe(later)
    np.testing.assert_allclose(obs["sensor_age_s"], [0.05], atol=1e-6)
    np.testing.assert_allclose(obs["state"][:, -1], [0.5], atol=1e-6)
    newframe = env.step(later, np.zeros((1, 3), dtype=np.float32))
    obs = env.observe(newframe)
    np.testing.assert_allclose(obs["sensor_age_s"], [0.0], atol=1e-6)
    np.testing.assert_allclose(obs["state"][:, -1], [0.0], atol=1e-6)
    env.sim.close()


def test_same_frequency_no_latency_retains_accepted_state_dimension_and_values():
    """Existing 10 Hz OA release checkpoints are not invalidated by zero-age channels."""
    env = setup_sensor(10.0, 10)
    state = env.reset(jax.random.key(7))
    before = env.observe(state)
    assert before["state"].shape[-1] == 10
    after = env.observe(env.step(state, np.zeros((1, 3), dtype=np.float32)))
    assert after["state"].shape[-1] == 10
    np.testing.assert_allclose(after["sensor_age_s"], [0.0], atol=1e-6)
    env.sim.close()


def test_transport_delayed_sensor_age_enters_policy_state():
    """Actual delayed publication adds one age feature even at equal nominal rates."""
    env = setup_sensor(10.0, 10, latency=0.03)
    state = env.reset(jax.random.key(7))
    assert env.observe(state)["state"].shape[-1] == 11
    later = env.step(state, np.zeros((1, 3), dtype=np.float32))
    obs = env.observe(later)
    assert obs["state"].shape[-1] == 11
    assert float(obs["sensor_age_s"][0]) > 0
    assert float(obs["state"][0, -1]) > 0
    env.sim.close()
