"""Sensor observations through the fully composed navigation environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.composition import build_environment, compose_experiment, validate_config
from drone_playground.environments.observations.state import NavigationSensorObservation


def test_depth_environment_closed_loop_and_frame_timestamps():
    config = compose_experiment('papers/dva')
    validate_config(config)
    env = build_environment(config, "cpu", "train", 1)
    observation = NavigationSensorObservation(
        name="navigation_depth",
        history=env.sensor.history,
        points_per_frame=env.points_per_frame,
        channels=env.sensor.channels,
        near_m=env.sensor.near_m,
        far_m=env.sensor.far_m,
    )
    assert env.observation_size == observation.size
    assert env.realised_sensor_rate_hz == 25.0
    assert env.sensor_calibration["intrinsics"]["fx_px"] == pytest.approx(65.25, abs=1e-3)

    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    assert state.pipeline_state.sensor_values.shape == (
        env.sensor.history,
        env.points_per_frame,
        env.sensor.channels,
    )
    assert int(state.pipeline_state.sensor_sequence) == 1

    def body(carry, _):
        nxt = env.step(carry, env.hover_action)
        return nxt, (nxt.pipeline_state.sensor_time, nxt.pipeline_state.sensor_sequence)

    final, (times, sequences) = jax.jit(
        lambda s: jax.lax.scan(body, s, None, length=8)
    )(state)
    assert list(np.asarray(sequences).reshape(-1)) == [1, 2, 2, 3, 3, 4, 4, 5]
    last_times = np.asarray(times[-1])
    assert last_times[-1] == pytest.approx(8 * env.dt, abs=1e-6)
    assert last_times[-2] == pytest.approx(6 * env.dt, abs=1e-6)
    assert last_times[-3] == pytest.approx(4 * env.dt, abs=1e-6)
    assert bool(jnp.all(jnp.isfinite(final.obs)))
    env.close()


def test_lidar_batch_environment_resets_sensor_phase():
    config = compose_experiment('navigation/ppo')
    validate_config(config)
    env = build_environment(config, "cpu", "train", 1)
    assert env.sensor.channels == 5
    assert env.points_per_frame == 120
    assert env.observation_size == 20 + 4 * 120 * 5

    keys = jnp.asarray([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
    ids = jnp.asarray([0, env.bank.num_instances // 3])
    states = jax.vmap(env.reset)(keys, ids)
    clouds = jnp.asarray(states.pipeline_state.sensor_values[:, -1, :, :3])
    assert not np.allclose(np.asarray(clouds[0]), np.asarray(clouds[1]))

    cursor = states
    for _ in range(5):
        cursor = jax.vmap(env.step)(
            cursor, jnp.broadcast_to(env.hover_action, (2, 4))
        )
    assert list(np.asarray(cursor.pipeline_state.sensor_sequence)) == [2, 2]

    fresh = jax.vmap(env.reset)(keys, ids)
    assert list(np.asarray(fresh.pipeline_state.sensor_sequence)) == [1, 1]
    np.testing.assert_allclose(
        fresh.pipeline_state.sensor_time, states.pipeline_state.sensor_time
    )
    env.close()
