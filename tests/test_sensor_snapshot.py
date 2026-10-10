"""Snapshot acquisition and released Livox scan resource contracts."""

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.simulation.scene import Scene
from drone_playground.simulation.sensors import SensorConfig, measure, sensor_rays


def test_snapshot_measurement_has_one_acquisition_time():
    """All rays belong to the declared frame time, including misses."""
    config = SensorConfig.mid360(points_per_frame=32, latency=0.04)
    times = jnp.array([0.1, 0.2])
    result = jax.jit(
        lambda t: measure(Scene("empty"), config, jnp.zeros((2, 3)), jnp.array([0.0, 0, 0, 1.0]), t)
    )(times)
    np.testing.assert_array_equal(
        result.times, np.broadcast_to(np.asarray(times)[:, None], (2, 32))
    )
    np.testing.assert_allclose(result.available_time, times + 0.04)
    assert not result.mask.any()


def test_mid360_uses_sequential_released_angles_and_wraps():
    """Each environment selects its own frame from the unchanged source table."""
    from mujoco_lidar.scan_gen import LivoxGenerator

    source = LivoxGenerator("mid360").ray_angles
    config = SensorConfig.mid360(points_per_frame=20000)
    frames = jnp.array([0, 1, len(source) // config.points_per_frame])
    actual, offsets = jax.jit(lambda frame: sensor_rays(config, frame_index=frame))(frames)
    indices = (np.asarray(frames)[:, None] * 20000 + np.arange(20000)) % len(source)
    theta, phi = source[indices, 0], source[indices, 1]
    expected = np.stack((np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)), -1)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_array_equal(actual[0], actual[2])
    np.testing.assert_array_equal(offsets, 0)


def test_camera_calibration_does_not_hide_a_training_mode():
    """Image dimensions and frequency are explicit simulation parameters."""
    assert "mode" not in inspect.signature(SensorConfig.d435i).parameters
    camera = SensorConfig.d435i(width=64, height=48, frequency_hz=15, max_range=10)
    assert (camera.width, camera.height, camera.frequency_hz, camera.pitch_deg) == (64, 48, 15, 0)


def test_empty_depth_does_not_turn_range_roundoff_into_hits():
    """A miss stays invalid when radial range is converted to axial depth."""
    config = SensorConfig.d435i(width=64, height=48, max_range=10)
    result = measure(Scene("empty"), config, jnp.zeros(3), jnp.array([0.0, 0, 0, 1]), 0.0)
    assert not np.asarray(result.mask).any()
    np.testing.assert_array_equal(result.values, 0)


def test_trainer_records_sensor_semantics_and_source_hash():
    """A checkpoint identifies the snapshot model, scan table and observation budget."""
    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.environment import Environment

    env = Environment(
        task="navigation",
        scene="S01",
        sensor="lidar",
        point_count=32,
        sensor_config={"points_per_frame": 32},
        action={"level": "acceleration"},
    )
    trainer = Trainer(env, kind="lidar", algorithm="apg", loss="liu")
    assert "sensor" in trainer.resolved_config
    sensor = trainer.resolved_config["sensor"]
    assert sensor["config"]["acquisition"] == "snapshot"
    assert sensor["config"]["scan_length"] == 800000
    assert len(sensor["config"]["scan_sha256"]) == 64
    assert sensor["point_count"] == 32


def test_adjacent_pose_interpolation_handles_antipodal_quaternions():
    """Quaternion sign changes do not turn a stationary sensor into an invalid pose."""
    from drone_playground.simulation.observation import SensorObservation

    previous = jnp.array([[0.032, 0, 0, 0, 0, 0, 0, 1.0]])
    current = jnp.array([[0.034, 0.06, 0, 0, 0, 0, 0, -1.0]])
    position, quaternion = jax.jit(SensorObservation._pose_at)(
        previous, current, jnp.array([1 / 30])
    )
    np.testing.assert_allclose(position, [[0.04, 0, 0]], atol=2e-7)
    np.testing.assert_allclose(quaternion, [[0, 0, 0, 1]], atol=1e-7)


@pytest.mark.parametrize("speed", [3.0, 20.0])
@pytest.mark.parametrize("latency", [0.0, 0.04])
def test_snapshot_clock_motion_delay_and_acquisition_gradient(tmp_path, speed, latency):
    """A 30 Hz snapshot uses one exact moving-world pose, even with delayed delivery."""
    from drone_playground.simulation.observation import SensorObservation

    path = tmp_path / "moving.xml"
    path.write_text(
        '<mujoco><worldbody><body pos="5.5 0 0" mocap="true" user="2 2 0 0 4 0">'
        '<geom type="box" size=".5 10 10"/></body></worldbody></mujoco>'
    )
    scene = Scene(path)
    config = SensorConfig.d435i(width=1, height=1, max_range=10, latency=latency)
    sensor = SensorObservation(scene, config, physics_hz=500)

    def physics(tick, offset=0.0):
        position = jnp.array([[[speed * tick / 500 + offset, 0.0, 0.0]]])
        return SimpleNamespace(
            states=SimpleNamespace(pos=position, quat=jnp.array([[[0.0, 0, 0, 1.0]]])),
            core=SimpleNamespace(
                steps=jnp.array([[tick]]), freq=500, rng_key=jax.random.PRNGKey(1)
            ),
        )

    initial = sensor.reset(physics(0))

    def advance(offset):
        def step(state, tick):
            return sensor.update(state, physics(tick, offset), jnp.array([True])), None

        return jax.lax.scan(step, initial, jnp.arange(1, 81))[0]

    result = jax.jit(advance)(0.0)
    acquired = float(result.measurement.acquisition_time[0])
    assert abs(acquired - (4 / 30 if latency == 0 else 3 / 30)) < 1e-7
    expected = 3.0 + 2 * acquired - speed * acquired
    np.testing.assert_allclose(result.measurement.values, expected, atol=2e-6)
    np.testing.assert_allclose(result.acquisition_pose[0, 0], speed * acquired, atol=2e-6)
    np.testing.assert_array_equal(result.measurement.times, acquired)
    derivative = jax.jit(jax.grad(lambda x: advance(x).measurement.values.sum()))(0.0)
    assert float(derivative) == 0
    assert np.linalg.norm(jax.grad(scene.clearance)(jnp.array([0.0, 0, 0]))) > 0

    frozen = sensor.update(result, physics(81), jnp.array([False]))
    for before, after in zip(jax.tree.leaves(result), jax.tree.leaves(frozen), strict=True):
        np.testing.assert_array_equal(before, after)
