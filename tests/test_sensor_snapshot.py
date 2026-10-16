"""Snapshot acquisition and released Livox scan resource contracts."""

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.simulation.observation import SensorObservation
from drone_playground.simulation.scene import Scene
from drone_playground.simulation.sensors import SensorConfig, measure, sensor_rays


@pytest.mark.parametrize("count", [None, True, 0, -1, 1.5, 20001])
def test_lidar_observation_rejects_invalid_point_count(count):
    """LiDAR capture requires one explicit, bounded observation count."""
    with pytest.raises(ValueError, match=r"observation\.lidar\.point_count"):
        SensorObservation(
            Scene("empty"),
            SensorConfig.mid360(),
            physics_hz=500,
            observation_config={"lidar": {"point_count": count}},
        )


def test_factory_uses_only_the_observation_point_count():
    """The resolved count reaches acquisition without a second sensor parameter."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from drone_playground import cli
    from drone_playground.simulation.factory import create_environment

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(cli.__file__).parent / "configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                "experiment=oa_static_lidar",
                "simulation.device=cpu",
                "simulation.num_envs=1",
                "observation.lidar.point_count=512",
            ],
        )
        depth = compose(config_name="config", overrides=["experiment=oa_static_depth"])
    assert depth.sensor.config.frequency_hz == 10
    assert set(config.sensor) == {"name", "config"}
    env = create_environment(config)
    try:
        assert env.sensor.point_count == 512
        initial = env.reset(jax.random.key(7))
        assert initial.observation.measurement.points_body.shape == (1, 512, 3)
    finally:
        env.sim.close()


def test_measurement_keeps_only_data_needed_after_capture():
    """Per-ray directions and identical snapshot times are reconstructible, not state."""
    assert tuple(
        measure(
            Scene("empty"),
            SensorConfig.mid360(points_per_frame=8),
            jnp.zeros(3),
            jnp.array([0.0, 0.0, 0.0, 1.0]),
            0.0,
        )._fields
    ) == ("values", "mask", "points_body", "acquisition_time", "available_time")


def test_snapshot_measurement_has_one_acquisition_time():
    """All rays belong to the declared frame time, including misses."""
    config = SensorConfig.mid360(points_per_frame=32, latency=0.04)
    times = jnp.array([0.1, 0.2])
    result = jax.jit(
        lambda t: measure(Scene("empty"), config, jnp.zeros((2, 3)), jnp.array([0.0, 0, 0, 1.0]), t)
    )(times)
    encoded = SensorObservation(
        Scene("empty"), config, physics_hz=500, observation_config={"lidar": {"point_count": 32}}
    ).encode(SimpleNamespace(measurement=result))
    np.testing.assert_array_equal(
        encoded["point_times"], np.broadcast_to(np.asarray(times)[:, None], (2, 32))
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


@pytest.mark.parametrize("raw_frames", [1, 3])
def test_fan_history_consumes_snapshots_only_for_active_worlds(raw_frames):
    """Fan history follows the snapshot clock and does not advance inactive worlds."""
    sensor = SensorObservation(
        Scene("S01"),
        SensorConfig.mid360(points_per_frame=32),
        physics_hz=500,
        observation_config={
            "lidar": {"point_count": 32},
            "fan_lidar": {"raw_history_frames": raw_frames},
        },
    )

    def physics(tick):
        return SimpleNamespace(
            states=SimpleNamespace(
                pos=jnp.tile(jnp.array([[[2.0, 0, 3.0]]]), (2, 1, 1)),
                quat=jnp.tile(jnp.array([[[0.0, 0, 0, 1.0]]]), (2, 1, 1)),
            ),
            core=SimpleNamespace(
                steps=jnp.full((2, 1), tick), freq=500, rng_key=jax.random.PRNGKey(1)
            ),
        )

    initial = sensor.reset(physics(0))
    updated = jax.jit(lambda state: sensor.update(state, physics(50), jnp.array([True, False])))(
        initial
    )
    np.testing.assert_array_equal(updated.frame, [1, 0])
    encoded = sensor.encode(updated)
    assert encoded["fan_map"].shape == (2, 36, 36, 1)
    assert np.isfinite(encoded["fan_map"]).all()
    np.testing.assert_array_equal(encoded["fan_map"][1], 1)
    if raw_frames > 1:
        np.testing.assert_allclose(
            updated.paper_fan_history["points_world"][0, 0],
            jnp.where(
                updated.measurement.mask[0, :, None],
                updated.measurement.points_body[0] + jnp.array([2.0, 0, 3.0]),
                0,
            ),
        )
        np.testing.assert_array_equal(updated.paper_fan_history["mask"][1], False)


def test_empty_depth_does_not_turn_range_roundoff_into_hits():
    """A miss stays invalid when radial range is converted to axial depth."""
    config = SensorConfig.d435i(width=64, height=48, max_range=10)
    result = measure(Scene("empty"), config, jnp.zeros(3), jnp.array([0.0, 0, 0, 1]), 0.0)
    assert not np.asarray(result.mask).any()
    np.testing.assert_array_equal(result.values, 0)


def test_trainer_records_sensor_semantics_and_source_hash():
    """A checkpoint identifies the snapshot model, scan table and observation budget."""
    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.environment import make_environment

    env = make_environment(
        task="navigation",
        scene="S01",
        sensor="lidar",
        observation_config={"lidar": {"point_count": 32}},
        sensor_config={"points_per_frame": 32},
        action={"level": "acceleration"},
    )
    trainer = Trainer(env, kind="lidar", algorithm="bptt", loss="liu")
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
    np.testing.assert_array_equal(result.measurement.acquisition_time, acquired)
    derivative = jax.jit(jax.grad(lambda x: advance(x).measurement.values.sum()))(0.0)
    assert float(derivative) == 0
    assert np.linalg.norm(jax.grad(scene.clearance)(jnp.array([0.0, 0, 0]))) > 0

    frozen = sensor.update(result, physics(81), jnp.array([False]))
    for before, after in zip(jax.tree.leaves(result), jax.tree.leaves(frozen), strict=True):
        np.testing.assert_array_equal(before, after)
