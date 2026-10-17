"""Coordinate and timestamp contracts for the new egocentric OA training defaults."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.methods import Action
from drone_playground.simulation.observation import SensorObservation
from drone_playground.simulation.procedural_scene import ProceduralScene
from drone_playground.simulation.scene import Scene
from drone_playground.simulation.sensors import Measurement, SensorConfig


def resolved(experiment):
    """Resolve the public recipe through the real Hydra config graph."""
    cfgroot = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(cfgroot)):
        return compose(config_name="config", overrides=[f"experiment={experiment}"])


@pytest.mark.parametrize("region", ("static", "dynamic"))
@pytest.mark.parametrize("sensor", ("depth", "lidar"))
@pytest.mark.parametrize("algorithm", ("bptt", "shac", "ppo"))
def test_oa_native_training_defaults_share_body_action_without_global_rotation(
    region, sensor, algorithm
):
    """All 12 algorithm/modalities OA public recipes use egocentric acceleration."""
    config = resolved(f"oa_{region}_{sensor}_{algorithm}")
    assert config.method.action.level == "acceleration"
    assert config.method.action.frame == "body"
    assert config.observation.navigation_world_rotation is False
    assert config.simulation.procedural.horizontal_beam_probability == 0.7
    assert config.simulation.procedural.obstacle_motion == region
    assert config.observation[sensor].motion_compensate is True
    assert config.initial_checkpoint is None and config.resume is None


@pytest.mark.parametrize("region", ("static", "dynamic"))
@pytest.mark.parametrize("sensor", ("depth", "lidar"))
@pytest.mark.parametrize("algorithm", ("bptt", "shac"))
@pytest.mark.parametrize(
    "backward,level", (("lotf", "body_rate"), ("point_mass_lag", "acceleration"))
)
def test_oa_backward_variants_keep_consistent_action_frames(
    region, sensor, algorithm, backward, level
):
    """All 16 model-based OA extensions avoid world-action/body-observation mismatch."""
    cfgroot = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(cfgroot)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment=oa_{region}_{sensor}_{algorithm}",
                f"learning.backward_model={backward}",
                f"method.action.level={level}",
            ],
        )
    assert cfg.method.action.frame == "body"
    assert Action(level=level, frame=cfg.method.action.frame).contract["frame"] == "body"


@pytest.mark.parametrize("task", ("tracking", "racing"))
def test_accepted_reference_tasks_retain_their_world_action_contract(task):
    """A shared default migration must not silently reinterpret accepted models."""
    cfg = resolved(f"{task}_bptt_native")
    assert cfg.method.action.get("frame", "world") == "world"
    assert cfg.observation.get("navigation_world_rotation", False) is False
    assert (
        cfg.simulation.procedural.get("horizontal_beam_probability", 0) == 0
        if "procedural" in cfg.simulation
        else True
    )


@pytest.mark.parametrize("sensor", ("depth", "lidar"))
def test_legacy_velocity_navigation_recipe_converts_body_velocity_in_policy(sensor):
    """Velocity-yaw navigation is not allowed to silently use world velocity."""
    cfg = resolved(f"navigation_procedural_{sensor}_ppo_velocity")
    assert cfg.method.action.level == "velocity_yaw"
    assert cfg.method.action.frame == "body"
    legacy = Action(level="velocity_yaw", frame="world")
    corrected = Action(level="velocity_yaw", frame="body")
    assert corrected.contract["frame"] == "body"
    assert legacy.contract["frame"] == "world"


def test_body_velocity_yaw_is_converted_once_at_issuing_pose():
    """Body x velocity at yaw 90° must request world y velocity from the PID."""
    from drone_playground.simulation.environment import make_environment

    env = make_environment(num_envs=1)
    initial = env.reset(jax.random.key(51))
    attitude = Rotation.from_euler("z", jnp.array([[90.0]]), degrees=True)
    physics = initial.physics.replace(
        states=initial.physics.states.replace(
            quat=attitude.as_quat()[:, None],
            vel=jnp.zeros_like(initial.physics.states.vel),
        )
    )
    body = Action(level="velocity_yaw", frame="body")
    world = Action(level="velocity_yaw", frame="world")
    reference = jnp.array([[2.0, 0.0, 0.0, 0.0]])
    rotated = jnp.array([[0.0, 2.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        body.setpoint(physics, reference).value,
        world.setpoint(physics, rotated).value,
        atol=1e-5,
        rtol=1e-5,
    )
    assert not np.allclose(
        body.setpoint(physics, reference).value, world.setpoint(physics, reference).value
    )
    env.sim.close()


def fake_physics(position, quaternion):
    """A minimal batch pose with the simulator's native world/body quaternion layout."""
    return SimpleNamespace(
        states=SimpleNamespace(pos=position[:, None], quat=quaternion[:, None]),
        core=SimpleNamespace(steps=jnp.array([[100]]), freq=500),
    )


def test_lidar_acquired_at_past_pose_is_converted_to_current_body_coordinates():
    """A motion-delayed point must be rotated and translated, not relabeled."""
    sensor = SensorObservation(
        Scene("empty"),
        SensorConfig.mid360(points_per_frame=3),
        500,
        observation_config={"lidar": {"point_count": 3, "motion_compensate": True}},
    )
    points = jnp.array([[[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])
    measurement = Measurement(
        jnp.zeros((1, 3)),
        jnp.array([[True, True, False]]),
        points,
        jnp.array([0.10]),
        jnp.array([0.14]),
    )
    capture_pose = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    state = SimpleNamespace(measurement=measurement, acquisition_pose=capture_pose)
    current = fake_physics(
        jnp.array([[1.0, 0.0, 0.0]]),
        Rotation.from_euler("z", jnp.array([[90.0]]), degrees=True).as_quat(),
    )
    output = sensor.encode(state, physics=current)
    np.testing.assert_allclose(
        output["points"][0, :2], [[0.0, -4.0, 0.0], [4.0, 1.0, 0.0]], atol=1e-6
    )
    np.testing.assert_array_equal(output["points"][0, 2], 0)
    np.testing.assert_array_equal(state.measurement.points_body, points)
    np.testing.assert_allclose(output["point_times"], [[0.10, 0.10, 0.10]], atol=1e-7)
    assert sensor.specification["motion_compensate"] == "snapshot-to-current-body-pose"
    with pytest.raises(ValueError, match="current physics pose"):
        sensor.encode(state)


def test_depth_reprojection_uses_current_camera_pose_and_preserves_raw_snapshot():
    """A known past pixel moves to its current pixel without synthesizing unknown surfaces."""
    cfg = SensorConfig.d435i(
        width=3, height=1, min_range=0.2, max_range=10.0, horizontal_fov_deg=90, vertical_fov_deg=60
    )
    sensor = SensorObservation(
        Scene("empty"),
        cfg,
        500,
        observation_config={"depth": {"motion_compensate": True, "representation": "metric"}},
    )
    points = jnp.array([[[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    mask = jnp.array([[[False, True, False]]])
    measurement = Measurement(
        jnp.array([[[0.0, 4.0, 0.0]]]), mask, points, jnp.array([0.1]), jnp.array([0.14])
    )
    state = SimpleNamespace(
        measurement=measurement, acquisition_pose=jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    )
    now = fake_physics(jnp.array([[0.0, 2.0, 0.0]]), jnp.array([[0.0, 0.0, 0.0, 1.0]]))
    encoded = sensor.encode(state, physics=now)
    np.testing.assert_array_equal(encoded["depth_mask"], [[[False, False, True]]])
    np.testing.assert_allclose(encoded["depth"][0, 0, 2, 0], 4.0, atol=1e-6)
    np.testing.assert_array_equal(state.measurement.values, measurement.values)
    same_pose = fake_physics(jnp.zeros((1, 3)), jnp.array([[0.0, 0.0, 0.0, 1.0]]))
    same = sensor.encode(state, physics=same_pose)
    np.testing.assert_array_equal(same["depth_mask"], mask)
    np.testing.assert_allclose(same["depth"][0, 0, 1, 0], 4.0)


@pytest.mark.parametrize("kind", ("depth", "lidar"))
def test_motion_compensation_requires_boolean_configuration(kind):
    """No truthy strings or numbers silently change the sensor reference frame."""
    config = (
        SensorConfig.d435i(width=4, height=4)
        if kind == "depth"
        else SensorConfig.mid360(points_per_frame=4)
    )
    options = {kind: {"motion_compensate": "yes"}}
    if kind == "lidar":
        options[kind]["point_count"] = 4
    with pytest.raises(ValueError, match="motion_compensate must be boolean"):
        SensorObservation(Scene("empty"), config, 500, observation_config=options)


def test_default_static_and_dynamic_beam_generators_have_real_3d_passages():
    """OA geometry samples variable overhead beams rather than leaking S06 MJCF."""
    for motion in ("static", "dynamic"):
        c = resolved(f"oa_{motion}_depth_shac")
        distribution = c.simulation.procedural
        scene = ProceduralScene({k: v for k, v in distribution.items()})
        state = scene.sample(jax.random.key(27), 8)
        assert scene.distribution.horizontal_beam_probability == 0.7
        assert np.asarray(state.active[:, :2]).any()
        assert np.any(np.asarray(state.sizes[:, :2, 1]) > 15)
        assert Scene("S06").geometry_hash != scene.geometry_hash
