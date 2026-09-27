"""P5-02 acceptance: D435 calibration, optical geometry, parity, timing, occlusion.

The depth sensor is validated against ``mujoco.mj_ray`` on the same primitives,
because the analytic batch caster replaced the unavailable GL and MJX render
backends and therefore carries the whole verification burden.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from drone_playground.composition import compose_config, validate_config
from drone_playground.tasks.observations import NavigationSensorObservation
from drone_playground.tasks.scenes.navigation import KIND_BOX, KIND_CYLINDER, obstacle_positions
from drone_playground.tasks.sensors.depth import (
    BODY_FROM_OPTICAL,
    DepthCamera,
    cast_depth,
    sensor_pose,
)
from drone_playground.tasks.sensors.rays import cast_rays

LOW = np.array([0.0, -5.0, 0.0], np.float32)
HIGH = np.array([20.0, 5.0, 5.0], np.float32)


def synthetic_bank(obstacles, capacity=4, start=(0.5, 0.0, 2.0), goal=(15.5, 0.0, 2.0)):
    from drone_playground.tasks.scenes.navigation import MOTION_STATIC, SceneBank

    kind = np.zeros((1, capacity), np.int32)
    size = np.zeros((1, capacity, 3), np.float32)
    origin = np.zeros((1, capacity, 3), np.float32)
    motion = np.zeros((1, capacity), np.int32)
    params = np.zeros((1, capacity, 5), np.float32)
    active = np.zeros((1, capacity), bool)
    for index, obstacle in enumerate(obstacles):
        kind[0, index] = obstacle["kind"]
        size[0, index] = obstacle["size"]
        origin[0, index] = obstacle["origin"]
        motion[0, index] = obstacle.get("motion", MOTION_STATIC)
        params[0, index] = obstacle.get("params", (0.0,) * 5)
        active[0, index] = True
    return SceneBank(
        kind=jnp.asarray(kind),
        size=jnp.asarray(size),
        origin=jnp.asarray(origin),
        motion=jnp.asarray(motion),
        params=jnp.asarray(params),
        active=jnp.asarray(active),
        start=jnp.asarray(np.asarray([start], np.float32)),
        goal=jnp.asarray(np.asarray([goal], np.float32)),
        difficulty=jnp.zeros((1,), jnp.int32),
        subtype=jnp.zeros((1,), jnp.int32),
        world_low=jnp.asarray(LOW),
        world_high=jnp.asarray(HIGH),
        subtype_names=("synthetic",),
    )


def mujoco_scene(obstacles):
    """MuJoCo model with the same primitives and the same bounded ground plane."""
    bodies = []
    for index, obstacle in enumerate(obstacles):
        x, y, z = obstacle["origin"]
        if obstacle["kind"] == KIND_CYLINDER:
            radius, height = obstacle["size"][0], obstacle["size"][1]
            geometry = f'<geom type="cylinder" size="{radius} {height / 2.0}"/>'
        else:
            hx, hy, hz = obstacle["size"]
            geometry = f'<geom type="box" size="{hx} {hy} {hz}"/>'
        bodies.append(f'<body pos="{x} {y} {z}">{geometry}</body>')
    xml = (
        '<mujoco model="depth-parity"><compiler angle="radian"/><option timestep="0.002"/>'
        # MuJoCo bounds plane ray casting by geom size, so the reference floor
        # must cover the same 100 m margin the analytic floor uses. Dynamic
        # collision treats a plane as infinite, ray casting does not.
        '<worldbody><geom name="ground" type="plane" size="110 105 .1" pos="10 0 0"/>'
        # MJX refuses to scan a model with zero degrees of freedom, so a
        # massless placeholder carries the free joint and is excluded from rays.
        '<body name="mjx_dummy" pos="500 500 500"><freejoint/>'
        '<geom name="mjx_dummy_geom" type="sphere" size="0.01" mass="0.001"/>'
        "</body>"
        + "".join(bodies)
        + "</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


CYLINDER = {"kind": 1, "size": (1.2, 5.0, 0.0), "origin": (6.0, 0.6, 2.5)}
BOX = {"kind": 2, "size": (0.5, 0.4, 0.6), "origin": (9.0, -1.0, 2.2)}


# --------------------------------------------------------------------------------------
# Calibration and optical geometry
# --------------------------------------------------------------------------------------


def test_intrinsics_follow_the_pinned_d435_configuration():
    camera = DepthCamera()
    # 85.2 degrees is the rounded SANDO value, so the focal length lands just
    # below 120 / (2 tan(42.6 deg)).
    assert camera.focal_px == pytest.approx(65.25, abs=1e-3)
    assert camera.principal_point_px == (60.0, 45.0)
    assert camera.vertical_fov_deg == pytest.approx(69.185, abs=1e-3)
    calibration = camera.calibration()
    assert calibration["intrinsics"]["depth_units"] == "metres, axial (optical z)"
    assert calibration["source_availability_hz"] == 30.0
    assert calibration["policy_grid"] == [20, 15]
    # Increasing the field of view must reduce the focal length.
    wider = DepthCamera(horizontal_fov_deg=120.0)
    assert wider.focal_px < camera.focal_px


def test_optical_directions_are_axial_and_centred():
    camera = DepthCamera()
    directions = camera.optical_directions(1)
    assert directions.shape == (120 * 90, 3)
    assert np.allclose(np.asarray(directions[:, 2]), 1.0)
    # An even resolution has no pixel exactly on the axis, so the nearest pixel
    # to the principal point is within half a pixel of it.
    centre = np.asarray(directions).reshape(120, 90, 3)[60, 45]
    # MuJoCo/ROS optical convention: +z forward, +x right, +y down.
    assert abs(centre[0]) < 0.5 / camera.focal_px + 1e-6
    assert abs(centre[1]) < 0.5 / camera.focal_px + 1e-6
    assert centre[2] == 1.0
    grid = np.asarray(directions).reshape(120, 90, 3)
    assert grid[0, 45, 0] < 0 < grid[-1, 45, 0]
    assert grid[60, 0, 1] < 0 < grid[60, -1, 1]


def test_sensor_pose_mounts_the_optical_frame_on_the_body_forward_axis():
    camera = DepthCamera()
    origin, rotation = sensor_pose(camera, jnp.zeros(3), jnp.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(np.asarray(origin), [0.05, 0.0175, 0.0125], atol=1e-6)
    # A level body looks along +x; right is -y; down is -z.
    forward = np.asarray(rotation) @ np.array([0.0, 0.0, 1.0])
    right = np.asarray(rotation) @ np.array([1.0, 0.0, 0.0])
    down = np.asarray(rotation) @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(forward, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(right, [0.0, -1.0, 0.0], atol=1e-6)
    assert np.allclose(down, [0.0, 0.0, -1.0], atol=1e-6)
    # The stored body-from-optical rotation is orthonormal and right handed.
    matrix = np.asarray(BODY_FROM_OPTICAL)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(matrix) == pytest.approx(1.0)

    # A quarter turn about the body z axis (xyzw) must carry the look axis to +y.
    yawed, yawed_rotation = sensor_pose(
        camera, jnp.zeros(3), jnp.array([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    )
    assert np.allclose(np.asarray(yawed_rotation) @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0], atol=1e-6)
    assert not np.allclose(np.asarray(yawed), np.asarray(origin))


# --------------------------------------------------------------------------------------
# Analytic ray casting against native MuJoCo
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pose,quat",
    [
        ((1.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
        ((1.0, 0.0, 2.0), (0.3826834, 0.0, 0.0, 0.9238795)),
        ((3.0, -2.0, 3.0), (0.0, 0.0, 0.7071068, 0.7071068)),
        ((5.0, 0.5, 1.5), (0.0, 0.258819, 0.0, 0.9659258)),
        ((8.0, 0.0, 2.5), (0.1830127, 0.1830127, 0.6830127, 0.6830127)),
    ],
)
def test_depth_rays_match_native_mujoco(pose, quat):
    """Every pixel must agree with mujoco.mj_ray on the same geometry."""
    bank = synthetic_bank([CYLINDER, BOX])
    model, data = mujoco_scene([CYLINDER, BOX])
    camera = DepthCamera()
    origin, rotation = sensor_pose(camera, jnp.asarray(pose), jnp.asarray(quat))
    origin = np.asarray(origin)
    optical = np.asarray(camera.optical_directions(6))
    directions = optical @ np.asarray(rotation).T
    centres = obstacle_positions(bank, jnp.int32(0), jnp.float32(0.0))
    analytic = np.asarray(
        cast_rays(
            bank.kind[0],
            bank.size[0],
            np.asarray(centres),
            bank.active[0],
            origin[None, :] + np.zeros_like(directions),
            directions,
            bank.world_low,
            bank.world_high,
            True,
        )
    )
    native = np.full(len(directions), np.inf)
    for index, direction in enumerate(directions):
        geom_id = np.zeros(1, np.int32)
        distance = mujoco.mj_ray(
            model, data, origin.astype(float), np.asarray(direction, float), None, 1, -1, geom_id
        )
        if distance >= 0.0:
            native[index] = distance

    hit_analytic = np.isfinite(analytic)
    hit_native = np.isfinite(native)
    assert np.array_equal(hit_analytic, hit_native), (
        f"{int((hit_analytic & ~hit_native).sum())} analytic-only and "
        f"{int((~hit_analytic & hit_native).sum())} native-only rays"
    )
    assert hit_analytic.sum() > 20
    # mj_ray does not normalise its direction, so the parameters are comparable.
    assert np.max(np.abs(analytic[hit_analytic] - native[hit_analytic])) < 1e-4


def test_occlusion_and_ordering_prefer_the_nearer_surface():
    near = {"kind": 2, "size": (0.5, 0.5, 0.5), "origin": (5.0, 0.0, 2.0)}
    far = {"kind": 2, "size": (0.5, 0.5, 0.5), "origin": (8.0, 0.0, 2.0)}
    camera = DepthCamera()
    frame = cast_depth(
        camera,
        synthetic_bank([far, near]),
        jnp.int32(0),
        jnp.array([0.5, 0.0, 2.0]),
        jnp.array([0.0, 0.0, 0.0, 1.0]),
        jnp.float32(0.0),
    )
    centre = frame.depth.reshape(20, 15)[10, 7]
    # The optical frame sits 0.05 m ahead of the body origin, so the near box
    # face at x = 4.5 is about 3.95 m away along the optical axis.
    assert float(centre) == pytest.approx(4.5 - 0.55, abs=0.03)
    frame_swapped = cast_depth(
        camera,
        synthetic_bank([near, far]),
        jnp.int32(0),
        jnp.array([0.5, 0.0, 2.0]),
        jnp.array([0.0, 0.0, 0.0, 1.0]),
        jnp.float32(0.0),
    )
    assert np.array_equal(np.asarray(frame.depth), np.asarray(frame_swapped.depth))


def test_range_clipping_marks_far_and_missing_pixels_invalid():
    camera = DepthCamera(far_m=5.0)
    frame = cast_depth(
        camera,
        synthetic_bank([]),
        jnp.int32(0),
        jnp.array([0.5, 0.0, 2.0]),
        jnp.array([0.0, 0.0, 0.0, 1.0]),
        jnp.float32(0.0),
    )
    # No obstacle: only the floor returns, and only inside the 2 m range.
    assert bool(jnp.any(frame.valid))
    assert bool(jnp.any(~frame.valid))
    assert float(jnp.max(frame.depth)) <= 5.0
    assert float(jnp.min(jnp.where(frame.valid, frame.depth, 1e9))) >= 0.1
    assert float(jnp.max(jnp.where(frame.valid, 0.0, frame.depth))) == 0.0
    # Straight ahead there is nothing within range, so the centre pixel is empty.
    assert not bool(frame.valid.reshape(20, 15)[10, 7])


def test_moving_obstacle_is_rendered_where_it_actually_is():
    from drone_playground.tasks.scenes.navigation import (
        MOTION_BOUNCE,
        obstacle_positions,
    )

    # The cube slides along the optical axis, so the same pixel must report a
    # depth that differs by exactly the travelled distance.
    crossing = {
        "kind": KIND_BOX,
        "size": (0.4, 0.4, 0.4),
        "origin": (6.0, 0.0, 2.0),
        "motion": MOTION_BOUNCE,
        "params": (-1.5, 0.0, 0.0, 8.0, 0.0),
    }
    bank = synthetic_bank([crossing])
    camera = DepthCamera()
    pose = (jnp.array([0.5, 0.0, 2.0]), jnp.array([0.0, 0.0, 0.0, 1.0]))
    early = cast_depth(camera, bank, jnp.int32(0), *pose, jnp.float32(0.0))
    late = cast_depth(camera, bank, jnp.int32(0), *pose, jnp.float32(4.0))
    centre = (10, 7)
    early_depth = float(early.depth.reshape(20, 15)[centre])
    late_depth = float(late.depth.reshape(20, 15)[centre])
    assert early_depth > 0.0 and late_depth > 0.0
    assert early_depth - late_depth == pytest.approx(3.0, abs=0.05)
    moved = np.asarray(obstacle_positions(bank, jnp.int32(0), jnp.float32(4.0)))[0]
    assert float(moved[0]) == pytest.approx(4.5, abs=1e-5)


# --------------------------------------------------------------------------------------
# Framing, timing and the full chain
# --------------------------------------------------------------------------------------


def test_depth_observation_encodes_range_validity_and_history():
    observation = NavigationSensorObservation(history=2, points_per_frame=3, channels=2)
    values = jnp.stack(
        [
            jnp.array([[0.0, 1.0, 100.0], [0.1, 10.0, 5.0]]),
            jnp.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        ],
        axis=-1,
    )
    encoded = observation.encode_sensor(values)
    assert encoded.shape == (2, 3, 2)
    assert encoded.reshape(-1).shape == (2 * 2 * 3,)
    signal = np.asarray(encoded[..., 0])
    mask = np.asarray(encoded[..., 1])
    assert np.allclose(mask, (np.asarray(values[..., 1]) > 0.5).astype(np.float32))
    # Inverse depth is one at the near plane and zero at the far plane.
    assert signal[1, 0] == pytest.approx(1.0, abs=1e-5)
    assert signal[1, 1] == pytest.approx(0.0, abs=1e-5)
    # Inverse depth is linear in 1/d, so 5 m lands near the far end.
    assert signal[1, 2] == pytest.approx((0.2 - 0.1) / (10.0 - 0.1), abs=1e-5)
    # An invalid pixel is encoded as the far plane, not as a near surface.
    assert signal[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert signal[0, 2] == pytest.approx(0.0, abs=1e-5)


def test_sensor_refresh_rate_is_an_exact_control_step_divisor():
    camera = DepthCamera()
    assert camera.period_steps(50) == 2
    assert 50 / camera.period_steps(50) == 25.0
    odd = DepthCamera(source_rate_hz=7.0)
    assert odd.period_steps(50) == 7


def test_depth_environment_closed_loop_and_frame_timestamps():
    config = compose_config("p5_static_depth_ppo")
    validate_config(config)
    from drone_playground.composition import build_environment

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
    assert state.obs.shape == (env.observation_size,)
    assert bool(jnp.all(jnp.isfinite(state.obs)))

    def body(carry, _):
        nxt = env.step(carry, env.hover_action)
        return nxt, (nxt.pipeline_state.sensor_time, nxt.pipeline_state.sensor_sequence)

    final, (times, sequences) = jax.jit(
        lambda s: jax.lax.scan(body, s, None, length=8)
    )(state)
    sequences = np.asarray(sequences).reshape(-1)
    # One new frame every two control steps, so the counter rises by one each pair.
    # Reset already produced frame one at control step zero, so the first
    # refresh after reset happens at control step two.
    assert list(sequences) == [1, 2, 2, 3, 3, 4, 4, 5]
    last_times = np.asarray(times[-1])
    assert last_times[-1] == pytest.approx(8 * env.dt, abs=1e-6)
    assert last_times[-2] == pytest.approx(6 * env.dt, abs=1e-6)
    assert last_times[-3] == pytest.approx(4 * env.dt, abs=1e-6)
    assert bool(jnp.all(jnp.isfinite(final.obs)))
    # A fresh reset must clear the history rather than leak the previous episode.
    fresh = env.reset(jax.random.PRNGKey(1), jnp.int32(0))
    assert int(fresh.pipeline_state.sensor_sequence) == 1
    assert not np.array_equal(
        np.asarray(fresh.pipeline_state.sensor_time), np.asarray(times[-1])
    )
    env.close()


def test_composition_requires_a_sensor_for_a_perception_observation():
    import copy

    config = compose_config("p5_static_depth_ppo")
    validate_config(config)
    stripped = copy.deepcopy(config)
    del stripped["observation"]["sensor"]
    with pytest.raises(ValueError, match="requires an observation.sensor"):
        validate_config(stripped)

    # A state-only observation must not smuggle in a sensor group.
    state_only = compose_config("p5_navigation_static")
    state_only["observation"]["sensor"] = {"_target_": "x.Y", "name": "z"}
    with pytest.raises(ValueError, match="must not declare a sensor"):
        validate_config(state_only)
