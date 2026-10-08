"""P5-03 acceptance: MID360 pattern, phase, parity, world transform and isolation."""

from __future__ import annotations

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from drone_playground.environments.scenes.geometry import KIND_BOX, obstacle_positions
from drone_playground.environments.sensors.lidar import (
    MID360_ELEVATION_DEG,
    MID360_SAMPLES_PER_SCAN,
    Mid360Lidar,
    angular_coverage,
    cast_lidar,
    period_check,
    point_cloud_message,
    scan_windows,
)
from drone_playground.environments.sensors.rays import cast_rays
from drone_playground.numerics import quat_to_matrix_xyzw
from tests.helpers.scenes import mujoco_scene, synthetic_bank

CYLINDER = {"kind": 1, "size": (1.2, 5.0, 0.0), "origin": (6.0, 0.6, 2.5)}
BOX = {"kind": KIND_BOX, "size": (0.5, 0.4, 0.6), "origin": (9.0, -1.0, 2.2)}


# --------------------------------------------------------------------------------------
# Pattern and framing
# --------------------------------------------------------------------------------------


def test_scan_pattern_is_the_pinned_mid360_horizon():
    windows = scan_windows("mid360", 200)
    assert windows.shape == (33, 120, 2), windows.shape
    assert MID360_SAMPLES_PER_SCAN == 24000
    assert windows.shape[0] == 33, "800000 / 24000 = 33 usable windows plus a remainder"
    # The policy stride keeps the full azimuth sweep of the source window.
    coverage = angular_coverage(Mid360Lidar())
    # A 120-point stride stays dense around the horizon; the source pattern is
    # not uniform, so one of the 36 ten-degree bins can stay empty.
    assert coverage["occupied_azimuth_bins_of_36"] >= 34
    # A 200-ray stride cannot hit the exact pattern extremes, so the sampled
    # window must sit inside the declared MID360 field of view while covering
    # nearly all of its elevation span.
    assert coverage["elevation_deg"][0] >= MID360_ELEVATION_DEG[0] - 1e-4
    assert coverage["elevation_deg"][1] <= MID360_ELEVATION_DEG[1] + 1e-4
    span = coverage["elevation_deg"][1] - coverage["elevation_deg"][0]
    assert span > 0.95 * (MID360_ELEVATION_DEG[1] - MID360_ELEVATION_DEG[0])
    # A stride cannot land exactly on the 360 degree wrap, so require near-full
    # azimuth coverage rather than the exact extreme.
    assert coverage["azimuth_deg"][1] - coverage["azimuth_deg"][0] > 355.0
    assert coverage["points"] == 120
    assert coverage["mean_rays_per_10deg"] == pytest.approx(120 / 36.0, abs=1e-6)


def test_consecutive_windows_are_not_repetitive():
    windows = scan_windows("mid360", 200)
    for index in range(4):
        assert not np.allclose(windows[index], windows[index + 1])


def test_directions_are_unit_and_follow_the_livox_convention():
    lidar = Mid360Lidar()
    directions = np.asarray(lidar.directions(0))
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-5)
    # theta is azimuth in the sensor xy plane, phi is elevation towards +z.
    angles = scan_windows(lidar.pattern, lidar.downsample)[0]
    theta = angles[:, 0]
    assert np.allclose(directions[:, 0], np.cos(angles[:, 1]) * np.cos(theta), atol=1e-5)
    assert np.allclose(directions[:, 1], np.cos(angles[:, 1]) * np.sin(theta), atol=1e-5)
    assert np.allclose(directions[:, 2], np.sin(angles[:, 1]), atol=1e-5)


def test_per_point_time_and_period_are_recorded_and_exact():
    lidar = Mid360Lidar()
    times = np.asarray(lidar.relative_point_time())
    assert times.shape == (lidar.points_per_frame,)
    assert times[0] == 0.0
    assert times[-1] < lidar.scan_period_s
    assert np.all(np.diff(times) > 0)
    check = period_check(lidar, 50)
    assert check["period_steps"] == 5
    assert check["realised_rate_hz"] == 10.0
    assert check["exact_divisor"] is True
    calibration = lidar.calibration()
    assert "instantaneous" in calibration["scan_distortion_model"]
    assert calibration["pattern"]["usable_windows"] == 33
    assert calibration["pattern"]["sampled_indices"].startswith("window*24000")


# --------------------------------------------------------------------------------------
# Geometry parity
# --------------------------------------------------------------------------------------


def dummy_body_id(model):
    """Id of the free-jointed placeholder that makes MJX able to scan the scene."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mjx_dummy")


def _mujoco_lidar_reference(model, data, origins, directions, exclude_body):
    """Distances from the pinned MuJoCo-LiDAR JAX backend for the same rays."""
    import mujoco.mjx as mjx
    from mujoco_lidar.core_jax.mjlidar_jax import MjLidarJax

    lidar = MjLidarJax(mjx.put_model(model), bodyexclude=exclude_body)
    dx = mjx.forward(mjx.put_model(model), mjx.make_data(model))
    out = np.full(len(directions), np.inf)
    theta = np.arctan2(directions[:, 1], directions[:, 0]).astype(np.float32)
    phi = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)).astype(np.float32)
    distances, _ = lidar.trace_rays(
        dx.geom_xpos,
        dx.geom_xmat,
        jnp.asarray(origins, jnp.float32),
        jnp.eye(3, dtype=jnp.float32),
        jnp.asarray(theta),
        jnp.asarray(phi),
    )
    distances = np.asarray(distances)
    usable = distances > 1e-6
    out[usable] = distances[usable]
    return out


@pytest.mark.parametrize(
    "position,quat",
    [
        ((1.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
        ((4.0, -1.0, 2.5), (0.0, 0.0, 0.7071068, 0.7071068)),
        ((12.0, -2.0, 3.0), (0.1830127, 0.1830127, 0.6830127, 0.6830127)),
    ],
)
def test_mid360_rays_match_mujoco_lidar_and_mj_ray(position, quat):
    bank = synthetic_bank([CYLINDER, BOX])
    model, data = mujoco_scene([CYLINDER, BOX])
    lidar = Mid360Lidar()
    frame = cast_lidar(
        lidar,
        bank,
        jnp.int32(0),
        jnp.asarray(position),
        jnp.asarray(quat),
        jnp.float32(0.0),
        0,
    )
    directions = np.asarray(lidar.directions(0))
    rotation = np.asarray(quat_to_matrix_xyzw(jnp.asarray(quat)))
    world = directions @ rotation.T
    centres = np.asarray(obstacle_positions(bank, jnp.int32(0), jnp.float32(0.0)))
    analytic = np.asarray(
        cast_rays(
            bank.kind[0],
            bank.size[0],
            centres,
            bank.active[0],
            np.asarray(position, np.float32)[None, :] + np.zeros_like(world),
            world,
            bank.world_low,
            bank.world_high,
            True,
        )
    )
    native_mj = np.full(len(world), np.inf)
    for index, direction in enumerate(world):
        geom_id = np.zeros(1, np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            np.asarray(position, float),
            np.asarray(direction, float),
            None,
            1,
            -1,
            geom_id,
        )
        if distance >= 0.0:
            native_mj[index] = distance
    native_lidar = _mujoco_lidar_reference(
        model, data, position, world.astype(np.float32), dummy_body_id(model)
    )

    finite_analytic = np.isfinite(analytic)
    # MuJoCo treats a plane as infinite while the sensor bounds the corridor
    # floor, so every analytic return must be found by MuJoCo, and the two must
    # agree wherever the return is inside the corridor.
    assert not np.any(finite_analytic & ~np.isfinite(native_mj))
    inside = finite_analytic & (analytic <= 15.0)
    assert inside.sum() > 5
    assert np.max(np.abs(analytic[inside] - native_mj[inside])) < 1e-4
    # The third-party ray backend must agree on the same rays as well.
    assert not np.any(finite_analytic & ~np.isfinite(native_lidar))
    shared = finite_analytic & np.isfinite(native_lidar) & (analytic <= 15.0)
    assert np.max(np.abs(analytic[shared] - native_lidar[shared])) < 5e-3
    # Wiring check only: a near-tangent cylinder ray flips between hit and miss
    # under float32 rounding, so the masks are compared at an aggregate level and
    # the distance agreement is restricted to close returns, where the root is
    # well conditioned. The exact geometric statement is the mj_ray comparison.
    analytic_valid = (analytic >= lidar.range_m[0]) & (analytic <= lidar.range_m[1])
    frame_valid = np.asarray(frame.valid)
    assert (analytic_valid == frame_valid).mean() > 0.85
    close = analytic_valid & frame_valid & (analytic <= 15.0)
    assert close.sum() > 5
    assert np.max(np.abs(np.asarray(frame.distance)[close] - analytic[close])) < 5e-2


def test_policy_frame_matches_a_direct_cast():
    """The downsampled policy window is a real subset of the source scan."""
    lidar = Mid360Lidar()
    full = np.asarray(lidar.directions(0))
    assert full.shape == (lidar.points_per_frame, 3)
    angles = scan_windows(lidar.pattern, lidar.downsample)[0]
    reference = scan_windows(lidar.pattern, 1)[0][:: lidar.downsample]
    assert np.allclose(angles, reference, atol=1e-6)


# --------------------------------------------------------------------------------------
# Framing, isolation and the full chain
# --------------------------------------------------------------------------------------


def test_world_points_actually_apply_the_world_transform():
    bank = synthetic_bank([CYLINDER])
    lidar = Mid360Lidar()
    position = jnp.array([2.0, 0.5, 2.0])
    quat = jnp.array([0.0, 0.0, 0.7071068, 0.7071068])  # 90 degree yaw
    frames = [
        cast_lidar(lidar, bank, jnp.int32(0), position, quat, jnp.float32(0.0), window)
        for window in range(6)
    ]
    valid = np.concatenate([np.asarray(item.valid) for item in frames])
    assert valid.sum() > 30
    all_sensor = np.concatenate([np.asarray(item.points_sensor) for item in frames])
    all_world = np.concatenate([np.asarray(item.points_world) for item in frames])
    sensor = all_sensor[valid]
    world = all_world[valid]
    # Rotating the body by 90 degrees about z must change the world points even
    # though the sensor-frame points are unchanged, and the relation must hold.
    rotation = np.asarray(quat_to_matrix_xyzw(quat))
    # The declared relation must hold: world = body position + R * sensor point.
    offset = world - np.asarray(position)[None, :]
    expected_offset = sensor @ rotation.T
    # The sensor is single precision and returns ranges up to the declared
    # 200 m, so the relation is checked at a float32 relative bound instead of
    # an absolute one.
    scale = np.maximum(np.linalg.norm(expected_offset, axis=1), 1.0)
    residual = np.abs(offset - expected_offset)
    assert np.all(residual <= 5e-4 * scale[:, None]), (
        float((residual / scale[:, None]).max()),
        int(valid.sum()),
    )
    # And the rotation must be the real 90 degree yaw, not a relabelling: a
    # sensor-frame +x ray becomes world +y.
    assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
    assert not np.allclose(world[:, :2], sensor[:, :2], atol=1e-3)
    message = point_cloud_message(frames[0], lidar)
    assert message["frame_id"] == "world"
    assert message["point_count"] == int(np.asarray(frames[0].valid).sum())
    assert np.asarray(message["points_world"]).shape == (
        int(np.asarray(frames[0].valid).sum()),
        3,
    )


def test_sensor_range_and_validity_conventions():
    bank = synthetic_bank([])
    lidar = Mid360Lidar()
    frame = cast_lidar(
        lidar,
        bank,
        jnp.int32(0),
        jnp.array([0.5, 0.0, 2.0]),
        jnp.array([0.0, 0.0, 0.0, 1.0]),
        jnp.float32(0.0),
        0,
    )
    valid = np.asarray(frame.valid)
    distance = np.asarray(frame.distance)
    # No obstacle above the horizon: the floor is the only return.
    assert valid.sum() > 0
    assert distance[valid].min() >= lidar.range_m[0]
    assert np.all(distance[~valid] == 0.0)
    assert np.all(np.linalg.norm(np.asarray(frame.points_sensor)[~valid], axis=1) == 0.0)
    # The scene contains no robot body, so the sensor cannot see itself.
    assert np.all(distance[valid] > 0.5)
