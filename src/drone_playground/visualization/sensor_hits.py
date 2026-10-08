"""Replay-only reconstruction of range-sensor hit points from recorded poses."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.environments.scenes.geometry import obstacle_positions
from drone_playground.environments.sensors.depth import (
    DepthCamera,
    PinholeDepthCamera,
    cast_depth,
    sensor_pose,
)
from drone_playground.environments.sensors.lidar import (
    MID360_SAMPLES_PER_SCAN,
    Mid360Lidar,
    cast_lidar,
)
from drone_playground.environments.sensors.rays import cast_rays
from drone_playground.numerics import quat_to_matrix_xyzw

REPLAY_HIT_SAMPLE_BUDGET = 2_400_000
"""Upper bound on ``frames × marker slots`` for one exported sensor overlay."""


@dataclass(frozen=True)
class ReplaySensorContext:
    """Everything needed to reconstruct one sensor overlay during export."""

    sensor: object
    bank: object
    scenario_id: int
    max_points: int = 2400


def supports_hit_overlay(sensor) -> bool:
    """Check whether the sensor supports replay of physical hit locations."""
    return isinstance(sensor, (Mid360Lidar, DepthCamera, PinholeDepthCamera))


def _mid360_downsample(max_points: int) -> int:
    if max_points < 1:
        raise ValueError("Replay sensor point budget must be positive")
    divisors = [
        d for d in range(1, MID360_SAMPLES_PER_SCAN + 1) if MID360_SAMPLES_PER_SCAN % d == 0
    ]
    return next(d for d in divisors if MID360_SAMPLES_PER_SCAN // d <= max_points)


def _grid_stride(width: int, height: int, max_points: int) -> int:
    if max_points < 1:
        raise ValueError("Replay sensor point budget must be positive")
    for stride in range(1, min(width, height) + 1):
        if width % stride or height % stride:
            continue
        if (width // stride) * (height // stride) <= max_points:
            return stride
    raise ValueError("No valid replay grid stride fits the requested point budget")


def _mid360_sampler(sensor: Mid360Lidar, max_points: int):
    display = replace(
        sensor,
        range_m=tuple(float(value) for value in sensor.range_m),
        downsample=_mid360_downsample(max_points),
        state_gradient="direct",
        calibration_id=f"{sensor.calibration_id}-replay-overlay",
    )
    key = (
        display.pattern,
        display.downsample,
        float(display.source_rate_hz),
        tuple(display.range_m),
        bool(display.include_ground),
        int(display.obstacle_batch_size),
    )
    cached = _MID360_SAMPLERS.get(key)
    if cached is not None:
        return display, cached

    def sample(bank, scenario_id, position, quat, time, window):
        frame = cast_lidar(display, bank, scenario_id, position, quat, time, window)
        return frame.points_world, frame.valid

    batched = jax.jit(jax.vmap(sample, in_axes=(None, None, 0, 0, 0, 0)))
    _MID360_SAMPLERS[key] = batched
    return display, batched


def _depth_sampler(camera: DepthCamera, stride: int):
    key = (
        int(camera.width),
        int(camera.height),
        float(camera.horizontal_fov_deg),
        float(camera.near_m),
        float(camera.far_m),
        tuple(float(value) for value in camera.mount_m),
        bool(camera.include_ground),
        int(stride),
    )
    cached = _DEPTH_SAMPLERS.get(key)
    if cached is not None:
        return cached

    def sample(bank, scenario_id, position, quat, time):
        frame = cast_depth(camera, bank, scenario_id, position, quat, time, stride)
        origin, rotation = sensor_pose(camera, position, quat)
        directions = camera.optical_directions(stride) @ rotation.T
        points = origin[None, :] + directions * frame.depth[:, None]
        return points, frame.valid

    batched = jax.jit(jax.vmap(sample, in_axes=(None, None, 0, 0, 0)))
    _DEPTH_SAMPLERS[key] = batched
    return batched


_MID360_SAMPLERS: dict[tuple, object] = {}
_DEPTH_SAMPLERS: dict[tuple, object] = {}
_DEPTH_FLIGHT_SAMPLERS: dict[tuple, object] = {}


def _depth_flight_sampler(camera: PinholeDepthCamera, stride: int):
    key = (
        int(camera.width),
        int(camera.height),
        float(camera.horizontal_fov_deg),
        float(camera.vertical_fov_deg),
        float(camera.pitch_degrees),
        float(camera.near_m),
        float(camera.far_m),
        int(stride),
    )
    cached = _DEPTH_FLIGHT_SAMPLERS.get(key)
    if cached is not None:
        return cached

    fx, fy = camera.focal_pixels
    angle = math.radians(camera.pitch_degrees)
    camera_rotation = jnp.asarray(
        [
            [math.cos(angle), 0.0, -math.sin(angle)],
            [0.0, 1.0, 0.0],
            [math.sin(angle), 0.0, math.cos(angle)],
        ],
        jnp.float32,
    )
    columns = (jnp.arange(0, camera.width, stride, dtype=jnp.float32) + 0.5 - camera.width / 2) / fx
    rows = (jnp.arange(0, camera.height, stride, dtype=jnp.float32) + 0.5 - camera.height / 2) / fy
    right, down = jnp.meshgrid(columns, rows)
    rays = jnp.stack((jnp.ones_like(right), -right, -down), -1).reshape(-1, 3)

    def sample(bank, scenario_id, position, quat, time):
        rotation = quat_to_matrix_xyzw(quat)
        directions = rays @ camera_rotation.T @ rotation.T
        distance = cast_rays(
            bank.kind[scenario_id],
            bank.size[scenario_id],
            obstacle_positions(bank, scenario_id, time),
            bank.active[scenario_id],
            jnp.broadcast_to(position, directions.shape),
            directions,
            bank.world_low,
            bank.world_high,
            True,
            rotations=None if bank.rotations is None else bank.rotations[scenario_id],
        )
        valid = jnp.isfinite(distance) & (distance >= camera.near_m) & (distance <= camera.far_m)
        points = position[None, :] + directions * jnp.where(valid, distance, 0.0)[:, None]
        return points, valid

    batched = jax.jit(jax.vmap(sample, in_axes=(None, None, 0, 0, 0)))
    _DEPTH_FLIGHT_SAMPLERS[key] = batched
    return batched


def _run_chunks(function, fixed, arrays, *, chunk_size=32):
    length = len(arrays[0])
    outputs = []
    masks = []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        actual = stop - start
        packed = []
        for array in arrays:
            part = np.asarray(array[start:stop])
            if actual < chunk_size:
                pad = np.repeat(part[-1:], chunk_size - actual, axis=0)
                part = np.concatenate([part, pad], axis=0)
            packed.append(jnp.asarray(part))
        points, valid = function(*fixed, *packed)
        outputs.append(np.asarray(points)[:actual])
        masks.append(np.asarray(valid, bool)[:actual])
    return np.concatenate(outputs, axis=0), np.concatenate(masks, axis=0)


def sensor_hit_sequence(context: ReplaySensorContext, trace: dict) -> tuple[np.ndarray, dict]:
    """Reconstruct display-density world-frame hits for one single-episode replay.

    The overlay is deliberately a replay rendering product. It is ray-cast from
    the recorded pose and the same frozen scene geometry; it is not claimed to be
    an archived copy of the policy's observation tensor.
    """
    positions = np.asarray(trace["pos"], np.float32)
    quaternions = np.asarray(trace["quat"], np.float32)
    times = np.asarray(trace["time"], np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (1, 3):
        raise ValueError("Replay sensor hits require one episode with pos [T,1,3]")
    if quaternions.shape != (len(positions), 1, 4) or times.shape != (
        len(positions),
        1,
    ):
        raise ValueError("Replay sensor pose/time shapes differ")
    position = positions[:, 0]
    quat = quaternions[:, 0]
    time = times[:, 0]
    scenario = jnp.int32(context.scenario_id)
    sensor = context.sensor
    effective_budget = min(
        int(context.max_points),
        max(1, REPLAY_HIT_SAMPLE_BUDGET // max(1, len(time))),
    )

    if isinstance(sensor, Mid360Lidar):
        display, sampler = _mid360_sampler(sensor, effective_budget)
        windows = np.floor(np.maximum(time, 0.0) * sensor.source_rate_hz + 1e-6).astype(np.int32)
        points, valid = _run_chunks(
            sampler,
            (context.bank, scenario),
            (position, quat, time, windows),
        )
        sampling = {
            "kind": "mid360",
            "points_per_frame": display.points_per_frame,
            "downsample": display.downsample,
            "source_pattern": display.calibration()["pattern_source"],
        }
    elif isinstance(sensor, DepthCamera):
        stride = _grid_stride(sensor.width, sensor.height, effective_budget)
        sampler = _depth_sampler(sensor, stride)
        points, valid = _run_chunks(
            sampler,
            (context.bank, scenario),
            (position, quat, time),
        )
        sampling = {
            "kind": "d435",
            "points_per_frame": (sensor.width // stride) * (sensor.height // stride),
            "pixel_stride": stride,
            "resolution": [sensor.width, sensor.height],
        }
    elif isinstance(sensor, PinholeDepthCamera):
        stride = _grid_stride(sensor.width, sensor.height, effective_budget)
        sampler = _depth_flight_sampler(sensor, stride)
        points, valid = _run_chunks(
            sampler,
            (context.bank, scenario),
            (position, quat, time),
        )
        sampling = {
            "kind": "d435i_depth_flight",
            "points_per_frame": (sensor.width // stride) * (sensor.height // stride),
            "pixel_stride": stride,
            "resolution": [sensor.width, sensor.height],
        }
    else:
        raise TypeError(f"Unsupported replay hit sensor: {type(sensor).__name__}")

    cloud = np.where(valid[..., None], points, np.nan).astype(np.float32)
    metadata = {
        **sampling,
        "source_sensor": getattr(sensor, "name", type(sensor).__name__),
        "frame": "world",
        "requested_max_points_per_frame": int(context.max_points),
        "total_marker_sample_budget": REPLAY_HIT_SAMPLE_BUDGET,
        "reconstruction": (
            "replay-only ray cast from recorded vehicle pose and frozen scene geometry; "
            "not an archived policy-observation tensor"
        ),
    }
    return cloud, metadata
