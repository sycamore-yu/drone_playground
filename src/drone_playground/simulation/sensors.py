"""Snapshot depth and LiDAR measurements, separate from policy preprocessing.

Body axes are x-forward, y-left, z-up; body-to-world quaternions are xyzw.
Depth is optical-axis distance; LiDAR reports radial range. MID360 directions
come from the released MuJoCo-LiDAR table. Every ray uses the frame pose/time;
the angle resource does not provide calibrated per-ray acquisition times.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib import resources
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.scene import Scene


@lru_cache(maxsize=1)
def _mid360_pattern() -> tuple[np.ndarray, str]:
    """Load the packaged scan once, without retaining a mutable generator cursor."""
    from mujoco_lidar.scan_gen import LivoxGenerator

    angles = np.asarray(LivoxGenerator("mid360").ray_angles, dtype=np.float32)
    angles.setflags(write=False)
    source = resources.files("mujoco_lidar").joinpath("scan_mode/mid360.npy")
    return angles, hashlib.sha256(source.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SensorConfig:
    """Static sensor calibration; use the named constructors for device profiles.

    Translation is sensor origin in body metres. Positive pitch elevates the
    forward optical axis; roll/yaw follow the right-hand rule. Range limits for
    depth are axial, and for LiDAR radial. The upper limit is exclusive. D435i
    range limits are simulated mode settings, not a universal hardware range.
    """

    profile: str = "d435i"
    width: int = 1280
    height: int = 720
    horizontal_fov_deg: float = 87.0
    vertical_fov_deg: float = 58.0
    frequency_hz: float = 30.0
    min_range: float = 0.3
    max_range: float = 3.0
    points_per_frame: int = 20000
    elevation_min_deg: float = -7.0
    elevation_max_deg: float = 52.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    latency: float = 0.0
    error_model: str = "ideal"

    def __post_init__(self):
        """Validate device geometry, sampling parameters and sensor extrinsics."""
        object.__setattr__(self, "translation", tuple(self.translation))
        if self.profile not in ("d435i", "mid360", "uniform_lidar_paper"):
            raise ValueError(f"Unknown sensor profile {self.profile!r}")
        if self.error_model != "ideal":
            raise ValueError("Only the explicit ideal error model is implemented")
        if any(
            not isinstance(x, int) or x <= 0
            for x in (self.width, self.height, self.points_per_frame)
        ):
            raise ValueError("Sensor dimensions and point count must be positive integers")
        numbers = (
            self.horizontal_fov_deg,
            self.vertical_fov_deg,
            self.frequency_hz,
            self.min_range,
            self.max_range,
            self.elevation_min_deg,
            self.elevation_max_deg,
            self.roll_deg,
            self.pitch_deg,
            self.yaw_deg,
            self.latency,
            *self.translation,
        )
        if len(self.translation) != 3 or not np.isfinite(numbers).all():
            raise ValueError("Sensor calibration must be finite with a 3-vector translation")
        if not 0 <= self.min_range < self.max_range or self.frequency_hz <= 0 or self.latency < 0:
            raise ValueError("Invalid range, frequency or latency")
        if not -90 <= self.elevation_min_deg < self.elevation_max_deg <= 90:
            raise ValueError("Invalid LiDAR elevation limits")
        if self.profile == "d435i" and not (
            0 < self.horizontal_fov_deg < 180 and 0 < self.vertical_fov_deg < 180
        ):
            raise ValueError("Depth FoV must lie between 0 and 180 degrees")

    @classmethod
    def d435i(cls, **overrides) -> SensorConfig:
        """D435i calibration; experiments supply image size, frequency and mounting."""
        return replace(cls(), **overrides)

    @classmethod
    def mid360(cls, **overrides) -> SensorConfig:
        """20,000 ideal first-return queries per 10 Hz snapshot, with a 40 m cutoff."""
        return replace(
            cls(profile="mid360", frequency_hz=10, min_range=0.1, max_range=40), **overrides
        )

    @classmethod
    def uniform_lidar_paper(cls, **overrides) -> SensorConfig:
        """180 azimuths x 30 elevations at 2 degrees; -30 to +28 elevation."""
        return replace(
            cls(
                profile="uniform_lidar_paper",
                width=180,
                height=30,
                frequency_hz=10,
                min_range=0.1,
                max_range=40,
                points_per_frame=5400,
                elevation_min_deg=-30,
                elevation_max_deg=28,
            ),
            **overrides,
        )

    @property
    def specification(self) -> dict:
        """Record calibration, acquisition semantics and scan-resource identity."""
        result = {**asdict(self), "acquisition": "snapshot"}
        result["translation"] = list(self.translation)
        if self.profile == "mid360":
            angles, digest = _mid360_pattern()
            result.update(
                scan_source="mujoco-lidar==0.3.5:scan_mode/mid360.npy",
                scan_sha256=digest,
                scan_length=len(angles),
                scan_index="(frame_index * points_per_frame + ray_index) % scan_length",
            )
        return result


class Measurement(NamedTuple):
    """JAX pytree of values, validity and acquisition metadata.

    Invalid values and body points are zero; directions/timestamps remain valid.
    Pixel fields have shape ``(..., height, width)``, point fields ``(..., N)``;
    vectors append a 3-axis. Acquisition time is frame completion, available time
    includes configured latency. All points use the body frame at acquisition;
    every entry in ``times`` equals the frame's ``acquisition_time``.
    """

    values: Array
    mask: Array
    points_body: Array
    directions_body: Array
    times: Array
    acquisition_time: Array
    available_time: Array


def sensor_rays(
    config: SensorConfig, t: Array | float = 0.0, frame_index: Array | int | None = None
) -> tuple[Array, Array]:
    """Unit rays and zero snapshot offsets; each world has an explicit scan phase."""
    if config.profile == "d435i":
        x = 2 * (jnp.arange(config.width) + 0.5) / config.width - 1
        y = 1 - 2 * (jnp.arange(config.height) + 0.5) / config.height
        horizontal, vertical = jnp.meshgrid(
            -x * jnp.tan(jnp.deg2rad(config.horizontal_fov_deg) / 2),
            y * jnp.tan(jnp.deg2rad(config.vertical_fov_deg) / 2),
        )
        rays = jnp.stack((jnp.ones_like(horizontal), horizontal, vertical), axis=-1)
        rays = rays / jnp.linalg.norm(rays, axis=-1, keepdims=True)
        return rays.reshape(-1, 3), jnp.zeros(config.height * config.width)
    if config.profile == "uniform_lidar_paper":
        azimuth, elevation = jnp.meshgrid(
            jnp.arange(config.width) * (2 * jnp.pi / config.width) - jnp.pi,
            jnp.deg2rad(
                jnp.linspace(config.elevation_min_deg, config.elevation_max_deg, config.height)
            ),
        )
        azimuth, elevation = azimuth.ravel(), elevation.ravel()
        offsets = jnp.zeros(config.width * config.height)
    else:
        frame = (
            jnp.floor(jnp.asarray(t) * config.frequency_hz)
            if frame_index is None
            else jnp.asarray(frame_index)
        )
        angles, _ = _mid360_pattern()
        indices = (
            frame.astype(jnp.int32)[..., None] * config.points_per_frame
            + jnp.arange(config.points_per_frame)
        ) % len(angles)
        selected = jnp.asarray(angles)[indices]
        azimuth, elevation = selected[..., 0], selected[..., 1]
        offsets = jnp.zeros(config.points_per_frame)
    rays = jnp.stack(
        (
            jnp.cos(elevation) * jnp.cos(azimuth),
            jnp.cos(elevation) * jnp.sin(azimuth),
            jnp.sin(elevation),
        ),
        axis=-1,
    )
    return rays, offsets


def measure(
    scene: Scene,
    config: SensorConfig,
    position: Array,
    quaternion: Array,
    t: Array | float,
    *,
    frame_index: Array | int | None = None,
) -> Measurement:
    """Measure one scene/robot snapshot per world at ``t`` with fixed calibration.

    Position/quaternion have shape ``(..., 3/4)``. Dynamic obstacle geometry is
    evaluated at that world's acquisition time. Close over scene/config for JIT.
    """
    position, quaternion = jnp.asarray(position), jnp.asarray(quaternion)
    batch = jnp.broadcast_shapes(position.shape[:-1], quaternion.shape[:-1])
    time = jnp.broadcast_to(jnp.asarray(t), batch)
    rays, offsets = sensor_rays(config, time, frame_index)
    count = rays.shape[-2]
    rays = jnp.broadcast_to(rays, (*batch, count, 3))
    times = time[..., None] + offsets
    mounting = Rotation.from_euler(
        "xyz", jnp.deg2rad(jnp.array([config.roll_deg, -config.pitch_deg, config.yaw_deg]))
    ).as_matrix()
    body_rays = jnp.einsum("ij,...nj->...ni", mounting, rays)
    attitudes = jnp.broadcast_to(quaternion, (*batch, 4))
    rotation = Rotation.from_quat(attitudes.reshape(-1, 4)).as_matrix().reshape((*batch, 3, 3))
    origins = position + jnp.einsum("...ij,j->...i", rotation, jnp.asarray(config.translation))
    world_rays = jnp.einsum("...ij,...nj->...ni", rotation, body_rays)
    # A depth cutoff is axial: off-axis rays must travel farther than max_range.
    limit = (
        config.max_range / jnp.min(rays[..., 0], axis=-1)
        if config.profile == "d435i"
        else config.max_range
    )
    ranges = scene.raycast(origins, world_rays, time, limit)
    values = ranges * rays[..., 0] if config.profile == "d435i" else ranges
    hit = ranges < jnp.broadcast_to(jnp.asarray(limit), batch)[..., None]
    mask = hit & jnp.isfinite(values) & (values >= config.min_range) & (values < config.max_range)
    points = jnp.asarray(config.translation) + body_rays * ranges[..., None]
    values = jnp.where(mask, values, 0)
    points = jnp.where(mask[..., None], points, 0)
    shape = batch + ((config.height, config.width) if config.profile == "d435i" else (count,))
    return Measurement(
        values.reshape(shape),
        mask.reshape(shape),
        points.reshape((*shape, 3)),
        body_rays.reshape((*shape, 3)),
        times.reshape(shape),
        time,
        time + config.latency,
    )


def preprocess_depth(
    measurement: Measurement,
    *,
    near: float = 0.3,
    far: float = 10.0,
    scale: float = 3.0,
    offset: float = -0.6,
    pool: int = 4,
) -> tuple[Array, Array]:
    """Inverse axial depth then 4x4 max pooling; 48x64 -> 12x16.

    Implements the documented training recipe ``3/clip(depth,.3,10)-.6``.
    Returns pooled values and masks; invalid pixels never win a pooled maximum,
    and wholly invalid cells are zero. This is independent of device simulation.
    """
    depth, mask = measurement.values, measurement.mask
    height, width = depth.shape[-2:]
    if pool < 1 or height % pool or width % pool or not 0 < near < far:
        raise ValueError("Depth dimensions must be divisible by pool and 0 < near < far")
    inverse = scale / jnp.clip(jnp.where(mask, depth, far), near, far) + offset
    blocks = (*depth.shape[:-2], height // pool, pool, width // pool, pool)
    pooled = jnp.max(jnp.where(mask, inverse, -jnp.inf).reshape(blocks), axis=(-3, -1))
    valid = jnp.any(mask.reshape(blocks), axis=(-3, -1))
    return jnp.where(valid, pooled, 0), valid


def reduce_points(measurement: Measurement, count: int) -> Measurement:
    """Deterministically subsample a point scan, preserving masks and timestamps.

    Evenly spaced acquisition indices preserve the scan pattern's coverage. Invalid
    returns remain invalid rather than being relabelled or padded as real points.
    """
    size = measurement.values.shape[-1]
    if not isinstance(count, int) or not 0 < count <= size:
        raise ValueError("Point count must be a positive integer no larger than the scan")
    indices = jnp.linspace(0, size - 1, count).astype(jnp.int32)
    return Measurement(
        jnp.take(measurement.values, indices, axis=-1),
        jnp.take(measurement.mask, indices, axis=-1),
        jnp.take(measurement.points_body, indices, axis=-2),
        jnp.take(measurement.directions_body, indices, axis=-2),
        jnp.take(measurement.times, indices, axis=-1),
        measurement.acquisition_time,
        measurement.available_time,
    )
