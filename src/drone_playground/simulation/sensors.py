"""Pure virtual sensors and separate training preprocessing.

Body axes are Crazyflow's x-forward, y-left, z-up; attitudes are body-to-world
xyzw quaternions. Depth is optical-axis distance, LiDAR values are radial ranges.
Mid-360 uses a synthetic nonrepeating angular sequence, not a reconstruction of
the vendor's physical scan. Its public envelope/timing are from
https://www.livoxtech.com/mid-360/specs. All profiles explicitly use ideal errors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.scene import Scene


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
    def d435i(cls, mode: str = "1280x720", **overrides) -> SensorConfig:
        """30 Hz axial depth; training64x48 defaults to 20 degree upward pitch."""
        if mode == "1280x720":
            config = cls()
        elif mode == "training64x48":
            config = cls(width=64, height=48, pitch_deg=20.0, max_range=10.0)
        else:
            raise ValueError(f"Unsupported D435i mode {mode!r}")
        return replace(config, **overrides)

    @classmethod
    def mid360(cls, **overrides) -> SensorConfig:
        """20,000 first returns / 0.1 s; ideal 40 m reflectivity-limited envelope."""
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


class Measurement(NamedTuple):
    """JAX pytree of values, validity and acquisition metadata.

    Invalid values and body points are zero; directions/timestamps remain valid.
    Pixel fields have shape ``(..., height, width)``, point fields ``(..., N)``;
    vectors append a 3-axis. Acquisition time is frame completion, available time
    includes configured latency. Each LiDAR point is in its acquisition body
    frame, not deskewed to the final frame.
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
    """Unit rays in sensor axes and time offsets relative to frame completion.

    Depth/uniform grids are instantaneous. Mid-360 rays span the preceding
    0.1 s, spaced by 5 us in the default profile. ``frame_index`` can carry an
    explicit scan phase across resets; otherwise phase follows absolute time.
    """
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
        index = jnp.arange(config.points_per_frame, dtype=jnp.float32)
        # Irrational rotations sample the published envelope without claiming optical fidelity.
        azimuth = (
            2
            * jnp.pi
            * jnp.mod(index * 0.61803398875 + jnp.mod(frame[..., None] * 0.41421356237, 1), 1)
        )
        fraction = jnp.mod(index * 0.75487766625 + jnp.mod(frame[..., None] * 0.73205080757, 1), 1)
        elevation = jnp.deg2rad(
            config.elevation_min_deg
            + (config.elevation_max_deg - config.elevation_min_deg) * fraction
        )
        offsets = (index - (config.points_per_frame - 1)) / (
            config.points_per_frame * config.frequency_hz
        )
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
    pose_at: Callable[[Array], tuple[Array, Array]] | None = None,
) -> Measurement:
    """Measure batched worlds at frame-completion time ``t`` (one time per world).

    Position/quaternion are ``(..., 3/4)``. By default the supplied pose is held
    throughout a scan while geometry moves at each ray's exact time. To account
    for ego motion, supply a pure ``pose_at(times)->(positions, xyzw)`` function
    accepting the full ``(..., rays)`` timestamp array. Static configs and scenes
    should be closed over when applying JIT. There is no mutable sensor state.
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
    if pose_at is None:
        positions = jnp.broadcast_to(position[..., None, :], (*batch, count, 3))
        attitudes = jnp.broadcast_to(quaternion[..., None, :], (*batch, count, 4))
    else:
        positions, attitudes = pose_at(times)
    rotation = Rotation.from_quat(attitudes.reshape(-1, 4)).as_matrix()
    rotation = rotation.reshape((*batch, count, 3, 3))
    origins = positions + jnp.einsum("...nij,j->...ni", rotation, jnp.asarray(config.translation))
    world_rays = jnp.einsum("...nij,...nj->...ni", rotation, body_rays)
    # A depth cutoff is axial: off-axis rays must travel farther than max_range.
    limit = config.max_range / rays[..., 0] if config.profile == "d435i" else config.max_range
    ranges = scene.raycast(origins, world_rays[..., None, :], times, limit)[..., 0]
    values = ranges * rays[..., 0] if config.profile == "d435i" else ranges
    mask = jnp.isfinite(values) & (values >= config.min_range) & (ranges < limit)
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
) -> tuple[Array, Array]:
    """Inverse axial depth then 4x4 max pooling; 48x64 -> 12x16.

    Implements the documented training recipe ``3/clip(depth,.3,10)-.6``.
    Returns pooled values and masks; invalid pixels never win a pooled maximum,
    and wholly invalid cells are zero. This is independent of device simulation.
    """
    depth, mask = measurement.values, measurement.mask
    height, width = depth.shape[-2:]
    if height % 4 or width % 4 or not 0 < near < far:
        raise ValueError("Depth dimensions must be divisible by 4 and 0 < near < far")
    inverse = scale / jnp.clip(jnp.where(mask, depth, far), near, far) + offset
    blocks = (*depth.shape[:-2], height // 4, 4, width // 4, 4)
    pooled = jnp.max(jnp.where(mask, inverse, -jnp.inf).reshape(blocks), axis=(-3, -1))
    valid = jnp.any(mask.reshape(blocks), axis=(-3, -1))
    return jnp.where(valid, pooled, 0), valid


def reduce_points(measurement: Measurement, count: int) -> Measurement:
    """Deterministically subsample a point scan, preserving masks and timestamps.

    Evenly spaced acquisition indices preserve the scan's temporal span. Invalid
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
