"""MID360 virtual LiDAR: non-repetitive scan pattern, ray casting and framing.

The scan pattern is reused from the pinned MuJoCo-LiDAR release (MIT) rather
than invented: ``LivoxGenerator('mid360')`` yields a non-repetitive horizon of
800,000 precomputed rays sampled as 24,000-ray windows, spanning the full 360
degree azimuth and the -7.2 to +52.2 degree elevation of a real MID360. The
windows are materialised once as a static host array so the scan *phase* becomes
episode state that resets with the episode, instead of a mutable cursor hidden
inside a third-party generator.

Geometry is queried through the same analytic primitive description the depth
camera and the collision test use, so all three agree by construction. The
range-limit convention is the source one (0.1 to 200 m); the observation uses a
separate normalisation range because the corridor's longest possible return is
about 25 m.

Scan distortion is deliberately frozen to the *instantaneous* model: every point
of a window is evaluated at the frame's capture time, while the per-point
relative time implied by the angular scan progress is recorded for later work.
SUPER's world-frame point cloud requirement is served by ``points_world``, which
applies the real rotation rather than relabelling a frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from drone_playground.environments.scenes.geometry import obstacle_positions
from drone_playground.environments.sensors.rays import cast_rays
from drone_playground.numerics import quat_to_matrix_xyzw

MID360_PATTERN = "mid360"
MID360_SOURCE_COMMIT_NOTE = "mujoco-lidar==0.3.5 scan_mode/mid360.npy"
MID360_SAMPLES_PER_SCAN = 24000
MID360_RANGE_M = (0.1, 200.0)
MID360_AZIMUTH_DEG = 360.0
MID360_ELEVATION_DEG = (-7.212297, 52.164)
MID360_SOURCE_RATE_HZ = 10.0


@struct.dataclass
class LidarFrame:
    """One MID360 measurement: per-point range, validity and point sets."""

    distance: jax.Array
    """``(P,)`` range in meters; zero where the point has no return."""

    valid: jax.Array
    """``(P,)`` true when the point returned inside the declared range."""

    points_sensor: jax.Array
    """``(P, 3)`` hit points in the sensor (body FLU) frame."""

    points_world: jax.Array
    """``(P, 3)`` hit points in the world frame, for world-frame consumers."""

    point_time: jax.Array
    """``(P,)`` relative time inside the scan period, from angular progress."""

    time: jax.Array
    """Scalar simulation time at which the frame was captured."""


@lru_cache(maxsize=4)
def scan_windows(pattern: str = MID360_PATTERN, downsample: int = 1) -> np.ndarray:
    """Static non-repetitive windows ``(W, P, 2)`` of ``(theta, phi)`` radians.

    The source generator only exposes a stateful cursor, so every window is
    drawn once here. Window order is the source order, which is what makes the
    scan phase reproducible and resettable.
    """
    from mujoco_lidar.scan_gen import LivoxGenerator

    generator = LivoxGenerator(pattern)
    count = generator.n_rays // generator.samples
    windows = []
    for _ in range(count):
        theta, phi = generator.sample_ray_angles(downsample=downsample)
        windows.append(np.stack([np.asarray(theta), np.asarray(phi)], axis=-1))
    return np.asarray(np.stack(windows), dtype=np.float32)


@dataclass(frozen=True)
class Mid360Lidar:
    """MID360 ideal range sensor reusing the pinned MuJoCo-LiDAR scan pattern."""

    name: str = "mid360"
    pattern: str = MID360_PATTERN
    downsample: int = 200
    source_rate_hz: float = MID360_SOURCE_RATE_HZ
    history: int = 4
    range_m: tuple[float, float] = MID360_RANGE_M
    normalise_far_m: float = 40.0
    include_ground: bool = True
    state_gradient: str = "direct"
    obstacle_batch_size: int = 1
    mount: str = "body FLU, sensor z up (no gimbal, no lever arm)"
    calibration_id: str = "p5-mid360-mujoco-lidar-0.3.5-120pt-v1"

    def __post_init__(self) -> None:
        """Validate and prepare the Mid360Lidar instance after initialization."""
        if self.downsample < 1 or MID360_SAMPLES_PER_SCAN % self.downsample:
            raise ValueError("downsample must divide the MID360 samples-per-scan")
        if self.history < 1:
            raise ValueError("lidar history must contain at least one frame")
        if not 0.0 < self.range_m[0] < self.range_m[1]:
            raise ValueError("lidar range must satisfy 0 < min < max")
        if self.normalise_far_m <= 0.0:
            raise ValueError("normalisation range must be positive")
        if self.state_gradient not in ("direct", "detached"):
            raise ValueError("LiDAR state derivative must be direct or explicitly detached")
        if not isinstance(self.obstacle_batch_size, int) or self.obstacle_batch_size < 1:
            raise ValueError("Obstacle batch size must be a positive integer")

    @property
    def points_per_frame(self) -> int:
        return MID360_SAMPLES_PER_SCAN // self.downsample

    channels = 5
    """Per-frame channels: ``(x, y, z, range, validity)`` in the sensor frame."""

    def frame_values(self, frame: LidarFrame) -> jax.Array:
        """``(P, 5)`` raw per-point channels consumed by the observation."""
        return jnp.concatenate(
            [
                frame.points_sensor,
                frame.distance[:, None],
                frame.valid.astype(jnp.float32)[:, None],
            ],
            axis=-1,
        )

    @property
    def window_count(self) -> int:
        return scan_windows(self.pattern, self.downsample).shape[0]

    @property
    def scan_period_s(self) -> float:
        return 1.0 / self.source_rate_hz

    def period_steps(self, policy_freq: int) -> int:
        """Control steps between scans, using the nearest exact divisor."""
        if policy_freq % self.source_rate_hz == 0:
            return int(policy_freq // self.source_rate_hz)
        return max(1, round(policy_freq / self.source_rate_hz))

    def angle_table(self) -> jax.Array:
        """``(W, P, 2)`` window table as a JAX constant.

        The host table is cached as NumPy data; converting inside the function
        keeps a traced value from ever being cached, which would let a tracer
        escape the transform that created it.
        """
        return jnp.asarray(scan_windows(self.pattern, self.downsample))

    def directions(self, window) -> jax.Array:
        """``(P, 3)`` unit sensor-frame directions for one scan window.

        ``window`` may be a traced array: the scan phase is episode state and
        must stay inside the JAX program rather than being read back to the host.
        """
        angles = self.angle_table()[window % self.window_count]
        theta = angles[:, 0]
        phi = angles[:, 1]
        return jnp.stack(
            [
                jnp.cos(phi) * jnp.cos(theta),
                jnp.cos(phi) * jnp.sin(theta),
                jnp.sin(phi),
            ],
            axis=-1,
        )

    def relative_point_time(self) -> jax.Array:
        """``(P,)`` time offset inside the scan period, from angular progress.

        The shipped pattern has no per-ray timestamp, so progress is assumed
        uniform across the window. The frozen P5 model evaluates geometry at the
        instantaneous capture time; this channel records the distortion that a
        per-point evaluation would need.
        """
        return jnp.arange(self.points_per_frame, dtype=jnp.float32) / (
            self.points_per_frame * self.source_rate_hz
        )

    def calibration(self) -> dict:
        return {
            "calibration_id": self.calibration_id,
            "sensor": "MID360 ideal range sensor",
            "pattern_source": MID360_SOURCE_COMMIT_NOTE,
            "license": "MIT",
            "pattern": {
                "total_rays": 800000,
                "samples_per_scan": MID360_SAMPLES_PER_SCAN,
                "usable_windows": self.window_count,
                "azimuth_deg": MID360_AZIMUTH_DEG,
                "elevation_deg": list(MID360_ELEVATION_DEG),
                "points_per_frame": self.points_per_frame,
                "downsample_stride": self.downsample,
                "sampled_indices": "window*24000 + arange(0, 24000, downsample)",
            },
            "range_m": list(self.range_m),
            "observation_normalise_far_m": self.normalise_far_m,
            "source_availability_hz": self.source_rate_hz,
            "state_gradient": self.state_gradient,
            "obstacle_batch_size": self.obstacle_batch_size,
            "history_frames": self.history,
            "policy_points_per_frame": self.points_per_frame,
            "policy_channels": self.channels,
            "mount": self.mount,
            "scan_distortion_model": (
                "instantaneous: every point is evaluated at the frame capture time; "
                "per-point relative time is recorded but not applied to geometry"
            ),
            "world_frame_conversion": "points_world = R_world_from_body @ points_sensor",
        }


def cast_lidar(
    lidar: Mid360Lidar,
    bank,
    scenario_id,
    position,
    quat,
    time,
    window: int,
) -> LidarFrame:
    """Cast one MID360 scan window at the body pose and simulation time."""
    direction_sensor = lidar.directions(window)
    rotation = quat_to_matrix_xyzw(quat)
    direction_world = direction_sensor @ rotation.T
    centres = obstacle_positions(bank, scenario_id, time)
    distance = cast_rays(
        bank.kind[scenario_id],
        bank.size[scenario_id],
        centres,
        bank.active[scenario_id],
        position[None, :] + jnp.zeros_like(direction_world),
        direction_world,
        bank.world_low,
        bank.world_high,
        lidar.include_ground,
        rotations=None if bank.rotations is None else bank.rotations[scenario_id],
        obstacle_batch_size=lidar.obstacle_batch_size,
    )
    valid = (distance >= lidar.range_m[0]) & (distance <= lidar.range_m[1])
    rng = jnp.where(valid, distance, 0.0)
    points_sensor = direction_sensor * rng[:, None]
    points_world = position[None, :] + direction_world * rng[:, None]
    frame = LidarFrame(
        distance=rng,
        valid=valid,
        points_sensor=points_sensor,
        points_world=points_world,
        point_time=lidar.relative_point_time(),
        time=jnp.asarray(time, jnp.float32),
    )
    # The optional measurement boundary changes only the training derivative.
    # The encoder's parameter gradients and the plant/reward derivatives remain
    # active; physical ranges and inference observations are bitwise identical.
    return (
        jax.tree.map(jax.lax.stop_gradient, frame) if lidar.state_gradient == "detached" else frame
    )


def point_cloud_message(frame: LidarFrame, lidar: Mid360Lidar) -> dict:
    """Minimal world-frame point cloud record for a native planner adapter.

    Fields follow the consumer contract rather than a ROS message type: the
    caller owns serialisation, this only guarantees that the world transform has
    actually been applied to every valid point.
    """
    valid = np.asarray(frame.valid)
    world = np.asarray(frame.points_world)[valid]
    return {
        "frame_id": "world",
        "sensor_frame_id": "mid360",
        "point_count": int(world.shape[0]),
        "points_world": world,
        "ranges": np.asarray(frame.distance)[valid],
        "point_time": np.asarray(frame.point_time)[valid],
        "capture_time": float(np.asarray(frame.time)),
        "calibration_id": lidar.calibration_id,
        "elevation_deg": list(MID360_ELEVATION_DEG),
        "min_range_m": lidar.range_m[0],
        "max_range_m": lidar.range_m[1],
    }


def angular_coverage(lidar: Mid360Lidar) -> dict:
    """Measured azimuth/elevation coverage of the downsampled policy window."""
    angles = scan_windows(lidar.pattern, lidar.downsample)[0]
    theta = np.degrees(np.asarray(angles[:, 0]))
    phi = np.degrees(np.asarray(angles[:, 1]))
    bins = np.unique(np.floor(theta / 10.0).astype(int))
    return {
        "points": int(theta.shape[0]),
        "azimuth_deg": [float(theta.min()), float(theta.max())],
        "elevation_deg": [float(phi.min()), float(phi.max())],
        "occupied_azimuth_bins_of_36": len(bins),
        "mean_rays_per_10deg": float(theta.shape[0] / 36.0),
        "expected_mean_rays_per_10deg": float(MID360_SAMPLES_PER_SCAN / lidar.downsample / 36.0),
        "sensor": lidar.calibration_id,
        "note": "uniform stride over the source window preserves azimuthal coverage",
        "declared_elevation_deg": list(MID360_ELEVATION_DEG),
        "declared_azimuth_deg": MID360_AZIMUTH_DEG,
        "period_s": lidar.scan_period_s,
    }


def period_check(lidar: Mid360Lidar, policy_freq: int) -> dict:
    """Check LiDAR sampling cadence against the environment control rate."""
    steps = lidar.period_steps(policy_freq)
    return {
        "policy_freq_hz": policy_freq,
        "source_rate_hz": lidar.source_rate_hz,
        "period_steps": steps,
        "realised_rate_hz": policy_freq / steps,
        "exact_divisor": policy_freq % lidar.source_rate_hz == 0,
        "source_period_s": lidar.scan_period_s,
        "math_note": math.isclose(policy_freq / steps, lidar.source_rate_hz, rel_tol=1e-9),
    }


@dataclass(frozen=True)
class UniformRayLidar:
    policy_value_channels = 4
    name: str = "uniform_lidar"
    azimuth_count: int = 180
    elevation_count: int = 30
    elevation_start_deg: float = -7.2
    elevation_span_deg: float = 60.0
    range_m: tuple[float, float] = (0.1, 100.0)
    source_rate_hz: float = 10.0
    include_ground: bool = True
    state_gradient: str = "detached"

    def __post_init__(self):
        """Validate and prepare the UniformRayLidar instance after initialization."""
        if self.azimuth_count < 1 or self.elevation_count < 1:
            raise ValueError("Ray counts must be positive")
        if not 0 < self.range_m[0] < self.range_m[1]:
            raise ValueError("Sensor range must be ordered and positive")
        if self.state_gradient != "detached":
            raise ValueError("This reconstruction qualifies detached ray measurements only")

    @property
    def points_per_frame(self):
        return self.azimuth_count * self.elevation_count

    def directions(self, window=0):
        del window
        theta = jnp.arange(self.azimuth_count) * (2 * jnp.pi / self.azimuth_count)
        phi = jnp.deg2rad(
            self.elevation_start_deg
            + jnp.arange(self.elevation_count) * self.elevation_span_deg / self.elevation_count
        )
        theta, phi = jnp.meshgrid(theta, phi, indexing="ij")
        return jnp.stack(
            [
                jnp.cos(phi) * jnp.cos(theta),
                jnp.cos(phi) * jnp.sin(theta),
                jnp.sin(phi),
            ],
            axis=-1,
        ).reshape(-1, 3)

    def sample(self, bank, scenario_id, position, rotation, time):
        """Return body XYZ points and a validity mask; state derivatives stop here."""
        position, rotation, time = jax.tree.map(jax.lax.stop_gradient, (position, rotation, time))
        directions = self.directions()
        world_directions = directions @ rotation.T
        centres = obstacle_positions(bank, scenario_id, time)
        distance = cast_rays(
            bank.kind[scenario_id],
            bank.size[scenario_id],
            centres,
            bank.active[scenario_id],
            position[None] + jnp.zeros_like(directions),
            world_directions,
            bank.world_low,
            bank.world_high,
            self.include_ground,
            rotations=None if bank.rotations is None else bank.rotations[scenario_id],
        )
        valid = (distance >= self.range_m[0]) & (distance <= self.range_m[1])
        points = directions * jnp.where(valid, distance, 0.0)[:, None]
        return jax.lax.stop_gradient(points), valid

    def calibration(self):
        return dict(
            name=self.name,
            ray_count=self.points_per_frame,
            azimuth_count=self.azimuth_count,
            elevation_count=self.elevation_count,
            elevation_start_deg=self.elevation_start_deg,
            elevation_span_deg=self.elevation_span_deg,
            range_m=list(self.range_m),
            source_rate_hz=self.source_rate_hz,
            state_gradient=self.state_gradient,
            frame="body FLU; XYZ in metres",
            acquisition="instantaneous regular-angle rays",
        )
