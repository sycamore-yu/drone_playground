"""D435 ideal metric depth: camera model, timing, ray grid and framing.

The camera follows the pinned SANDO D435 configuration: 120x90, horizontal field
of view 85.2 degrees, 0.1-10 m range and a 30 Hz availability, mounted on the
body with the SANDO offset and the standard ROS optical axis convention. The
depth value is ideal metric depth with no RealSense noise model; the body load
stays the project's virtual-sensor convention.

Timing is part of the sensor contract, not an afterthought: ``capture_time`` is
the simulation time of the sample, ``available_time`` adds the declared latency
(zero in the first version) and the training loop refreshes the frame on a fixed
control-step divisor so the realised rate is exact and recordable.

Only ideal geometry is produced here, so D435 and MID360 share the same scene
query and differ only in their ray grids and framing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import struct

from drone_playground.environments.scenes.navigation import obstacle_positions
from drone_playground.environments.sensors.rays import cast_rays

# SANDO urdf/_d435.urdf.xacro: base_link -> d435_bottom_screw_frame (0.05, 0, 0),
# then camera_link (0, 0.0175, 0.0125); depth_optical_frame rpy (-pi/2, 0, -pi/2).
MOUNT_TRANSLATION_M = (0.05, 0.0175, 0.0125)
"""Optical frame origin expressed in the body FLU frame."""

BODY_FROM_OPTICAL = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)
"""``R_body_from_optical``: maps an optical-frame vector into the body FLU frame.

Columns are the optical axes expressed in the body frame: optical x (right) is
body -y, optical y (down) is body -z, optical z (forward) is body +x. This is
the transpose of the URDF ``depth_optical_frame`` rotation, which expresses the
child frame in the parent.
"""

D435_HORIZONTAL_FOV_DEG = 85.2
D435_RESOLUTION = (120, 90)
D435_RANGE_M = (0.1, 10.0)
D435_SOURCE_RATE_HZ = 30.0


@struct.dataclass
class DepthFrame:
    """One depth measurement: ideal metric depth plus its validity mask."""

    depth: jax.Array
    """``(N,)`` axial depth in metres; zero where the pixel has no return."""

    valid: jax.Array
    """``(N,)`` true when the pixel returned within ``[near, far]``."""

    time: jax.Array
    """Scalar simulation time at which the frame was captured."""


@dataclass(frozen=True)
class DepthCamera:
    """D435-like ideal depth camera with a fixed extrinsic and ray grid."""

    name: str = "d435_depth"
    width: int = D435_RESOLUTION[0]
    height: int = D435_RESOLUTION[1]
    horizontal_fov_deg: float = D435_HORIZONTAL_FOV_DEG
    near_m: float = D435_RANGE_M[0]
    far_m: float = D435_RANGE_M[1]
    source_rate_hz: float = D435_SOURCE_RATE_HZ
    mount_m: tuple[float, float, float] = MOUNT_TRANSLATION_M
    stride: int = 6
    """Pixel stride of the policy grid; the full frame stays available."""
    history: int = 4
    """Policy history length in frames."""
    calibration_id: str = "p5-d435-ideal-120x90-hfov85.2-v1"
    include_ground: bool = True

    def __post_init__(self) -> None:
        if self.width % self.stride or self.height % self.stride:
            raise ValueError("depth stride must divide the sensor resolution")
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError("horizontal field of view must be inside (0, 180) degrees")
        if not 0.0 < self.near_m < self.far_m:
            raise ValueError("depth range must satisfy 0 < near < far")
        if self.history < 1:
            raise ValueError("depth history must contain at least one frame")

    # -- calibration ------------------------------------------------------------------

    @property
    def focal_px(self) -> float:
        """Focal length in pixels; the pixel grid is square, so fy == fx."""
        return 0.5 * self.width / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)

    @property
    def vertical_fov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(0.5 * self.height / self.focal_px))

    @property
    def principal_point_px(self) -> tuple[float, float]:
        return (self.width / 2.0, self.height / 2.0)

    def intrinsics(self) -> dict:
        fx = self.focal_px
        return {
            "model": "pinhole",
            "width": self.width,
            "height": self.height,
            "fx_px": fx,
            "fy_px": fx,
            "cx_px": self.principal_point_px[0],
            "cy_px": self.principal_point_px[1],
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
            "near_m": self.near_m,
            "far_m": self.far_m,
            "depth_units": "metres, axial (optical z)",
            "invalid_value": 0.0,
        }

    def calibration(self) -> dict:
        """Static record attached to every run that used this sensor."""
        return {
            "calibration_id": self.calibration_id,
            "sensor": "D435 ideal metric depth",
            "noise_model": "none (ideal geometry; RealSense noise is out of scope)",
            "extrinsics": {
                "parent_frame": "body FLU",
                "child_frame": "camera optical (x right, y down, z forward)",
                "translation_m": list(self.mount_m),
                "rotation_body_from_optical": [list(row) for row in BODY_FROM_OPTICAL],
            },
            "intrinsics": self.intrinsics(),
            "source_availability_hz": self.source_rate_hz,
            "policy_stride": self.stride,
            "policy_points_per_frame": self.rays_per_frame,
            "policy_channels": self.channels,
            "policy_grid": [self.width // self.stride, self.height // self.stride],
            "policy_history_frames": self.history,
            "declared_latency_s": 0.0,
            "floor_extends_beyond_corridor_m": 100.0,
            "source": "SANDO urdf/_d435.gazebo.xacro + gazebo_ros_realsense intrinsics",
        }

    # -- ray grid ---------------------------------------------------------------------

    def pixel_grid(self, stride: int = 1) -> tuple[jax.Array, jax.Array]:
        """Pixel centres ``(u, v)`` in image coordinates for the requested stride."""
        if stride < 1 or self.width % stride or self.height % stride:
            raise ValueError("stride must divide the sensor resolution")
        columns = jnp.arange(0, self.width, stride, dtype=jnp.float32) + 0.5
        rows = jnp.arange(0, self.height, stride, dtype=jnp.float32) + 0.5
        grid_u, grid_v = jnp.meshgrid(columns, rows, indexing="ij")
        return grid_u, grid_v

    def optical_directions(self, stride: int = 1) -> jax.Array:
        """``(N, 3)`` optical-frame ray directions with ``z = 1``.

        The z component is exactly one, so the ray parameter equals the axial
        depth and no post-hoc unit conversion is needed.
        """
        grid_u, grid_v = self.pixel_grid(stride)
        fx = self.focal_px
        cx, cy = self.principal_point_px
        return jnp.stack(
            [(grid_u - cx) / fx, (grid_v - cy) / fx, jnp.ones_like(grid_u)], axis=-1
        ).reshape(-1, 3)

    @property
    def rays_per_frame(self) -> int:
        return (self.width // self.stride) * (self.height // self.stride)

    @property
    def points_per_frame(self) -> int:
        """Uniform name shared with the other range sensor."""
        return self.rays_per_frame

    channels = 2
    """Per-frame channels: ``(axial depth in metres, validity)``."""

    def frame_values(self, frame: DepthFrame) -> jax.Array:
        """``(P, 2)`` raw per-point channels consumed by the observation."""
        return jnp.stack([frame.depth, frame.valid.astype(jnp.float32)], axis=-1)

    # -- scheduling -------------------------------------------------------------------

    def period_steps(self, policy_freq: int) -> int:
        """Control steps between frames, using the nearest exact divisor."""
        if policy_freq % self.source_rate_hz == 0:
            return int(policy_freq // self.source_rate_hz)
        return max(1, int(round(policy_freq / self.source_rate_hz)))

    @property
    def body_from_optical(self) -> jax.Array:
        return jnp.asarray(BODY_FROM_OPTICAL, jnp.float32)


def sensor_pose(camera: DepthCamera, position, quat):
    """World pose of the optical frame from the body pose (xyzw quaternion).

    Args:
        camera: the depth camera model.
        position: ``(3,)`` body position in world coordinates.
        quat: ``(4,)`` body orientation in world coordinates, xyzw order.

    Returns:
        ``(origin (3,), rotation (3, 3))`` for the optical frame, where the
        rotation maps optical-frame vectors into world coordinates.
    """
    from drone_playground.environments.scenes.navigation import rotate_body_offset

    mount = jnp.asarray(camera.mount_m, jnp.float32)
    origin = position + rotate_body_offset(quat, mount)

    # R_world_from_optical = R_world_from_body @ R_body_from_optical
    x, y, z, w = quat
    body = jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            jnp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            jnp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ]
    )
    return origin, body @ camera.body_from_optical


def cast_depth(
    camera: DepthCamera,
    bank,
    scenario_id,
    position,
    quat,
    time,
    stride: int | None = None,
) -> DepthFrame:
    """Render one ideal depth frame at the body pose and simulation time."""
    stride = camera.stride if stride is None else stride
    origin, rotation = sensor_pose(camera, position, quat)
    directions = camera.optical_directions(stride) @ rotation.T
    centres = obstacle_positions(bank, scenario_id, time)
    distance = cast_rays(
        bank.kind[scenario_id],
        bank.size[scenario_id],
        centres,
        bank.active[scenario_id],
        origin[None, :] + jnp.zeros_like(directions),
        directions,
        bank.world_low,
        bank.world_high,
        camera.include_ground,
        rotations=None if bank.rotations is None else bank.rotations[scenario_id],
    )
    valid = (distance >= camera.near_m) & (distance <= camera.far_m)
    depth = jnp.where(valid, distance, 0.0)
    return DepthFrame(depth=depth, valid=valid, time=jnp.asarray(time, jnp.float32))


@dataclass
class DepthObservationConfig:
    """Preprocessing frozen before the network encoder (spec section 8.1)."""

    name: str = "depth"
    normalize: bool = True
    unknown_depth_value: float = 0.0
    minimum_depth_m: float = 0.1
    metadata: dict = field(default_factory=dict)

    def preprocess(self, frame: DepthFrame, camera: DepthCamera) -> jax.Array:
        """Range clip, invalid mask and normalisation into the policy input.

        Depth is converted to a bounded, monotone inverse-depth signal so that
        far and invalid pixels are both representable without a discontinuity,
        and the validity mask is appended so the encoder can distinguish a
        missing return from a distant surface.
        """
        clipped = jnp.clip(
            jnp.where(frame.valid, frame.depth, camera.far_m),
            self.minimum_depth_m,
            camera.far_m,
        )
        inverse = (1.0 / clipped - 1.0 / camera.far_m) / (
            1.0 / self.minimum_depth_m - 1.0 / camera.far_m
        )
        signal = jnp.where(self.normalize, inverse, clipped)
        return jnp.concatenate([signal, frame.valid.astype(jnp.float32)])
