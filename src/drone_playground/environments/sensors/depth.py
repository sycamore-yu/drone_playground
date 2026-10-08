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
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from drone_playground.environments.scenes.geometry import obstacle_positions
from drone_playground.environments.sensors.rays import cast_rays
from drone_playground.numerics import quat_to_matrix_xyzw

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
    """``(N,)`` axial depth in meters; zero where the pixel has no return."""

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
        """Validate and prepare the DepthCamera instance after initialization."""
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
            "policy_grid": [
                self.width // self.stride,
                self.height // self.stride,
            ],
            "policy_history_frames": self.history,
            "declared_latency_s": 0.0,
            "floor_extends_beyond_corridor_m": 100.0,
            "source": "SANDO urdf/_d435.gazebo.xacro + gazebo_ros_realsense intrinsics",
        }

    # -- ray grid ---------------------------------------------------------------------

    def pixel_grid(self, stride: int = 1) -> tuple[jax.Array, jax.Array]:
        """Pixel centers ``(u, v)`` in image coordinates for the requested stride."""
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
            [(grid_u - cx) / fx, (grid_v - cy) / fx, jnp.ones_like(grid_u)],
            axis=-1,
        ).reshape(-1, 3)

    @property
    def rays_per_frame(self) -> int:
        return (self.width // self.stride) * (self.height // self.stride)

    @property
    def points_per_frame(self) -> int:
        """Uniform name shared with the other range sensor."""
        return self.rays_per_frame

    channels = 2
    """Per-frame channels: ``(axial depth in meters, validity)``."""

    def frame_values(self, frame: DepthFrame) -> jax.Array:
        """``(P, 2)`` raw per-point channels consumed by the observation."""
        return jnp.stack([frame.depth, frame.valid.astype(jnp.float32)], axis=-1)

    # -- scheduling -------------------------------------------------------------------

    def period_steps(self, policy_freq: int) -> int:
        """Control steps between frames, using the nearest exact divisor."""
        if policy_freq % self.source_rate_hz == 0:
            return int(policy_freq // self.source_rate_hz)
        return max(1, round(policy_freq / self.source_rate_hz))

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
    from drone_playground.environments.scenes.geometry import rotate_body_offset

    mount = jnp.asarray(camera.mount_m, jnp.float32)
    origin = position + rotate_body_offset(quat, mount)

    # R_world_from_optical = R_world_from_body @ R_body_from_optical
    body = quat_to_matrix_xyzw(quat)
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


@dataclass(frozen=True)
class PinholeDepthCamera:
    policy_value_channels = 1
    name: str = "pinhole_depth"
    source_rate_hz: float = 10.0
    capture_rate_hz: float = 30.0
    width: int = 64
    height: int = 48
    horizontal_fov_deg: float = 87.0
    vertical_fov_deg: float = 58.0
    pitch_degrees: float = 20.0
    near_m: float = 0.2
    far_m: float = 10.0
    state_gradient: str = "detached"

    def __post_init__(self):
        """Validate and prepare the PinholeDepthCamera instance after initialization."""
        if (self.width, self.height) != (
            64,
            48,
        ) or self.state_gradient != "detached":
            raise ValueError(
                "Depth-flight camera requires 64x48 and detached measurement gradients"
            )
        numbers = (
            self.horizontal_fov_deg,
            self.vertical_fov_deg,
            self.pitch_degrees,
            self.near_m,
            self.far_m,
            self.source_rate_hz,
            self.capture_rate_hz,
        )
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Camera calibration must be finite")
        if not all(0 < value < 180 for value in numbers[:2]):
            raise ValueError("Camera fields of view must be inside (0, 180) degrees")
        if not 0 < self.near_m < self.far_m or not 0 < self.source_rate_hz <= self.capture_rate_hz:
            raise ValueError("Require 0 < near < far and 0 < used rate <= capture rate")
        if not math.isclose(
            self.capture_rate_hz / self.source_rate_hz,
            round(self.capture_rate_hz / self.source_rate_hz),
        ):
            raise ValueError("The used camera rate must divide the nominal capture rate")

    @property
    def focal_pixels(self):
        # Independently resized axes preserve the HD sensor's full 87x58 FOV.
        # The resulting 64x48 policy image does not have square focal pixels.
        return (
            self.width / (2 * math.tan(math.radians(self.horizontal_fov_deg) / 2)),
            self.height / (2 * math.tan(math.radians(self.vertical_fov_deg) / 2)),
        )

    @property
    def points_per_frame(self):
        return self.width * self.height

    def sample(self, bank, index, position, rotation, time):
        position, rotation = jax.tree.map(jax.lax.stop_gradient, (position, rotation))
        fx, fy = self.focal_pixels
        u = (jnp.arange(self.width) + 0.5 - self.width / 2) / fx
        v = (jnp.arange(self.height) + 0.5 - self.height / 2) / fy
        right, down = jnp.meshgrid(u, v)
        rays = jnp.stack((jnp.ones_like(right), -right, -down), -1).reshape(-1, 3)
        angle = math.radians(self.pitch_degrees)
        camera = jnp.array(
            [
                [math.cos(angle), 0, -math.sin(angle)],
                [0, 1, 0],
                [math.sin(angle), 0, math.cos(angle)],
            ]
        )
        directions = rays @ camera.T @ rotation.T
        depth = cast_rays(
            bank.kind[index],
            bank.size[index],
            obstacle_positions(bank, index, time),
            bank.active[index],
            jnp.broadcast_to(position, directions.shape),
            directions,
            bank.world_low,
            bank.world_high,
            True,
            rotations=None if bank.rotations is None else bank.rotations[index],
        )
        valid = jnp.isfinite(depth) & (depth >= self.near_m) & (depth <= self.far_m)
        depth = jnp.where(valid, depth, self.far_m)
        return (
            jax.lax.stop_gradient(depth.reshape(self.height, self.width)),
            valid.reshape(self.height, self.width),
        )

    def calibration(self):
        return dict(
            sensor=self.name,
            width=self.width,
            height=self.height,
            horizontal_fov_deg=self.horizontal_fov_deg,
            vertical_fov_deg=self.vertical_fov_deg,
            fx_px=self.focal_pixels[0],
            fy_px=self.focal_pixels[1],
            cx_px=self.width / 2,
            cy_px=self.height / 2,
            pitch_degrees=self.pitch_degrees,
            source_rate_hz=self.source_rate_hz,
            nominal_capture_rate_hz=self.capture_rate_hz,
            frame_decimation=round(self.capture_rate_hz / self.source_rate_hz),
            rendering="direct rays on resized 64x48 grid; skipped source frames are not rendered",
            mount_translation_m=[0.0, 0.0, 0.0],
            body_load="virtual sensor; no added camera mass",
            depth_units="metres along optical forward axis",
            near_m=self.near_m,
            far_m=self.far_m,
            invalid_value=self.far_m,
            validity_mask=True,
            noise="none; navigation component adaptation",
            reference="RealSense D435i nominal depth FOV; 10m is the experiment cutoff",
            state_gradient=self.state_gradient,
        )
