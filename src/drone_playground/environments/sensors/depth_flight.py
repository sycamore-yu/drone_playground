"""D435i-like ideal depth for the navigation adaptation (not stereo matching)."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.navigation import obstacle_positions

from .rays import cast_rays


@dataclass(frozen=True)
class DepthFlightCamera:
    name: str = "d435i_depth_flight_ideal"
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
        if (self.width, self.height) != (64, 48) or self.state_gradient != "detached":
            raise ValueError(
                "Depth-flight camera requires 64x48 and detached measurement gradients"
            )
        numbers = (self.horizontal_fov_deg, self.vertical_fov_deg, self.pitch_degrees,
                   self.near_m, self.far_m, self.source_rate_hz, self.capture_rate_hz)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Camera calibration must be finite")
        if not all(0 < value < 180 for value in numbers[:2]):
            raise ValueError("Camera fields of view must be inside (0, 180) degrees")
        if not 0 < self.near_m < self.far_m or not 0 < self.source_rate_hz <= self.capture_rate_hz:
            raise ValueError("Require 0 < near < far and 0 < used rate <= capture rate")
        if not math.isclose(self.capture_rate_hz / self.source_rate_hz,
                            round(self.capture_rate_hz / self.source_rate_hz)):
            raise ValueError("The used camera rate must divide the nominal capture rate")

    @property
    def focal_pixels(self):
        # Independently resized axes preserve the HD sensor's full 87x58 FOV.
        # The resulting 64x48 policy image does not have square focal pixels.
        return (self.width / (2 * math.tan(math.radians(self.horizontal_fov_deg) / 2)),
                self.height / (2 * math.tan(math.radians(self.vertical_fov_deg) / 2)))

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
            fx_px=self.focal_pixels[0], fy_px=self.focal_pixels[1],
            cx_px=self.width / 2, cy_px=self.height / 2,
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
