"""Paper's regular-angle MID-360 approximation using the common scene ray caster."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drone_playground.tasks.scenes.navigation import obstacle_positions
from drone_playground.tasks.sensors.rays import cast_rays


@dataclass(frozen=True)
class UniformMid360Lidar:
    name: str = "paper_mid360_uniform"
    azimuth_count: int = 180
    elevation_count: int = 30
    elevation_start_deg: float = -7.2
    elevation_span_deg: float = 60.0
    range_m: tuple[float, float] = (0.1, 100.0)
    source_rate_hz: float = 10.0
    include_ground: bool = True
    state_gradient: str = "detached"

    def __post_init__(self):
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
            [jnp.cos(phi) * jnp.cos(theta), jnp.cos(phi) * jnp.sin(theta), jnp.sin(phi)], axis=-1
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
