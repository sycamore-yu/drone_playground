"""Explicit, differentiable decoders from neural outputs to physical decisions.

Only the host message conversion uses NumPy. The numerical decoder supports
JIT, batching and derivatives without a dependency on any execution controller.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PhysicalOutput:
    kind: str
    anchor: str = "position"
    count: int = 1
    position_scale_m: tuple = (1., 1., 1.)
    velocity_scale_mps: tuple = (1., 1., 1.)
    acceleration_scale_mps2: tuple = (1., 1., 1.)
    horizon_seconds: float = 1.
    tolerance_m: float = .2

    def __post_init__(self):
        if self.kind not in ("waypoint", "trajectory") or self.anchor not in ("position", "goal", "world"):
            raise ValueError("Choose a physical waypoint/trajectory head and explicit world anchor")
        if not isinstance(self.count, int) or self.count < 1 or (self.kind == "trajectory" and self.count != 1):
            raise ValueError("Waypoint count must be positive; trajectory head has one quintic segment")
        for field in ("position_scale_m", "velocity_scale_mps", "acceleration_scale_mps2"):
            value = np.asarray(getattr(self, field), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError("Physical output scales must be three positive finite values")
            object.__setattr__(self, field, tuple(value))
        if not np.isfinite([self.horizon_seconds, self.tolerance_m]).all() or min(self.horizon_seconds, self.tolerance_m) <= 0:
            raise ValueError("Physical output horizon and tolerance must be positive and finite")

    @property
    def action_size(self):
        return 3*self.count if self.kind == "waypoint" else 9

    def decode(self, action, position, velocity, goal):
        """Return world waypoints [...,N,3] or ascending coefficients [...,4,6].

        A trajectory predicts its final position, velocity and acceleration.
        Initial position/velocity are observed; initial reference acceleration
        is explicitly zero. Duration is a fixed part of the artifact contract.
        Endpoint interpolation alone does not guarantee obstacle or limit safety.
        """
        action, position, velocity, goal = map(jnp.asarray, (action, position, velocity, goal))
        if action.shape[-1] != self.action_size or any(x.shape[-1] != 3 for x in (position, velocity, goal)):
            raise ValueError("Neural output dimensions disagree with their physical decoder")
        origin = {"position": position, "goal": goal, "world": jnp.zeros_like(position)}[self.anchor]
        if self.kind == "waypoint":
            return origin[..., None, :] + action.reshape((*action.shape[:-1], self.count, 3))*jnp.asarray(self.position_scale_m)
        end = origin + action[..., :3]*jnp.asarray(self.position_scale_m)
        end_velocity = action[..., 3:6]*jnp.asarray(self.velocity_scale_mps)
        end_acceleration = action[..., 6:9]*jnp.asarray(self.acceleration_scale_mps2)
        t = self.horizon_seconds
        displacement = end-position-velocity*t
        delta_velocity = end_velocity-velocity
        xyz = jnp.stack((position, velocity, jnp.zeros_like(position),
                         10*displacement/t**3-4*delta_velocity/t**2+end_acceleration/(2*t),
                         -15*displacement/t**4+7*delta_velocity/t**3-end_acceleration/t**2,
                         6*displacement/t**5-3*delta_velocity/t**4+end_acceleration/(2*t**3)), axis=-1)
        return jnp.concatenate((xyz, jnp.zeros_like(xyz[..., :1, :])), axis=-2)

    def message(self, decoded, time):
        from drone_playground.native.contracts import Trajectory, Waypoint

        value = np.asarray(decoded)
        if self.kind == "waypoint":
            return Waypoint(value, self.tolerance_m)
        return Trajectory(time, [self.horizon_seconds], value[None], yaw_defined=False)
