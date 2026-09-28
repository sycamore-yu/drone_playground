"""Paper-style acceleration-driven point mass with an explicit derivative contract.

The paper specifies a point mass and first-order lag but omits its integrator and
time constant. The named reconstruction uses the predecessor's trapezoidal
velocity integration and 1/12 s lag. The exponential rule implements the paper's
Eq. (9) for the (position, velocity, acceleration) state, independently of the
predecessor's selectively decayed CUDA derivative.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct


@jax.custom_jvp
def scale_state_gradient(value, scale):
    """Identity forward map with a deliberately scaled state derivative."""
    return value


@scale_state_gradient.defjvp
def _scaled_state_jvp(primals, tangents):
    value, scale = primals
    tangent, _ = tangents
    return value, tangent * scale


def _unit(vector, fallback):
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return jnp.where(norm > 1e-6, vector / jnp.maximum(norm, 1e-6), fallback)


def acceleration_attitude(acceleration, velocity, previous_rotation, gravity=9.80665):
    """Orthonormal body-to-world frame from thrust direction and flight heading."""
    acceleration, velocity, previous_rotation = jax.tree.map(
        jax.lax.stop_gradient, (acceleration, velocity, previous_rotation)
    )
    up = _unit(acceleration + jnp.array([0.0, 0.0, gravity]), jnp.array([0.0, 0.0, 1.0]))
    heading = _unit(velocity.at[..., 2].set(0.0), previous_rotation[..., :, 0])
    # A near-collinear heading uses a second reference axis to keep the frame finite.
    side = jnp.cross(up, heading)
    alternate = jnp.cross(up, jnp.array([0.0, 1.0, 0.0]))
    side = _unit(side, _unit(alternate, jnp.array([0.0, 0.0, 1.0])))
    forward = _unit(jnp.cross(side, up), jnp.array([1.0, 0.0, 0.0]))
    return jax.lax.stop_gradient(jnp.stack([forward, side, up], axis=-1))


@struct.dataclass
class PointMassState:
    pos: jax.Array
    vel: jax.Array
    acc: jax.Array
    rotation: jax.Array

    @classmethod
    def create(cls, position):
        position = jnp.asarray(position, jnp.float32)
        return cls(
            position,
            jnp.zeros_like(position),
            jnp.zeros_like(position),
            jnp.broadcast_to(jnp.eye(3), (*position.shape[:-1], 3, 3)),
        )

    @classmethod
    def from_vector(cls, vector):
        return cls.create(vector[..., :3]).replace(vel=vector[..., 3:6], acc=vector[..., 6:9])

    def vector(self):
        return jnp.concatenate([self.pos, self.vel, self.acc], axis=-1)


@dataclass(frozen=True)
class PointMassLag:
    time_constant: float = 1 / 12
    backward: str = "exponential"
    decay_rate: float = -math.log(0.4)
    gravity: float = 9.80665
    forward: str = "point_mass_lag"
    drone: str = "paper_point_mass"

    def __post_init__(self):
        if self.time_constant <= 0 or self.decay_rate < 0:
            raise ValueError("Lag must be positive and gradient decay nonnegative")
        if self.backward not in ("direct", "exponential"):
            raise ValueError(f"Unknown point-mass derivative: {self.backward}")

    def step(self, state: PointMassState, command, dt):
        """Advance net acceleration in world coordinates; gravity is already compensated."""
        vector = state.vector()
        if self.backward == "exponential":
            vector = scale_state_gradient(vector, jnp.exp(-self.decay_rate * dt))
        p, v, a = vector[..., :3], vector[..., 3:6], vector[..., 6:9]
        lag = jnp.exp(-dt / self.time_constant)
        next_a = lag * a + (1 - lag) * command
        next_p = p + v * dt + 0.5 * a * dt * dt
        next_v = v + 0.5 * (a + next_a) * dt
        rotation = acceleration_attitude(next_a, next_v, state.rotation, self.gravity)
        return PointMassState(next_p, next_v, next_a, rotation)
