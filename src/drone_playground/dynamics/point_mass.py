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

from drone_playground.control.setpoints import StateSetpoint
from drone_playground.dynamics.base import DynamicsBackend


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
    up = _unit(
        acceleration + jnp.array([0.0, 0.0, gravity]),
        jnp.array([0.0, 0.0, 1.0]),
    )
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
    measurement_key: jax.Array
    motor_strength: jax.Array
    lag_factor: jax.Array
    elapsed_time: jax.Array

    @classmethod
    def create(cls, position):
        position = jnp.asarray(position, jnp.float32)
        return cls(
            position,
            jnp.zeros_like(position),
            jnp.zeros_like(position),
            jnp.broadcast_to(jnp.eye(3), (*position.shape[:-1], 3, 3)),
            jnp.broadcast_to(jax.random.PRNGKey(0), (*position.shape[:-1], 2)),
            jnp.ones(position.shape[:-1]),
            jnp.ones(position.shape[:-1]),
            jnp.zeros(position.shape[:-1]),
        )

    @classmethod
    def from_vector(cls, vector):
        return cls.create(vector[..., :3]).replace(vel=vector[..., 3:6], acc=vector[..., 6:9])

    def vector(self):
        return jnp.concatenate([self.pos, self.vel, self.acc], axis=-1)


@dataclass(frozen=True)
class PointMassLag(DynamicsBackend):
    time_constant: float = 1 / 12
    backward: str = "exponential"
    decay_rate: float = -math.log(0.4)
    gravity: float = 9.80665
    forward: str = "point_mass_lag"
    drone: str = "point_mass_lag"
    domain_randomization: dict | None = None
    parameter_overrides: dict | None = None
    disturbance: dict | None = None

    def __post_init__(self):
        if (
            not math.isfinite(self.time_constant)
            or not math.isfinite(self.decay_rate)
            or self.time_constant <= 0
            or self.decay_rate < 0
        ):
            raise ValueError("Lag must be positive and gradient decay nonnegative")
        if self.backward not in ("direct", "exponential"):
            raise ValueError(f"Unknown point-mass derivative: {self.backward}")
        self._parameter_ranges()
        if self.disturbance and set(self.disturbance) - {
            "acceleration_world_mps2",
            "gust_std_mps2",
            "gust_period_s",
        }:
            raise ValueError(
                "Acceleration-driven point mass supports acceleration disturbances; force/torque need a rigid-body model"
            )

    def _parameter_ranges(self):
        settings = self.domain_randomization or {"enabled": False}
        if set(settings) - {"enabled", "dynamics"} or not isinstance(settings.get("enabled"), bool):
            raise ValueError("Unsupported domain_randomization settings")
        declared = settings.get("dynamics", {})
        if not isinstance(declared, dict) or set(declared) - {
            "mass",
            "inertia",
            "motor_strength",
            "lag",
            "drag",
        }:
            raise ValueError("Unknown dynamics randomization fields")
        ranges = (
            {k: v for k, v in settings.get("dynamics", {}).items() if v is not None}
            if settings["enabled"]
            else {}
        )
        ranges.update({k: [v, v] for k, v in (self.parameter_overrides or {}).items()})
        if set(ranges) - {"motor_strength", "lag"}:
            raise ValueError(
                "Point mass domain_randomization supports motor_strength and lag; mass/inertia are absent from acceleration-driven equations"
            )
        for bounds in ranges.values():
            if (
                len(bounds) != 2
                or not all(math.isfinite(v) for v in bounds)
                or not 0 < bounds[0] <= bounds[1]
            ):
                raise ValueError("Dynamics parameter factors must be finite, positive and ordered")
        if settings["enabled"] and not ranges:
            raise ValueError("Enabled domain_randomization needs effective parameter ranges")
        return ranges

    def randomize(self, state, key):
        changes = {}
        for index, (name, bounds) in enumerate(self._parameter_ranges().items()):
            value = jax.random.uniform(
                jax.random.fold_in(key, index),
                state.pos.shape[:-1],
                minval=bounds[0],
                maxval=bounds[1],
            )
            changes["lag_factor" if name == "lag" else name] = value
        return state.replace(**changes)

    def step(self, state: PointMassState, control, dt):
        """Advance net acceleration in world coordinates; gravity is already compensated."""
        if not isinstance(control, StateSetpoint) or control.acceleration is None:
            raise TypeError("PointMass control requires a world acceleration StateSetpoint")
        if any(
            value is not None
            for value in (control.position, control.velocity, control.yaw, control.yaw_rate)
        ):
            raise TypeError("PointMass only executes the acceleration field")
        if isinstance(dt, (int, float)) and (not math.isfinite(dt) or dt <= 0):
            raise ValueError("PointMass timestep must be positive and finite")
        command = control.acceleration
        vector = state.vector()
        if self.backward == "exponential":
            vector = scale_state_gradient(vector, jnp.exp(-self.decay_rate * dt))
        p, v, a = vector[..., :3], vector[..., 3:6], vector[..., 6:9]
        lag = jnp.exp(-dt / (self.time_constant * state.lag_factor))[..., None]
        effective_command = command * state.motor_strength[..., None]
        disturbance = self.disturbance or {}
        if disturbance:
            indices = jnp.floor(state.elapsed_time / disturbance.get("gust_period_s", 1.0)).astype(
                jnp.int32
            )
            keys = (
                jax.vmap(jax.random.fold_in)(state.measurement_key, indices)
                if state.measurement_key.ndim == 2
                else jax.random.fold_in(state.measurement_key, indices)
            )
            gust = (
                jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys)
                if keys.ndim == 2
                else jax.random.normal(keys, (3,))
            )
            effective_command += jnp.asarray(
                disturbance.get("acceleration_world_mps2", [0, 0, 0])
            ) + gust * jnp.asarray(disturbance.get("gust_std_mps2", 0.0))
        next_a = lag * a + (1 - lag) * effective_command
        next_p = p + v * dt + 0.5 * a * dt * dt
        next_v = v + 0.5 * (a + next_a) * dt
        rotation = acceleration_attitude(next_a, next_v, state.rotation, self.gravity)
        return state.replace(
            pos=next_p,
            vel=next_v,
            acc=next_a,
            rotation=rotation,
            elapsed_time=state.elapsed_time + dt,
        )
