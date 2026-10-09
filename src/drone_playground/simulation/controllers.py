"""Standalone tracking controllers with nominal parameters and explicit state."""

import math

import jax.numpy as jnp
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude
from crazyflow.dynamics import load_params
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.methods import Setpoint


class MellingerController:
    """Official Crazyflow position controller with thrust expressed in newtons."""

    execution = "jax"
    output_level = "attitude_thrust"
    reference_offsets = (0.0,)

    def __init__(self, drone="cf21B_500", frequency=50.0, yaw=0.0):
        """Bind the controller to nominal hardware and a control frequency."""
        self.frequency, self.yaw = frequency, yaw
        self.convert = parametrize(state2attitude, drone, xp=jnp)
        self.mass = jnp.asarray(load_params("first_principles", drone)["mass"])
        params = self.convert.keywords
        self.mass_thrust = params["pwm_max"] / (4 * params["thrust_max"])

    def initialize_memory(self, batch, seed=0):
        """Allocate the integral error per world."""
        return jnp.zeros((batch, 3))

    def _convert(self, physics, reference, memory, yaw):
        pos, vel, acc = reference
        cmd = jnp.zeros((*pos.shape[:-1], 16)).at[..., :3].set(pos)
        cmd = cmd.at[..., 3:6].set(vel).at[..., 6:9].set(acc)
        cmd = cmd.at[..., 11].set(jnp.sin(yaw / 2)).at[..., 12].set(jnp.cos(yaw / 2))
        output, memory = self.convert(
            physics.states.pos[:, 0],
            physics.states.quat[:, 0],
            physics.states.vel[:, 0],
            cmd,
            memory,
            ctrl_freq=self.frequency,
            mass=self.mass,
            mass_thrust=self.mass_thrust,
        )
        return Setpoint(output), memory

    def __call__(self, physics, reference, memory):
        """Track position, velocity and acceleration using nominal parameters."""
        return self._convert(physics, reference, memory, self.yaw)

    def acceleration(self, physics, acceleration, yaw):
        """Convert a net world acceleration using the official formula."""
        pos, vel = physics.states.pos[:, 0], physics.states.vel[:, 0]
        return self._convert(physics, (pos, vel, acceleration), jnp.zeros_like(pos), yaw)[0]

    def close(self):
        """No external resources are allocated."""


def so3_control(position, velocity, acceleration, reference, *, mass, kx, kv, yaw=0.0):
    """Compute EGO SO3Control force and orientation in SI units.

    Source: ZJU-FAST-Lab/ego-planner bfda512, SO3Control.cpp. Its actual tilt
    limit is PI/2. The reference uses finite values instead of NaN enable flags.
    Returned orientation uses the xyzw quaternion convention.
    """
    pos, vel, acc = reference
    error = pos - position + vel - velocity + acc - acceleration
    ka = jnp.where(jnp.abs(error) > 3.0, 0.0, jnp.abs(error) * 0.2)
    gravity = jnp.array([0.0, 0.0, mass * 9.81])
    force = gravity + jnp.asarray(kx) * (pos - position) + jnp.asarray(kv) * (vel - velocity)
    force = force + mass * (ka * (acc - acceleration) + acc)
    correction = force - gravity
    scale = -mass * 9.81 / jnp.minimum(correction[..., 2], -1e-6)
    force = jnp.where(
        (force[..., 2] < 0)[..., None], scale[..., None] * correction + gravity, force
    )
    norm = jnp.linalg.norm(force, axis=-1, keepdims=True)
    up = jnp.where(norm > 1e-6, force / jnp.maximum(norm, 1e-6), jnp.array([0.0, 0.0, 1.0]))
    heading = jnp.broadcast_to(jnp.array([jnp.cos(yaw), jnp.sin(yaw), 0.0]), up.shape)
    side = jnp.cross(up, heading)
    alternate = jnp.cross(up, jnp.array([0.0, 1.0, 0.0]))
    side = jnp.where((jnp.linalg.norm(side, axis=-1) > 1e-6)[..., None], side, alternate)
    side = side / jnp.maximum(jnp.linalg.norm(side, axis=-1, keepdims=True), 1e-6)
    matrix = jnp.stack([jnp.cross(side, up), side, up], axis=-1)
    return force, Rotation.from_matrix(matrix).as_quat()


class SO3Controller:
    """EGO's SO3 outer loop followed by Crazyflow's attitude controller."""

    execution = "jax"
    output_level = "attitude_thrust"
    reference_offsets = (0.0,)

    def __init__(
        self, drone="cf21B_500", frequency=50.0, kx=(5.7, 5.7, 6.2), kv=(3.4, 3.4, 4.0), yaw=0.0
    ):
        """Bind nominal mass and acceleration-normalized position/velocity gains."""
        self.mass = float(np.asarray(load_params("first_principles", drone)["mass"]).item())
        self.frequency, self.yaw = frequency, yaw
        self.kx, self.kv = jnp.asarray(kx) * self.mass, jnp.asarray(kv) * self.mass

    def initialize_memory(self, batch, seed=0):
        """Store last velocity and its validity for acceleration estimation."""
        return jnp.zeros((batch, 4))

    def __call__(self, physics, reference, memory):
        """Return desired attitude and projected collective thrust."""
        velocity = physics.states.vel[:, 0]
        acceleration = jnp.where(memory[:, 3:] > 0, (velocity - memory[:, :3]) * self.frequency, 0)
        force, quaternion = so3_control(
            physics.states.pos[:, 0],
            velocity,
            acceleration,
            reference,
            mass=self.mass,
            kx=self.kx,
            kv=self.kv,
            yaw=self.yaw,
        )
        current_z = Rotation.from_quat(physics.states.quat[:, 0]).as_matrix()[..., :, 2]
        thrust = jnp.maximum(jnp.sum(force * current_z, axis=-1), 0.0)
        command = jnp.concatenate(
            [Rotation.from_quat(quaternion).as_euler("xyz"), thrust[:, None]], -1
        )
        memory = jnp.concatenate([velocity, jnp.ones((len(velocity), 1))], -1)
        return Setpoint(command), memory

    def close(self):
        """No external resources are allocated."""


class IdealTracking:
    """SUPER's zero-error reference response, explicitly selected at construction."""

    execution = "host"
    output_level = "ideal"

    def __init__(
        self, drone="cf21B_500", frequency=50.0, yaw=0.0, physics_hz=500, action_delay_s=0.0
    ):
        """Configure reference yaw without constructing another physics engine."""
        self.yaw = yaw
        from drone_playground.simulation.delay import delay_range

        count = math.ceil(delay_range(action_delay_s)[1] * physics_hz) + round(
            physics_hz / frequency
        )
        self.reference_offsets = (np.arange(count) + 1) / physics_hz

    def initialize_memory(self, batch, seed=0):
        """Ideal tracking has no controller memory."""
        return jnp.zeros((batch, 0))

    def __call__(self, physics, reference, memory):
        """Return requested position, velocity, acceleration and yaw."""
        pos, vel, acc = reference
        if pos.ndim == 2:
            pos, vel, acc = (
                jnp.broadcast_to(x[:, None], (len(x), len(self.reference_offsets), 3))
                for x in reference
            )
        value = jnp.concatenate([pos, vel, acc, jnp.full((*pos.shape[:-1], 1), self.yaw)], -1)
        return Setpoint(value.reshape(len(value), -1), level="ideal"), memory

    def close(self):
        """No external resources are allocated."""
