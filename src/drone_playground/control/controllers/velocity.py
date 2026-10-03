"""Explicit world-velocity/yaw tracking above the pinned attitude controller.

This module has no scene, goal, reference path, obstacle map or safety shield.
It converts the learned velocity command to attitude/thrust using current
velocity and known mass. The same full Crazyflow rigid-body plant advances.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp

from drone_playground.control.setpoints import AttitudeSetpoint, StateSetpoint
from drone_playground.environments.scenes.geometry import euclidean_norm


@dataclass
class VelocityControl:
    name: str = "velocity_yaw"
    input_kind: str = "state"
    max_speed: float = 20.0
    velocity_gain: tuple[float, float, float] = (3.0, 3.0, 5.0)
    acceleration_limit: tuple[float, float, float] = (6.0, 6.0, 6.0)
    tilt_limit: float = 0.55
    gravity: float = 9.81

    input_fields = ("vx", "vy", "vz", "yaw")
    input_units = ("m/s", "m/s", "m/s", "rad")
    input_frame = "world"
    native_mode = "attitude"
    differentiable = True

    def __post_init__(self):
        if self.input_kind != "state":
            raise ValueError("VelocityControl accepts state setpoints")
        if self.max_speed <= 0 or not 0 < self.tilt_limit < math.pi / 2:
            raise ValueError(
                "Velocity control requires positive speed and a nonsingular tilt limit"
            )
        if len(self.velocity_gain) != 3 or len(self.acceleration_limit) != 3:
            raise ValueError("Three axis gains and acceleration limits are required")
        if any(
            not math.isfinite(x) or x <= 0 for x in (*self.velocity_gain, *self.acceleration_limit)
        ):
            raise ValueError("Velocity gains and acceleration limits must be finite and positive")

    def bind(self, low, high, model=None):
        del model
        self.attitude_low, self.attitude_high = jnp.asarray(low), jnp.asarray(high)
        self.low = jnp.array([-self.max_speed] * 3 + [-math.pi])
        self.high = -self.low
        return self

    def normalize(self, physical):
        return 2 * (physical - self.low) / (self.high - self.low) - 1

    def setpoint(self, physical):
        return StateSetpoint(velocity=physical[..., :3], yaw=physical[..., 3])

    def input_values(self, value):
        if not isinstance(value, StateSetpoint) or value.velocity is None or value.yaw is None:
            raise TypeError("VelocityControl requires velocity and yaw StateSetpoint fields")
        if any(field is not None for field in (value.position, value.acceleration, value.yaw_rate)):
            raise TypeError("VelocityControl accepts only velocity and yaw fields")
        return jnp.concatenate((value.velocity, jnp.asarray(value.yaw)[..., None]), axis=-1)

    def contract(self):
        return dict(
            kind=self.input_kind,
            fields=self.input_fields,
            units=self.input_units,
            frame=self.input_frame,
        )

    def physical_action(self, action):
        command = self.low + (jnp.clip(action, -1, 1) + 1) * 0.5 * (self.high - self.low)
        velocity = command[:3] * jnp.minimum(
            1.0, self.max_speed / jnp.maximum(euclidean_norm(command[:3]), 1e-6)
        )
        return command.at[:3].set(velocity)

    def hover(self, data):
        del data
        return jnp.zeros(4)

    def attitude_command(self, velocity, command, mass):
        acceleration = jnp.clip(
            jnp.asarray(self.velocity_gain) * (command[:3] - velocity),
            -jnp.asarray(self.acceleration_limit),
            jnp.asarray(self.acceleration_limit),
        )
        force = mass * (acceleration + jnp.array([0.0, 0.0, self.gravity]))
        thrust = jnp.maximum(euclidean_norm(force), 1e-6)
        yaw = command[3]
        horizontal_x = jnp.cos(yaw) * force[0] + jnp.sin(yaw) * force[1]
        horizontal_y = -jnp.sin(yaw) * force[0] + jnp.cos(yaw) * force[1]
        roll = -jnp.arcsin(jnp.clip(horizontal_y / thrust, -0.999, 0.999))
        pitch = jnp.arctan2(horizontal_x, force[2])
        output = jnp.array(
            [
                jnp.clip(roll, -self.tilt_limit, self.tilt_limit),
                jnp.clip(pitch, -self.tilt_limit, self.tilt_limit),
                yaw,
                thrust,
            ]
        )
        return jnp.clip(output, self.attitude_low, self.attitude_high)

    def apply(self, data, command):
        if isinstance(command, AttitudeSetpoint):
            return command
        if isinstance(command, StateSetpoint):
            command = self.input_values(command)
        actual = self.attitude_command(
            data.states.vel[0, 0], command, data.params.mass.reshape(-1)[0]
        )
        return AttitudeSetpoint(rpy=actual[:3], thrust=actual[3])
