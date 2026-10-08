"""Differentiable PD reference tracking with an explicit reference lookahead."""

import math

import jax.numpy as jnp

from drone_playground.control.controllers.attitude import AttitudeControl
from drone_playground.control.setpoints import AttitudeSetpoint, StateSetpoint
from drone_playground.references import Trajectory, Waypoint


class JaxTrajectoryTracking(AttitudeControl):
    name = "jax_trajectory_tracking"
    input_kind = "trajectory"
    differentiable = True

    def __init__(
        self, kp=4.0, kv=3.0, max_acceleration=5.0, lead_seconds=0.1, input_kind="trajectory"
    ):
        if (
            not all(math.isfinite(x) for x in (kp, kv, max_acceleration, lead_seconds))
            or min(kp, kv, max_acceleration) <= 0
            or lead_seconds < 0
        ):
            raise ValueError(
                "Finite positive tracking gains and nonnegative reference lead required"
            )
        self.kp, self.kv = kp, kv
        self.max_acceleration, self.lead_seconds = (
            max_acceleration,
            lead_seconds,
        )
        if input_kind not in ("trajectory", "waypoint", "state"):
            raise ValueError("Reference tracking accepts trajectory, waypoint or state")
        self.input_kind = input_kind

    def apply(self, data, value):
        """Track a reference once; already lowered attitude bypasses this stage."""
        if isinstance(value, AttitudeSetpoint):
            return value
        if not isinstance(value, (Trajectory, Waypoint, StateSetpoint)):
            return super().apply(data, value)
        body = {name: getattr(data.states, name)[0, 0] for name in ("pos", "vel", "quat")}
        qx, qy, qz, qw = body["quat"]
        yaw = jnp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        if isinstance(value, Trajectory):
            now = data.core.steps[0, 0] / data.core.freq
            reference = value.sample(now + self.lead_seconds)
            if not value.yaw_defined:
                reference["yaw"] = yaw
        elif isinstance(value, Waypoint):
            if value.positions.shape[0] != 1:
                raise ValueError("Differentiable waypoint tracking requires one current target")
            reference = dict(
                position=value.positions[0],
                velocity=jnp.zeros(3),
                acceleration=jnp.zeros(3),
                yaw=yaw,
            )
        else:
            reference = dict(
                position=body["pos"] if value.position is None else value.position,
                velocity=jnp.zeros(3) if value.velocity is None else value.velocity,
                acceleration=jnp.zeros(3) if value.acceleration is None else value.acceleration,
                yaw=yaw if value.yaw is None else value.yaw,
            )
            if value.yaw_rate is not None:
                raise TypeError("This tracking controller consumes yaw, not yaw-rate commands")
        physical = self.command(body, reference, data.params.mass.reshape(-1)[0])
        return AttitudeSetpoint(rpy=physical[:3], thrust=physical[3])

    def command(self, state, reference, mass):
        pos, vel, quat = (jnp.asarray(state[k]) for k in ("pos", "vel", "quat"))
        acc = (
            jnp.asarray(reference["acceleration"])
            + self.kp * (jnp.asarray(reference["position"]) - pos)
            + self.kv * (jnp.asarray(reference["velocity"]) - vel)
        )
        # The ordinary Euclidean norm has an undefined derivative at hover.
        magnitude = jnp.sqrt(jnp.sum(acc * acc) + 1e-12)
        acc = acc * jnp.minimum(1.0, self.max_acceleration / magnitude)
        force = acc + jnp.array([0.0, 0.0, 9.81])
        z = force / jnp.sqrt(jnp.sum(force * force) + 1e-12)
        yaw = reference["yaw"]
        y = jnp.cross(z, jnp.array([jnp.cos(yaw), jnp.sin(yaw), 0.0]))
        y = y / jnp.sqrt(jnp.sum(y * y) + 1e-12)
        x = jnp.cross(y, z)
        # XYZ Euler angles of the same basis used by the host PD controller.
        angles = jnp.array(
            [
                jnp.arctan2(y[2], z[2]),
                jnp.arctan2(-x[2], jnp.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12)),
                jnp.arctan2(x[1], x[0]),
            ]
        )
        qx, qy, qz, qw = quat
        current_z = jnp.array(
            [
                2 * (qx * qz + qy * qw),
                2 * (qy * qz - qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ]
        )
        thrust = mass * jnp.dot(force, current_z)
        return jnp.clip(
            jnp.concatenate([angles, jnp.atleast_1d(thrust)]),
            self.low,
            self.high,
        )
