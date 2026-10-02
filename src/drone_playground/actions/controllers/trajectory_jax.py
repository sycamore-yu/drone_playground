"""Differentiable PD reference tracking with an explicit reference lookahead."""

import math

import jax.numpy as jnp

from drone_playground.actions.controllers.crazyflow import AttitudeControl


class JaxTrajectoryTracking(AttitudeControl):
    name = "jax_trajectory_tracking"
    input_kind = "trajectory"
    differentiable = True

    def __init__(self, kp=4.0, kv=3.0, max_acceleration=5.0, lead_seconds=0.1):
        if (
            not all(
                math.isfinite(x)
                for x in (kp, kv, max_acceleration, lead_seconds)
            )
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
