"""LOTF simplified translation and exact body-rate rotation, without its runtime."""

import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def lotf_step(position, quaternion, velocity, command, dt, *, mass):
    """Predict position, xyzw orientation and velocity from [thrust N, rates rad/s].

    This is the analytic update of LOTF simplified_dyn: translation uses the
    current orientation throughout the step and rotation integrates the requested
    constant body rates. No Betaflight, motor state or residual learner is needed.
    """
    rotation = Rotation.from_quat(quaternion)
    thrust = jnp.zeros_like(position).at[..., 2].set(command[..., 0] / mass)
    acceleration = rotation.apply(thrust) + jnp.array([0.0, 0.0, -9.81])
    next_position = position + velocity * dt + 0.5 * acceleration * dt**2
    next_velocity = velocity + acceleration * dt
    rotvec = command[..., 1:] * dt
    squared = jnp.sum(rotvec**2, axis=-1, keepdims=True)
    angle = jnp.sqrt(jnp.maximum(squared, 1e-12))
    scale = jnp.where(squared < 1e-6, 0.5 - squared / 48, jnp.sin(angle / 2) / angle)
    real = jnp.where(squared < 1e-6, 1 - squared / 8, jnp.cos(angle / 2))
    increment = Rotation.from_quat(jnp.concatenate([rotvec * scale, real], -1))
    next_rotation = rotation * increment
    return next_position, next_rotation.as_quat(), next_velocity
