"""Acceleration-driven point mass with the predecessor's first-order lag."""

import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.dynamics.lotf import integrate_body_rates


def point_mass_step(position, velocity, acceleration, command, dt, *, time_constant=1 / 12):
    """Predict p/v/a from a world-frame net-acceleration command.

    Gravity is already compensated in the input. The discrete update preserves
    the old PointMassLag's position and trapezoidal velocity formulas; no gradient
    decay is embedded in this mathematical model.
    """
    lag = jnp.exp(-dt / time_constant)
    next_acceleration = lag * acceleration + (1 - lag) * command
    next_position = position + velocity * dt + 0.5 * acceleration * dt**2
    next_velocity = velocity + 0.5 * (acceleration + next_acceleration) * dt
    return next_position, next_velocity, next_acceleration


def point_mass_body_rate_step(
    position, quaternion, velocity, acceleration, command, dt, *, mass, time_constant=1 / 12
):
    """Use CTBR input with kinematic attitude and the existing translational lag.

    This is a reduced closed-loop backward model, not a new forward simulator.
    The due command is [collective thrust N, body rates rad/s]. Attitude follows
    commanded rates; translation retains point_mass_step's acceleration lag.
    Its quaternion derivative is required: treating rates as an acceleration
    vector, or dropping attitude derivatives, would hide rate-to-motion coupling.
    """
    next_quaternion = integrate_body_rates(quaternion, command[..., 1:], dt)
    thrust_body = jnp.zeros_like(position).at[..., 2].set(command[..., 0] / mass)
    requested = Rotation.from_quat(quaternion).apply(thrust_body) + jnp.array([0.0, 0.0, -9.81])
    next_position, next_velocity, next_acceleration = point_mass_step(
        position, velocity, acceleration, requested, dt, time_constant=time_constant
    )
    return next_position, next_quaternion, next_velocity, next_acceleration
