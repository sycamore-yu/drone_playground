"""Acceleration-driven point mass with the predecessor's first-order lag."""

import jax.numpy as jnp


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
