"""Typed flight-control inputs in SI units, with explicit coordinate semantics.

The containers are JAX pytrees. Host transports validate values at their boundary;
compiled control code keeps the same types without converting tracers to NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    # Native ROS processes only serialize SI values; they do not run JAX physics.
    jax, jnp = None, np


@dataclass(frozen=True)
class StateSetpoint:
    """World-frame desired motion; absent fields are not controlled.

    Acceleration is net acceleration in m/s², with gravity already compensated.
    Yaw and yaw rate are radians and radians per second about the world up axis.
    """

    position: jax.Array | None = None
    velocity: jax.Array | None = None
    acceleration: jax.Array | None = None
    yaw: jax.Array | None = None
    yaw_rate: jax.Array | None = None
    kind: ClassVar[str] = "state"
    frame: ClassVar[str] = "world"


@dataclass(frozen=True)
class AttitudeSetpoint:
    """World roll/pitch/yaw in radians and collective body thrust in newtons."""

    rpy: jax.Array
    thrust: jax.Array
    kind: ClassVar[str] = "attitude"
    frame: ClassVar[str] = "world attitude / body thrust"

    def as_array(self):
        """Pack the native Crazyflow [roll, pitch, yaw, thrust] input."""
        return jnp.concatenate(
            (jnp.asarray(self.rpy), jnp.asarray(self.thrust)[..., None]), axis=-1
        )


@dataclass(frozen=True)
class RateSetpoint:
    """Collective thrust in newtons and body angular velocity in radians/s."""

    thrust: jax.Array
    body_rates: jax.Array
    kind: ClassVar[str] = "rates"
    frame: ClassVar[str] = "body"

    def as_array(self):
        """Pack [collective thrust, roll rate, pitch rate, yaw rate]."""
        return jnp.concatenate(
            (jnp.asarray(self.thrust)[..., None], jnp.asarray(self.body_rates)), axis=-1
        )


@dataclass(frozen=True)
class ForceTorque:
    """Collective body thrust in newtons and body torque in newton-meters."""

    thrust: jax.Array
    torque: jax.Array
    kind: ClassVar[str] = "force_torque"
    frame: ClassVar[str] = "body"

    def as_array(self):
        """Pack the native collective-thrust and body-torque input."""
        return jnp.concatenate(
            (jnp.asarray(self.thrust)[..., None], jnp.asarray(self.torque)), axis=-1
        )


@dataclass(frozen=True)
class MotorRPM:
    """Four rotor-speed inputs in revolutions per minute, in native motor order."""

    rpm: jax.Array
    kind: ClassVar[str] = "motor_rpm"
    frame: ClassVar[str] = "rotor"

    def as_array(self):
        """Return the native motor-speed vector."""
        return jnp.asarray(self.rpm)


Setpoint = StateSetpoint | AttitudeSetpoint | RateSetpoint
Actuation = ForceTorque | MotorRPM

if jax is not None:
    for _type in (StateSetpoint, AttitudeSetpoint, RateSetpoint, ForceTorque, MotorRPM):
        jax.tree_util.register_dataclass(_type)


def validate_setpoint(value: Setpoint | Actuation) -> None:
    """Validate one unbatched value before host serialization or execution."""
    if not isinstance(
        value, (StateSetpoint, AttitudeSetpoint, RateSetpoint, ForceTorque, MotorRPM)
    ):
        raise TypeError("Expected a typed Setpoint or Actuation input")
    supplied = {field.name: getattr(value, field.name) for field in fields(value)}
    supplied = {name: component for name, component in supplied.items() if component is not None}
    if not supplied:
        raise ValueError("StateSetpoint requires at least one controlled field")
    for name, component in supplied.items():
        array = np.asarray(component)
        if not np.isfinite(array).all():
            raise ValueError(f"Setpoint field {name} must be finite")
        if name in ("position", "velocity", "acceleration", "rpy", "body_rates", "torque"):
            if array.shape != (3,):
                raise ValueError(f"Setpoint field {name} must contain three components")
        elif name == "rpm":
            if array.shape != (4,):
                raise ValueError("MotorRPM must contain four rotor speeds")
        elif array.shape != ():
            raise ValueError(f"Setpoint field {name} must be scalar")
