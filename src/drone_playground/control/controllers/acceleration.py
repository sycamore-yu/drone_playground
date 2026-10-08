"""Net world-acceleration command contract for the paper point-mass model."""

import jax.numpy as jnp

from drone_playground.control.setpoints import StateSetpoint


class AccelerationControl:
    """Pass world-frame net acceleration to the point-mass dynamics.

    Args:
        input_kind: Must be ``state``; only the acceleration field is accepted.
    """

    name = "acceleration_passthrough"
    input_kind = "state"
    differentiable = True
    input_fields = ("ax", "ay", "az")
    input_units = ("m/s^2",) * 3
    input_frame = "world; gravity-compensated net acceleration"
    native_mode = "acceleration"

    def __init__(self, input_kind="state"):
        if input_kind != "state":
            raise ValueError("AccelerationControl accepts state setpoints")
        self.low = jnp.full(3, -jnp.inf)
        self.high = jnp.full(3, jnp.inf)

    def normalize(self, physical):
        """The existing point-cloud head already produces physical acceleration."""
        return physical

    def setpoint(self, physical):
        return StateSetpoint(acceleration=physical)

    def input_values(self, value):
        if not isinstance(value, StateSetpoint) or value.acceleration is None:
            raise TypeError("AccelerationControl requires an acceleration StateSetpoint")
        if any(
            field is not None
            for field in (value.position, value.velocity, value.yaw, value.yaw_rate)
        ):
            raise TypeError("AccelerationControl executes only acceleration")
        return value.acceleration

    def contract(self):
        return dict(
            kind=self.input_kind,
            fields=self.input_fields,
            units=self.input_units,
            frame=self.input_frame,
        )

    def physical_action(self, action):
        action = jnp.asarray(action)
        if action.shape[-1] != 3:
            raise ValueError("World acceleration requires exactly three components")
        return action

    def apply(self, state, physical):
        """Give the point-mass dynamics a typed net acceleration input."""
        del state
        if isinstance(physical, StateSetpoint):
            self.input_values(physical)
            return physical
        return self.setpoint(physical)
