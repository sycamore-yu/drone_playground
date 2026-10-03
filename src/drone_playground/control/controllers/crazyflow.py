"""The single control module owns command scaling and the native flight-control chain."""

import jax.numpy as jnp

from drone_playground.control.setpoints import AttitudeSetpoint


class AttitudeControl:
    name = "crazyflow_attitude"
    input_kind = "attitude"
    differentiable = True
    input_fields = ("roll", "pitch", "yaw", "thrust")
    input_units = ("rad", "rad", "rad", "N")
    input_frame = "world attitude / body thrust"
    native_mode = "attitude"

    def __init__(self, input_kind="attitude"):
        if input_kind != "attitude":
            raise ValueError("AttitudeControl accepts the attitude input kind")

    def bind(self, low, high, model=None):
        del model
        self.low, self.high = jnp.asarray(low), jnp.asarray(high)
        return self

    def normalize(self, physical):
        """Encode physical controls for the configured policy action space."""
        return 2 * (physical - self.low) / (self.high - self.low) - 1

    def setpoint(self, physical):
        """Give an array its declared physical meaning at the policy boundary."""
        return AttitudeSetpoint(rpy=physical[..., :3], thrust=physical[..., 3])

    def input_values(self, value):
        """Read a typed input without inferring units from its vector width."""
        if not isinstance(value, AttitudeSetpoint):
            raise TypeError("AttitudeControl requires AttitudeSetpoint")
        return value.as_array()

    def contract(self):
        """Describe only this controller's actual input, not a global command list."""
        return dict(
            kind="attitude",
            fields=self.input_fields,
            units=self.input_units,
            frame=self.input_frame,
        )

    def physical_action(self, action):
        return self.low + (jnp.clip(action, -1.0, 1.0) + 1) * 0.5 * (self.high - self.low)

    def apply(self, data, physical):
        """Return the physical input; the dynamics stages native control buffers."""
        del data
        if isinstance(physical, AttitudeSetpoint):
            return physical
        return self.setpoint(physical)
