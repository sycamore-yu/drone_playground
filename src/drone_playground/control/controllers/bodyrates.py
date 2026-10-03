"""Physical thrust/body-rate commands; the dynamics owns its motor controller."""

import jax.numpy as jnp

from drone_playground.control.controllers.crazyflow import AttitudeControl
from drone_playground.control.setpoints import RateSetpoint


class BodyRateControl(AttitudeControl):
    name = "bodyrates"
    input_kind = "rates"
    input_fields = ("thrust", "roll_rate", "pitch_rate", "yaw_rate")
    input_units = ("N", "rad/s", "rad/s", "rad/s")
    input_frame = "body"
    native_mode = "body_rate"

    def __init__(self, max_body_rates=None, input_kind="rates"):
        if input_kind != "rates":
            raise ValueError("BodyRateControl accepts the rates input kind")
        self.max_body_rates = max_body_rates

    def bind(self, low, high, model=None):
        native = getattr(model, "native", None)
        if native is not None:
            low = jnp.r_[native._thrust_min * 4, -native._omega_max]
            high = jnp.r_[native._thrust_max * 4, native._omega_max]
        else:
            if self.max_body_rates is None:
                raise ValueError("Native body-rate control requires configured rate limits")
            rates = jnp.asarray(self.max_body_rates)
            if rates.shape != (3,) or not bool(jnp.all(jnp.isfinite(rates) & (rates > 0))):
                raise ValueError("Body-rate limits require three finite positive values")
            low, high = jnp.r_[low[-1], -rates], jnp.r_[high[-1], rates]
        return super().bind(low, high)

    def setpoint(self, physical):
        """Decode collective thrust and body rates in their native order."""
        return RateSetpoint(thrust=physical[..., 0], body_rates=physical[..., 1:])

    def input_values(self, value):
        """Require rates rather than an equally sized attitude vector."""
        if not isinstance(value, RateSetpoint):
            raise TypeError("BodyRateControl requires RateSetpoint")
        return value.as_array()

    def contract(self):
        return dict(
            kind="rates", fields=self.input_fields, units=self.input_units, frame=self.input_frame
        )

    def hover(self, data):
        return jnp.r_[data.params.mass[0] * 9.81, jnp.zeros(3)]

    def apply(self, data, physical):
        """Keep body rates explicit; never put them in an attitude buffer."""
        del data
        if isinstance(physical, RateSetpoint):
            return physical
        return self.setpoint(physical)
