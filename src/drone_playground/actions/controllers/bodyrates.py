"""Physical thrust/body-rate commands; the dynamics owns its motor controller."""

import jax.numpy as jnp
from crazyflow.sim import functional

from drone_playground.actions.controllers.crazyflow import AttitudeControl


class BodyRateControl(AttitudeControl):
    name = "bodyrates"
    input_kind = "thrust_bodyrates"

    def __init__(self, model, name="bodyrates"):
        native = model.native
        self.bounds = (
            jnp.r_[native._thrust_min * 4, -native._omega_max],
            jnp.r_[native._thrust_max * 4, native._omega_max],
        )

    def bind(self, low, high):
        return super().bind(*self.bounds)

    def hover(self, data):
        return jnp.r_[data.params.mass[0] * 9.81, jnp.zeros(3)]

    def apply(self, data, physical):
        # The shared command buffer carries SI thrust/bodyrates unchanged.
        # LOTF.advance consumes it; no Crazyflow attitude controller is executed.
        return functional.attitude_control(data, physical[None, None])
