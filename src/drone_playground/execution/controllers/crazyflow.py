"""The single control module owns command scaling and the native flight-control chain."""

import jax.numpy as jnp
from crazyflow.sim import functional as functional


class AttitudeControl:
    name = "crazyflow_attitude"
    input_kind = "attitude_thrust"
    differentiable = True

    def bind(self, low, high):
        self.low, self.high = jnp.asarray(low), jnp.asarray(high)
        return self

    def physical_action(self, action):
        return self.low + (jnp.clip(action, -1.0, 1.0) + 1) * 0.5 * (self.high - self.low)

    def apply(self, data, physical):
        return functional.attitude_control(data, physical[None, None])
