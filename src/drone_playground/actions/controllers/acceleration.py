"""Net world-acceleration command contract for the paper point-mass model."""

import jax.numpy as jnp


class AccelerationControl:
    name = "acceleration_passthrough"
    input_kind = "world_acceleration"
    differentiable = True

    def __init__(self, name="acceleration_passthrough"):
        if name != self.name:
            raise ValueError("Unknown acceleration controller preset")

    def physical_action(self, action):
        action = jnp.asarray(action)
        if action.shape[-1] != 3:
            raise ValueError(
                "World acceleration requires exactly three components"
            )
        return action
