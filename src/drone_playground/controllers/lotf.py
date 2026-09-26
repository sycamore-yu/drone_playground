"""The native LOTF flight-control preset, including its motor command mapping."""

import jax.numpy as jnp


class LOTFControl:
    name = "lotf_betaflight"
    input_kind = "thrust_bodyrates"
    differentiable = True

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model.native, name)

    def motor_commands(self, state, command, dt):
        return self.model.native._llc_betaflight(state, command[0], command[1:], dt)

    def step(self, state, thrust, bodyrates, res_model_params, dt):
        del res_model_params
        command = jnp.concatenate([jnp.atleast_1d(thrust), bodyrates])
        return self.model.step(state, command, dt)
