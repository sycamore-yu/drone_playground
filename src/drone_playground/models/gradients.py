"""Explicit surrogate derivatives around the complete native control/physics step."""

from functools import partial

import jax
from lotf.objects.quadrotor_obj import simplified_dyn


def lotf_surrogate_step(forward, mass):
    """LOTF's p/R/v analytical derivative, with valid zero tangents for PRNG data.

    Other floating-point state tangents follow the upstream identity rule.
    PRNG and discrete state tangents retain JAX's zero (float0) dtype instead of
    copying primal RNG bytes into the tangent, as the old JAX source did.
    """

    @partial(jax.custom_jvp, nondiff_argnums=(2,))
    def step(state, command, dt):
        return forward(state, command, dt)

    @step.defjvp
    def derivative(dt, primals, tangents):
        state, command = primals
        state_tan, command_tan = tangents
        actual = step(state, command, dt)
        simple_inputs = (state.p, state.R, state.v, command[0] / mass, command[1:], dt)
        simple_tangents = (
            state_tan.p,
            state_tan.R,
            state_tan.v,
            command_tan[0] / mass,
            command_tan[1:],
            0.0,
        )
        _, (dp, dR, dv) = jax.jvp(simplified_dyn, simple_inputs, simple_tangents)
        return actual, state_tan.replace(p=dp, R=dR, v=dv)

    return step


def physical_view(state):
    """Project native LOTF state to replay coordinates; no physics is evaluated."""
    from jax.scipy.spatial.transform import Rotation

    return {
        "pos": state.p,
        "quat": Rotation.from_matrix(state.R).as_quat(),
        "vel": state.v,
        "ang_vel": state.omega,
    }
