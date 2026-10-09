"""Connect a backward model to the existing physical tick without a second rollout."""

import math
from functools import partial

import jax

from drone_playground.simulation.dynamics.lotf import lotf_step
from drone_playground.simulation.dynamics.point_mass import point_mass_step


def training_step(env, model=None, options=None):
    """Build a forward-identical step with selected physical-output derivatives.

    PointMass replaces p/v/acceleration derivatives. LOTF replaces p/q/v and the
    derived acceleration. All other output derivatives remain those of the real
    Crazyflow tick. The shared delay buffer supplies the actually due input, so
    gradients do not act earlier than commands do.
    """
    if model is None:
        return env.step
    if env.control_level == "ideal":
        raise ValueError("Backward physical models require physical forward execution")
    options = dict(options or {})
    if model == "point_mass_lag":
        if env.action.level != "acceleration":
            raise ValueError("PointMassLag requires an acceleration action interface")
        if set(options) - {"time_constant"}:
            raise ValueError("Unknown PointMassLag parameter")
        tau = float(options.get("time_constant", 1 / 12))
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("PointMass time constant must be finite and positive")
        predict = partial(point_mass_step, dt=1 / env.sim.freq, time_constant=tau)
    elif model == "lotf":
        if env.action.level != "body_rate":
            raise ValueError("LOTF requires total thrust and body-rate commands")
        if set(options) - {"mass"}:
            raise ValueError("Unknown LOTF parameter")
        mass = float(options.get("mass", env.action.mass))
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("LOTF mass must be finite and positive")
        predict = partial(lotf_step, dt=1 / env.sim.freq, mass=mass)
    else:
        raise ValueError(f"Unknown backward dynamics model: {model}")

    @jax.custom_jvp
    def physics_step(data, acceleration, native_command, model_input):
        return env.native_step(data, acceleration, native_command, model_input)

    @physics_step.defjvp
    def derivative(primals, tangents):
        data, acceleration, _, model_input = primals
        data_tan, acc_tan, _, input_tan = tangents
        actual, tangent = jax.jvp(env.native_step, primals, tangents)
        native_data_tan, _ = tangent
        p, q, v = data.states.pos[:, 0], data.states.quat[:, 0], data.states.vel[:, 0]
        dp, dq, dv = (
            data_tan.states.pos[:, 0],
            data_tan.states.quat[:, 0],
            data_tan.states.vel[:, 0],
        )
        if model == "point_mass_lag":
            _, (next_dp, next_dv, next_da) = jax.jvp(
                predict, (p, v, acceleration, model_input), (dp, dv, acc_tan, input_tan)
            )
            modeled = {"pos": next_dp[:, None], "vel": next_dv[:, None]}
        else:
            _, (next_dp, next_dq, next_dv) = jax.jvp(
                predict, (p, q, v, model_input), (dp, dq, dv, input_tan)
            )
            modeled = {"pos": next_dp[:, None], "quat": next_dq[:, None], "vel": next_dv[:, None]}
            next_da = (next_dv - dv) * env.sim.freq
        native_data_tan = native_data_tan.replace(states=native_data_tan.states.replace(**modeled))
        return actual, (native_data_tan, next_da)

    return env.build_step(physics_step)
