"""Pinned LOTF model reuse with selectable native or surrogate differentiation."""

from __future__ import annotations

import jax
import numpy as np
from lotf.objects.quadrotor_obj import Quadrotor, simplified_dyn

from drone_playground.execution.controllers.lotf import LOTFControl

from .gradients import lotf_surrogate_step


class LOTFModel:
    def __init__(
        self,
        forward="lotf_high_fidelity",
        backward="lotf_analytical",
        drone="example_quad",
        learned_residual=False,
    ):
        if learned_residual:
            raise ValueError(
                "The approved LOTF scope uses the fixed model with learned residual disabled"
            )
        if forward not in ("lotf_high_fidelity", "lotf_simplified"):
            raise ValueError(f"Unknown LOTF forward model: {forward}")
        if backward not in ("direct", "lotf_analytical"):
            raise ValueError(f"Unknown LOTF gradient rule: {backward}")
        self.forward, self.backward, self.drone = forward, backward, drone
        self.native = Quadrotor.from_name(
            drone,
            dict(use_high_fidelity=forward == "lotf_high_fidelity", use_forward_residual=False),
        )
        native = self.native
        self.execution = LOTFControl(self)

        def advance(state, command, dt):
            # Calling the upstream primitives preserves every equation while
            # exposing a genuine direct derivative for the explicit control case.
            dt = float(np.round(dt, 5))
            if dt <= 0:
                return state
            if native.use_high_fidelity:
                count = int(np.ceil(dt / native._dt_low_level))
                if not np.isclose(count * native._dt_low_level, dt):
                    raise ValueError(
                        "LOTF high-fidelity dt must be a multiple of its low-level timestep"
                    )

                def substep(s, _):
                    motors = self.execution.motor_commands(s, command, native._dt_low_level)
                    return native._full_dyn(s, motors, native._dt_low_level), None

                return jax.lax.scan(substep, state, None, length=count)[0]
            p, R, v = simplified_dyn(
                state.p, state.R, state.v, command[0] / native._mass, command[1:], dt
            )
            return state.replace(p=p, R=R, v=v)

        self.forward_step = advance
        self._step = (
            lotf_surrogate_step(advance, native._mass) if backward == "lotf_analytical" else advance
        )

    def step(self, state, command, dt=0.02):
        return self._step(state, command, float(dt))
