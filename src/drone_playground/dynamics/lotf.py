"""Pinned LOTF model reuse with selectable native or surrogate differentiation."""

from __future__ import annotations

from copy import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation
from lotf.objects.quadrotor_obj import Quadrotor, simplified_dyn

from drone_playground.dynamics.base import DynamicsBackend
from drone_playground.dynamics.crazyflow import CrazyflowModel
from drone_playground.dynamics.parameters import parameter_ranges, physical_parameters, randomize_parameters


class LOTFModel(DynamicsBackend):
    """LOTF physics in the shared physical-state container, without native tasks.

    Crazyflow supplies the scene/replay container only. Every integration and
    Betaflight motor update below runs the pinned LOTF equations at 1000 Hz.
    Motor state is stored in RPM in that container and converted to rad/s at
    the source boundary; acceleration and angular acceleration are retained.
    """

    scene_reference_dynamics = "first_principles"
    scene_reference_drone = "cf2x_L250"

    def __init__(
        self,
        forward="lotf_high_fidelity",
        backward="analytical_surrogate",
        drone="example_quad",
        learned_residual=False,
        domain_randomization=None,
        parameter_overrides=None,
    ):
        if learned_residual:
            raise ValueError(
                "The approved LOTF scope uses the fixed model with learned residual disabled"
            )
        if forward not in ("lotf_high_fidelity", "lotf_simplified"):
            raise ValueError(f"Unknown LOTF forward model: {forward}")
        if backward not in ("direct", "analytical_surrogate"):
            raise ValueError(f"Unknown LOTF gradient rule: {backward}")
        self.forward, self.backward, self.drone = forward, backward, drone
        self.native = Quadrotor.from_name(
            drone,
            dict(
                use_high_fidelity=forward == "lotf_high_fidelity",
                use_forward_residual=False,
            ),
        )
        self.domain_randomization = {"enabled": False} if domain_randomization is None else domain_randomization
        self.parameter_overrides = parameter_overrides
        parameter_ranges(domain_randomization, parameter_overrides)
        native = self.native

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
                    motors = native._llc_betaflight(
                        s, command[0], command[1:], native._dt_low_level
                    )
                    return (
                        native._full_dyn(s, motors, native._dt_low_level),
                        None,
                    )

                return jax.lax.scan(substep, state, None, length=count)[0]
            p, R, v = simplified_dyn(
                state.p,
                state.R,
                state.v,
                command[0] / native._mass,
                command[1:],
                dt,
            )
            return state.replace(p=p, R=R, v=v)

        self.forward_step = advance
        self._step = (
            lotf_surrogate_step(advance, native._mass)
            if backward == "analytical_surrogate"
            else advance
        )

        @partial(jax.custom_jvp, nondiff_argnums=(4,))
        def parameter_step(state, command, params, wrench, dt):
            return self._parameter_forward(state, command, params, wrench, dt)

        @parameter_step.defjvp
        def derivative(dt, primals, tangents):
            state, command, params, wrench = primals
            state_tan, command_tan, _, _ = tangents
            actual = parameter_step(state, command, params, wrench, dt)
            strength = params.rpm2thrust / (
                self.native._thrust_map[0] * (2 * np.pi / 60) ** 2
            )
            _, (dp, dR, dv) = jax.jvp(
                simplified_dyn,
                (
                    state.p,
                    state.R,
                    state.v,
                    command[0] * strength / params.mass[0],
                    command[1:],
                    dt,
                ),
                (
                    state_tan.p,
                    state_tan.R,
                    state_tan.v,
                    command_tan[0] * strength / params.mass[0],
                    command_tan[1:],
                    0.0,
                ),
            )
            return actual, state_tan.replace(p=dp, R=dR, v=dv)

        self.parameter_step = (
            parameter_step
            if backward == "analytical_surrogate"
            else self._parameter_forward
        )

    def step(self, state, command, dt=0.02):
        return self._step(state, command, float(dt))

    def randomize(self, data, key):
        """Sample effective LOTF parameters in its bound simulation container."""
        return randomize_parameters(
            data, key, parameter_ranges(self.domain_randomization, self.parameter_overrides)
        )

    physical_parameters = staticmethod(physical_parameters)

    def _validate_randomization(self, params):
        ranges = parameter_ranges(self.domain_randomization, self.parameter_overrides)
        unsupported = set(ranges) - {"mass", "motor_strength", "inertia"}
        if "inertia" in ranges and self.forward == "lotf_simplified":
            unsupported.add("inertia")
        if unsupported:
            raise ValueError(
                f"{self.forward} has no effective randomization for {sorted(unsupported)}"
            )

    def bind(self, sim):
        self.sim = sim
        sim.freq = round(1 / self.native._dt_low_level)
        params = sim.data.params.replace(
            mass=jnp.asarray([self.native._mass], jnp.float32),
            J=self.native.inertial_matrix(),
            J_inv=jnp.linalg.inv(self.native.inertial_matrix()),
            rpm2thrust=jnp.asarray(
                self.native._thrust_map[0] * (2 * np.pi / 60) ** 2
            ),
            rpm2torque=jnp.asarray(
                self.native._thrust_map[0]
                * self.native._kappa
                * (2 * np.pi / 60) ** 2
            ),
        )
        states = sim.data.states.replace(
            rotor_vel=jnp.full_like(
                sim.data.states.rotor_vel,
                self.native.hovering_motor_speed * 60 / (2 * np.pi),
            )
        )
        sim.data = sim.data.replace(
            params=params,
            states=states,
            core=sim.data.core.replace(freq=sim.freq),
        )
        sim.build_default_data()
        # Reset physical state to its actual nominal source values. Task reset
        # randomization runs separately, after this backend reset.
        sim.reset_pipeline.pop("reset_randomization", None)
        self._validate_randomization(params)
        return self

    def create_tracking(self, task, duration, freq, device, scene):
        return self.create_navigation(duration, freq, device, scene.takeoff)

    def create_navigation(self, duration, freq, device, start):
        renderer = CrazyflowModel(
            self.scene_reference_dynamics, self.scene_reference_drone
        )
        reference = renderer._drone_reference(duration, freq, device, start)
        self.bind(reference.sim)
        if self.sim.freq % freq:
            reference.close()
            raise ValueError(
                "Task frequency must divide LOTF's 1000 Hz physics clock"
            )
        reference.n_substeps = self.sim.freq // freq
        return reference

    def _parameter_forward(self, state, command, params, wrench, dt):
        native = copy(self.native)
        native._mass = params.mass[0]
        native._inertia = jnp.diag(params.J)
        native._thrust_map = native._thrust_map.at[0].set(
            params.rpm2thrust * (60 / (2 * np.pi)) ** 2
        )
        if native.use_high_fidelity:

            def substep(s, _):
                motors = native._llc_betaflight(
                    s, command[0], command[1:], native._dt_low_level
                )
                return (
                    native._full_dyn(s, motors, native._dt_low_level, *wrench),
                    None,
                )

            return jax.lax.scan(
                substep, state, None, length=round(dt / native._dt_low_level)
            )[0]
        strength = params.rpm2thrust / (
            self.native._thrust_map[0] * (2 * np.pi / 60) ** 2
        )
        p, R, v = simplified_dyn(
            state.p,
            state.R,
            state.v,
            command[0] * strength / native._mass,
            command[1:],
            dt,
            native._gravity + wrench[0] / native._mass,
        )
        return state.replace(p=p, R=R, v=v, omega=command[1:])

    def advance(self, data, substeps):
        state = self.native.create_state(
            data.states.pos[0, 0],
            Rotation.from_quat(data.states.quat[0, 0]).as_matrix(),
            data.states.vel[0, 0],
            omega=data.states.ang_vel[0, 0],
            domega=data.states_deriv.ang_acc[0, 0],
            acc=data.states_deriv.acc[0, 0],
            motor_omega=data.states.rotor_vel[0, 0] * (2 * np.pi / 60),
        )
        command = data.controls.attitude.staged_cmd[0, 0]
        if "external_force_world_n" in data.plugins:
            from drone_playground.environments.randomization import external_wrench

            def tick(native_state, index):
                clock = data.replace(
                    core=data.core.replace(steps=data.core.steps + index)
                )
                return (
                    self.parameter_step(
                        native_state,
                        command,
                        data.params,
                        external_wrench(clock),
                        1 / self.sim.freq,
                    ),
                    None,
                )

            output = jax.lax.scan(tick, state, jnp.arange(substeps))[0]
        else:
            output = self.parameter_step(
                state,
                command,
                data.params,
                (jnp.zeros(3), jnp.zeros(3)),
                substeps / self.sim.freq,
            )
        states = data.states.replace(
            pos=output.p[None, None],
            quat=Rotation.from_matrix(output.R).as_quat()[None, None],
            vel=output.v[None, None],
            ang_vel=output.omega[None, None],
            rotor_vel=output.motor_omega[None, None] * (60 / (2 * np.pi)),
        )
        derivatives = data.states_deriv.replace(
            acc=output.acc[None, None], ang_acc=output.domega[None, None]
        )
        return data.replace(
            states=states,
            states_deriv=derivatives,
            core=data.core.replace(
                steps=data.core.steps + substeps, mjx_synced=jnp.array(False)
            ),
        )


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
        simple_inputs = (
            state.p,
            state.R,
            state.v,
            command[0] / mass,
            command[1:],
            dt,
        )
        simple_tangents = (
            state_tan.p,
            state_tan.R,
            state_tan.v,
            command_tan[0] / mass,
            command_tan[1:],
            0.0,
        )
        _, (dp, dR, dv) = jax.jvp(
            simplified_dyn, simple_inputs, simple_tangents
        )
        return actual, state_tan.replace(p=dp, R=dR, v=dv)

    return step
