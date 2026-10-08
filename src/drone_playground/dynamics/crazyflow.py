"""Model selection and stepping for the four pinned Crazyflow dynamics."""

from dataclasses import dataclass

import jax.numpy as jnp

from drone_playground.control.setpoints import AttitudeSetpoint, ForceTorque, MotorRPM, RateSetpoint
from drone_playground.dynamics.base import DynamicsBackend, physics_steps
from drone_playground.dynamics.parameters import (
    parameter_ranges,
    physical_parameters,
    randomize_parameters,
)


@dataclass
class CrazyflowModel(DynamicsBackend):
    """Expose Crazyflow's native dynamics through the shared flight interface.

    Args:
        forward: Name of the selected Crazyflow forward dynamics model.
        drone: Crazyflow aircraft parameter identifier.
        backward: Must be ``direct`` for Crazyflow dynamics.
        domain_randomization: Settings for sampling physical parameter variations.
        parameter_overrides: Optional fixed physical parameter changes.
    """

    forward: str = "so_rpy"
    drone: str = "cf2x_L250"
    backward: str = "direct"
    domain_randomization: dict | None = None
    parameter_overrides: dict | None = None

    def __post_init__(self):
        """Validate and prepare the CrazyflowModel instance after initialization."""
        from crazyflow.dynamics.core import Dynamics

        Dynamics(self.forward)
        if self.backward != "direct":
            raise ValueError(
                "Crazyflow currently supports its direct derivative; select LOTF for the "
                "named surrogate"
            )
        if self.domain_randomization is None:
            self.domain_randomization = {"enabled": False}
        parameter_ranges(self.domain_randomization, self.parameter_overrides)

    def _validate_randomization(self, params) -> None:
        ranges = parameter_ranges(self.domain_randomization, self.parameter_overrides)
        supported = {
            "mass": hasattr(params, "mass"),
            # Fitted RPY dynamics use identified attitude coefficients. Their J
            # fields only affect an external torque disturbance, absent here.
            "inertia": self.forward == "first_principles" and hasattr(params, "J_inv"),
            "motor_strength": hasattr(params, "cmd_f_coef") or hasattr(params, "rpm2thrust"),
            "drag": hasattr(params, "drag_matrix"),
        }
        unavailable = [name for name in ranges if not supported[name]]
        if unavailable:
            raise ValueError(
                f"Dynamics {self.forward} does not expose randomizable parameters: {unavailable}"
            )

    def randomize(self, data, key):
        """Sample parameters in the simulation container at a training reset."""
        return randomize_parameters(
            data, key, parameter_ranges(self.domain_randomization, self.parameter_overrides)
        )

    physical_parameters = staticmethod(physical_parameters)

    def bind(self, sim, control_mode=None):
        from crazyflow.sim.pipeline import insert_fn_before
        from jax.scipy.spatial.transform import Rotation

        if control_mode is not None:
            from collections import OrderedDict

            from crazyflow.control import Control
            from crazyflow.sim.data import SimControls
            from crazyflow.sim.sim import build_control_fns

            mode = Control(control_mode)
            if mode != sim.control:
                if self.forward != "first_principles":
                    raise ValueError(
                        "Body-rate and actuator inputs require first-principles Crazyflow dynamics"
                    )
                old = sim.data.controls
                cadence = old.attitude.freq if old.attitude is not None else sim.freq
                force_cadence = old.force_torque.freq if old.force_torque is not None else cadence
                controls = SimControls.create(
                    sim.n_worlds,
                    sim.n_drones,
                    mode,
                    sim.drone,
                    None,
                    cadence,
                    cadence,
                    force_cadence,
                    sim.device,
                )
                old_stages = {name for name, _ in build_control_fns(sim.control, sim.dynamics)}
                sim.step_pipeline = OrderedDict(
                    [
                        *build_control_fns(mode, sim.dynamics),
                        *(
                            (name, fn)
                            for name, fn in sim.step_pipeline.items()
                            if name not in old_stages
                        ),
                    ]
                )
                sim.control = mode
                sim.data = sim.data.replace(controls=controls)
                sim.build_default_data()
        self.sim = sim
        self._validate_randomization(sim.default_data.params)

        def apply_external_wrench(data):
            if "external_force_world_n" not in data.plugins:
                return data
            from drone_playground.dynamics.disturbances import external_wrench

            force, body_torque = external_wrench(data)
            # The source integrator consumes world-frame force and torque.
            torque = Rotation.from_quat(data.states.quat[0, 0]).apply(body_torque)
            return data.replace(
                states=data.states.replace(
                    force=jnp.broadcast_to(force, data.states.force.shape),
                    torque=jnp.broadcast_to(torque, data.states.torque.shape),
                )
            )

        insert_fn_before(sim.step_pipeline, "integration", apply_external_wrench)
        self.step_fn = sim.build_step_fn()
        return self

    def step(self, state, control, dt):
        """Stage a physical control and call Crazyflow's public step pipeline."""
        from crazyflow.control import Control
        from crazyflow.sim import functional

        count = physics_steps(dt, self.sim.freq)
        mode = state.controls.mode
        if isinstance(control, AttitudeSetpoint) and mode == Control.attitude:
            state = functional.attitude_control(state, control.as_array()[None, None])
        elif isinstance(control, RateSetpoint) and mode == Control.body_rate:
            # Crazyflow uses [wx, wy, wz, T]; LOTF uses [T, wx, wy, wz].
            native = jnp.concatenate(
                (jnp.asarray(control.body_rates), jnp.asarray(control.thrust)[..., None]), axis=-1
            )
            state = functional.body_rate_control(state, native[None, None])
        elif isinstance(control, ForceTorque) and mode == Control.force_torque:
            state = functional.force_torque_control(state, control.as_array()[None, None])
        elif isinstance(control, MotorRPM) and mode == Control.rotor_vel:
            state = functional.rotor_vel_control(state, control.as_array()[None, None])
        else:
            raise TypeError(
                f"Crazyflow control mode {mode} cannot execute {type(control).__name__}"
            )
        return self.step_fn(state, count)
