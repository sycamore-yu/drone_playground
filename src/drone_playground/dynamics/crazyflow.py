"""Model selection and stepping for the four pinned Crazyflow dynamics."""

from dataclasses import dataclass

import jax.numpy as jnp
from crazyflow.envs import FigureEightEnv
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.utils import leaf_replace

from drone_playground.dynamics.parameters import parameter_ranges, physical_parameters, randomize_parameters


@dataclass
class CrazyflowModel:
    forward: str = "so_rpy"
    drone: str = "cf2x_L250"
    backward: str = "direct"
    domain_randomization: dict | None = None
    parameter_overrides: dict | None = None

    def __post_init__(self):
        from crazyflow.dynamics.core import Dynamics

        Dynamics(self.forward)
        if self.backward != "direct":
            raise ValueError(
                "Crazyflow currently supports its direct derivative; select LOTF for the named surrogate"
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
            "inertia": self.forward == "first_principles"
            and hasattr(params, "J_inv"),
            "motor_strength": hasattr(params, "cmd_f_coef")
            or hasattr(params, "rpm2thrust"),
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

    def _drone_reference(self, duration, freq, device, start):
        """Single-drone Crazyflow reference whose initial pose the caller owns."""

        def reset(data, default, mask):
            del default
            speed = 10000.0 if self.forward == "first_principles" else 0.05
            rotor = jnp.full_like(data.states.rotor_vel, speed)
            return data.replace(
                states=leaf_replace(data.states, mask, rotor_vel=rotor)
            )

        reference = DroneEnv(
            num_envs=1,
            freq=freq,
            max_episode_time=duration,
            dynamics=self.forward,
            drone=self.drone,
            device=device,
            reset_randomization=reset,
        )
        reference.sim.data = reference.sim.data.replace(
            states=reference.sim.data.states.replace(
                pos=reference.sim.data.states.pos.at[0, 0].set(
                    jnp.asarray(start)
                )
            )
        )
        reference.sim.build_default_data()
        return reference

    def create_navigation(self, duration, freq, device, start):
        """Reference simulation for the navigation task; the scene owns the pose."""
        reference = self._drone_reference(duration, freq, device, start)
        self.bind(reference.sim)
        return reference

    def create_tracking(self, reference, duration, freq, device, scene):
        if reference == "figure8":
            reference = FigureEightEnv(
                num_envs=1,
                freq=freq,
                dynamics=self.forward,
                drone=self.drone,
                device=device,
                trajectory_time=duration,
                max_episode_time=duration,
            )
        else:
            reference = self._drone_reference(
                duration, freq, device, scene.takeoff
            )
        self.bind(reference.sim)
        return reference

    def bind(self, sim):
        from crazyflow.sim.pipeline import insert_fn_before
        from jax.scipy.spatial.transform import Rotation

        self.sim = sim
        self._validate_randomization(sim.default_data.params)

        def apply_external_wrench(data):
            if "external_force_world_n" not in data.plugins:
                return data
            from drone_playground.environments.randomization import external_wrench

            force, body_torque = external_wrench(data)
            # The source integrator consumes world-frame force and torque.
            torque = Rotation.from_quat(data.states.quat[0, 0]).apply(
                body_torque
            )
            return data.replace(
                states=data.states.replace(
                    force=jnp.broadcast_to(force, data.states.force.shape),
                    torque=jnp.broadcast_to(torque, data.states.torque.shape),
                )
            )

        insert_fn_before(
            sim.step_pipeline, "integration", apply_external_wrench
        )
        self.step_fn = sim.build_step_fn()
        return self

    def advance(self, data, substeps):
        return self.step_fn(data, substeps)
