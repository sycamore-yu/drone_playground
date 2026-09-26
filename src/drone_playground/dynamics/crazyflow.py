"""Model selection and stepping for the four pinned Crazyflow dynamics."""

from dataclasses import dataclass

import jax.numpy as jnp
from crazyflow.envs import FigureEightEnv
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.utils import leaf_replace


@dataclass
class CrazyflowModel:
    forward: str = "so_rpy"
    drone: str = "cf2x_L250"
    backward: str = "direct"

    def __post_init__(self):
        from crazyflow.dynamics.core import Dynamics

        Dynamics(self.forward)
        if self.backward != "direct":
            raise ValueError(
                "Crazyflow currently supports its direct derivative; select LOTF for the named surrogate"
            )

    def create_tracking(self, task, duration, freq, device, scene):
        if task == "figure8":
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

            def reset(data, default, mask):
                del default
                speed = 10000.0 if self.forward == "first_principles" else 0.05
                rotor = jnp.full_like(data.states.rotor_vel, speed)
                return data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor))

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
                    pos=reference.sim.data.states.pos.at[0, 0].set(jnp.asarray(scene.takeoff))
                )
            )
            reference.sim.build_default_data()
        self.bind(reference.sim)
        return reference

    def bind(self, sim):
        self.sim = sim
        self.step_fn = sim.build_step_fn()
        return self

    def advance(self, data, substeps):
        return self.step_fn(data, substeps)
