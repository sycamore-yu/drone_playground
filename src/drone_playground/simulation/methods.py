"""Small physical contracts for independently replaceable flight methods."""

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude
from flax import struct
from jax import Array


@dataclass(frozen=True)
class Path:
    """Ordered world-frame geometric points, in metres, without a clock."""

    positions: np.ndarray
    frame: str = "world"


@dataclass(frozen=True)
class Trajectory:
    """A time-valid world-frame trajectory with explicit supplied derivatives."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None = None
    accelerations: np.ndarray | None = None
    frame: str = "world"

    def __post_init__(self):
        """Validate trajectory coordinates, samples and supplied derivatives."""
        if self.frame != "world":
            raise ValueError("Trajectory positions must use the world frame")
        if (
            len(self.times) < 2
            or np.any(np.diff(self.times) <= 0)
            or self.positions.shape != (len(self.times), 3)
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.positions).all()
        ):
            raise ValueError("Trajectory needs finite increasing times and Nx3 positions")
        for derivative in (self.velocities, self.accelerations):
            if derivative is not None and (
                derivative.shape != self.positions.shape or not np.isfinite(derivative).all()
            ):
                raise ValueError("Trajectory derivatives must be finite Nx3 arrays")

    def sample(self, time: float):
        """Interpolate only inside the supplied validity interval."""
        if not self.times[0] <= time <= self.times[-1]:
            raise ValueError(f"Trajectory is not valid at time {time}")
        return tuple(
            None
            if x is None
            else np.array([np.interp(time, self.times, x[:, i]) for i in range(3)])
            for x in (self.positions, self.velocities, self.accelerations)
        )


@struct.dataclass
class Setpoint:
    """Physical control with an explicit frame and control layer."""

    value: Array
    level: str = struct.field(pytree_node=False, default="attitude_thrust")
    frame: str = struct.field(pytree_node=False, default="world")


class Planner(Protocol):
    """Host planner: observations contain only the configured information permissions."""

    def reset(self, seed: int) -> None:
        """Reset episode-local planning state for the supplied seed."""
        ...

    def plan(self, observation: dict, time: float) -> Trajectory:
        """Compute a trajectory from the permitted observation at the stated time."""
        ...


class Controller(Protocol):
    """Pure control conversion with explicit controller memory."""

    def __call__(self, physics, reference, memory) -> tuple[Setpoint, Array]:
        """Convert the reference into a physical setpoint and updated memory."""
        ...


class Policy(Protocol):
    """Pure policy; its parameters, recurrent state and RNG belong to its caller."""

    def __call__(self, parameters, observation, memory, rng):
        """Compute an action from parameters, observation, memory and randomness."""
        ...


class MellingerController:
    """Use Crazyflow's position controller and its identified low-level control stack."""

    def __init__(self, sim, frequency: float = 50.0):
        """Bind the official controller conversion to the drone parameters."""
        self.frequency = frequency
        self.convert = parametrize(state2attitude, sim.drone, xp=jnp)
        self.mass = sim.data.params.mass
        core = self.convert.keywords
        # The firmware PWM gain is calibrated away for a command explicitly measured in newtons.
        self.mass_thrust = core["pwm_max"] / (4 * core["thrust_max"])

    def __call__(self, physics, reference, memory):
        """Produce a physical setpoint from position, velocity and acceleration."""
        pos, vel, acc = reference
        cmd = jnp.zeros((*pos.shape[:-1], 16)).at[..., :3].set(pos)
        cmd = cmd.at[..., 3:6].set(vel).at[..., 6:9].set(acc).at[..., 12].set(1.0)
        output, memory = self.convert(
            physics.states.pos[:, 0],
            physics.states.quat[:, 0],
            physics.states.vel[:, 0],
            cmd,
            memory,
            ctrl_freq=self.frequency,
            mass=self.mass,
            mass_thrust=self.mass_thrust,
        )
        return Setpoint(output), memory

    def acceleration(self, physics, acceleration, yaw):
        """Map net world acceleration to attitude/newtons through the upstream converter."""
        p, v = physics.states.pos[:, 0], physics.states.vel[:, 0]
        cmd = jnp.zeros((*p.shape[:-1], 16)).at[..., :3].set(p)
        cmd = cmd.at[..., 3:6].set(v).at[..., 6:9].set(acceleration)
        cmd = cmd.at[..., 11].set(jnp.sin(yaw / 2)).at[..., 12].set(jnp.cos(yaw / 2))
        output, _ = self.convert(
            p,
            physics.states.quat[:, 0],
            v,
            cmd,
            jnp.zeros_like(p),
            ctrl_freq=self.frequency,
            mass=self.mass,
            mass_thrust=self.mass_thrust,
        )
        return Setpoint(output)
