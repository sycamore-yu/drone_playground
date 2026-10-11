"""Small physical contracts for independently replaceable flight methods."""

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_params
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


class Action:
    """One normalized-to-physical mapping, independent of tasks and networks."""

    def __init__(
        self,
        level="attitude_thrust",
        *,
        drone="cf21B_500",
        low=None,
        high=None,
        frame=None,
        heading="fixed",
        frequency=50.0,
    ):
        """Resolve nominal limits once; physical randomization never changes them."""
        self.level, self.heading = level, heading
        self.mass = float(np.asarray(load_params("first_principles", drone)["mass"]).item())
        self.hover = self.mass * 9.81
        defaults = {
            "acceleration": ([-6.0] * 3, [6.0] * 3),
            "attitude_thrust": (
                [-0.6, -0.6, -np.pi, 0.4 * self.hover],
                [0.6, 0.6, np.pi, 1.6 * self.hover],
            ),
            "body_rate": ([0.4 * self.hover, -6.0, -6.0, -6.0], [1.6 * self.hover, 6.0, 6.0, 6.0]),
        }
        if level not in {*defaults, "force_torque", "rotor_vel"}:
            raise ValueError(f"Unknown action level: {level}")
        if (low is None) != (high is None):
            raise ValueError("Both physical action bounds must be supplied")
        if low is None:
            if level not in defaults:
                raise ValueError(f"{level} needs explicit physical low/high bounds")
            low, high = defaults[level]
        lower, upper = np.asarray(low, np.float32), np.asarray(high, np.float32)
        size = 3 if level == "acceleration" else 4
        if (
            lower.shape != (size,)
            or upper.shape != (size,)
            or not np.isfinite([lower, upper]).all()
        ):
            raise ValueError("Action bounds must be finite vectors matching the control interface")
        if np.any(lower >= upper):
            raise ValueError("Each lower action limit must be smaller than its upper limit")
        expected_frame = "world" if level in {"acceleration", "attitude_thrust"} else "body"
        self.frame = expected_frame if frame is None else frame
        if self.frame != expected_frame or heading not in {"fixed", "target"}:
            raise ValueError("Unsupported action frame or heading convention")
        self.low, self.high, self.size = jnp.asarray(lower), jnp.asarray(upper), size
        self.output_level = "attitude_thrust" if level == "acceleration" else level
        self._acceleration = None
        if level == "acceleration":
            from drone_playground.simulation.controllers import MellingerController

            self._acceleration = MellingerController(drone=drone, frequency=frequency)

    @property
    def contract(self):
        """Return the action meaning saved with network parameters."""
        return {
            "level": self.level,
            "frame": self.frame,
            "heading": self.heading,
            "low": np.asarray(self.low).tolist(),
            "high": np.asarray(self.high).tolist(),
        }

    def decode(self, action):
        """Decode bounded policy output into the declared physical units."""
        if action.shape[-1] != self.size:
            raise ValueError("Policy output dimension does not match its action interface")
        scale = (self.high - self.low) / 2
        return jnp.clip(action, -1.0, 1.0) * scale + (self.high + self.low) / 2

    def setpoint(self, physics, physical, target=None):
        """Convert a physical request to the selected native control input."""
        if self.level == "body_rate":
            # Policy/model input is [thrust, wx, wy, wz]; Crazyflow stages rates first.
            return Setpoint(physical[..., jnp.array([1, 2, 3, 0])], level="body_rate", frame="body")
        if self._acceleration is None:
            return Setpoint(physical, level=self.level, frame=self.frame)
        yaw = jnp.zeros(physical.shape[:-1])
        if self.heading == "target":
            if target is None:
                raise ValueError("Target heading requires an explicit reference position")
            delta = target - physics.states.pos[:, 0]
            yaw = jnp.arctan2(delta[:, 1], delta[:, 0])
        return self._acceleration.acceleration(physics, physical, yaw)


class PolicyMethod:
    """Policy execution with parameters and actor memory owned by the caller."""

    execution = "jax"
    trainable = True

    def initialize(self, env, seed, actor=None):
        """Initialize only the memory required by this actor."""
        if actor is None:
            raise ValueError("Policy execution requires an Actor")
        return actor.initialize_memory(env.num_envs)

    def build_step(self, env, actor=None):
        """Bind the pure policy/environment call before compilation."""
        if actor is None:
            raise ValueError("Policy execution requires an Actor")

        def step(state, memory, parameters):
            raw, memory, _ = actor.apply(parameters, env.observe(state), memory)
            state = env.step(state, jnp.tanh(raw))
            memory = jnp.where(state.done[:, None], 0.0, memory)
            return state, memory

        return step

    def close(self):
        """No external resources are allocated."""


class PlannerController:
    """A planner plus tracker, or a tracker following the task reference."""

    trainable = False

    def __init__(self, controller, planner=None):
        """Compose the two concrete roles without an arbitrary stage pipeline."""
        self.controller, self.planner = controller, planner
        self.execution = "host" if planner is not None else controller.execution

    def initialize(self, env, seed, actor=None):
        """Reset the controller and planner for an independent episode."""
        if self.planner is not None:
            self.planner.reset(seed)
        memory = self.controller.initialize_memory(env.num_envs, seed)
        if self.execution == "host":
            return {"controller": memory, "trajectory": None, "last_plan": -float("inf")}
        return memory

    def build_step(self, env, actor=None):
        """Bind a JAX tracker without putting a host planner inside jit."""
        if self.execution != "jax":
            raise ValueError("Host methods must use the host rollout")

        def step(state, memory, parameters):
            reference = env.task.target(state.time)
            setpoint, memory = self.controller(state.physics, reference, memory)
            return env.step_setpoint(state, setpoint), memory

        return step

    def host_command(self, env, state, memory):
        """Schedule planning and sample the complete reference needed by the tracker."""
        time = int(state.physics.core.steps[0, 0]) / env.sim.freq
        decision = None
        if self.planner is not None:
            if state.observation is None:
                raise ValueError("A perception planner needs a configured sensor")
            if (
                bool(state.observation.available[0])
                and time - memory["last_plan"] + 1e-9 >= self.planner.replan_interval
            ):
                body = state.physics.states
                observation = {
                    "position": np.asarray(body.pos[0, 0]),
                    "quaternion": np.roll(np.asarray(body.quat[0, 0]), 1),
                    "velocity": np.asarray(body.vel[0, 0]),
                    "acceleration": np.asarray(state.acceleration[0]),
                    "goal": np.asarray(env.task.goal),
                    "measurement": state.observation.measurement,
                    "sensor_pose": np.asarray(state.observation.acquisition_pose[0]),
                    "sensor_points": np.asarray(state.observation.measurement.points_body[0]),
                    "sensor_config": env.sensor.config,
                }
                trajectory = self.planner.plan(observation, time, force_replan=True)
                memory["trajectory"], memory["last_plan"] = trajectory, time
                decision = {
                    "time": time,
                    "sensor_time": float(state.observation.measurement.acquisition_time[0]),
                    "status": trajectory.status,
                    "sequence": trajectory.sequence,
                    "upstream_status": trajectory.upstream_status,
                    "valid_until": trajectory.valid_until,
                    **trajectory.timings,
                }
        times = time + np.asarray(self.controller.reference_offsets)
        trajectory = memory["trajectory"]
        if trajectory is not None:
            samples = [trajectory.sample(float(t)) for t in times]
            reference = tuple(
                jnp.asarray(np.stack([s[i] for s in samples]))[None] for i in range(3)
            )
        elif self.planner is None:
            reference = env.task.target(jnp.asarray(times)[None])
        else:
            pos = jnp.broadcast_to(state.physics.states.pos[:, 0, None], (1, len(times), 3))
            reference = (pos, jnp.zeros_like(pos), jnp.zeros_like(pos))
        if len(times) == 1:
            reference = tuple(x[:, 0] for x in reference)
        command, memory["controller"] = self.controller(
            state.physics, reference, memory["controller"]
        )
        return command, memory, trajectory, decision

    def close(self):
        """Release resources owned by the planner and controller."""
        if self.planner is not None:
            self.planner.close()
        self.controller.close()
