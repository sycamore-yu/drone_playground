"""Small physical contracts for independently replaceable flight methods."""

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_params
from flax import struct
from jax import Array
from jax.scipy.spatial.transform import Rotation


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


def sample_planner_horizon(trajectory: Trajectory, times: np.ndarray):
    """Pad only the *future* MPC horizon with a stationary terminal reference.

    EGO/SUPER may return a valid plan shorter than the controller's fixed OCP
    horizon. Keep all upstream samples inside their declared validity interval;
    fill only the prediction tail with an explicit stationary endpoint target.
    A plan that has expired at the current control step still fails normally.
    This is a reference boundary condition, not an obstacle-avoidance fallback
    or permission to reuse a stale plan after its valid_until.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Controller reference times must be a finite vector")
    valid_from, valid_until = float(trajectory.times[0]), float(trajectory.times[-1])
    # The physics clock uses integer ticks, while a ROS plan's first sample
    # may be based on float32 scene/sensor time. For example, the same 1.2 s
    # physics instant can be represented as 1.2000000476837158 in the native
    # trajectory. Attitude / full-state NMPC include t=now as stage zero,
    # unlike sampling MPC which starts at t=now+dt. Align representation only,
    # never make a genuinely future or expired plan valid.
    clock_ulp = float(np.spacing(np.float32(max(abs(times[0]), abs(valid_from)))))
    tolerance = max(1e-9, 2 * clock_ulp)
    if times[0] < valid_from - tolerance or times[0] > valid_until + tolerance:
        raise RuntimeError("Planner trajectory does not cover the current controller step")
    samples = []
    for time in times:
        if time < valid_from and valid_from - time <= tolerance:
            # Clamp only the sub-ULP mismatch to the original first sample.
            values = trajectory.sample(valid_from)
        elif time <= valid_until:
            values = trajectory.sample(float(time))
        elif time - valid_until <= tolerance:
            values = trajectory.sample(valid_until)
        else:
            # Hold the last planned point rather than extrapolating an unknown
            # collision-free path or pretending the original plan stays valid.
            values = (trajectory.positions[-1], np.zeros(3), np.zeros(3))
        if any(item is None for item in values):
            raise ValueError("Planner must provide position, velocity, and acceleration")
        samples.append(values)
    reference = tuple(jnp.asarray(np.stack([row[i] for row in samples]))[None] for i in range(3))
    extension = max(0.0, float(times[-1] - valid_until))
    return reference, extension


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
        thrust_to_weight_bounds=(0.4, 1.6),
        body_rate_limit=6.0,
    ):
        """Resolve nominal limits once; physical randomization never changes them."""
        self.level, self.heading = level, heading
        self.mass = float(np.asarray(load_params("first_principles", drone)["mass"]).item())
        self.hover = self.mass * 9.81
        thrust_low, thrust_high = thrust_to_weight_bounds
        if (
            not np.isfinite([thrust_low, thrust_high, body_rate_limit]).all()
            or not 0 <= thrust_low < thrust_high
            or body_rate_limit <= 0
        ):
            raise ValueError("Physical thrust/rate limits must be finite and ordered")
        if (tuple(thrust_to_weight_bounds) != (0.4, 1.6) or body_rate_limit != 6.0) and (
            level != "body_rate" or low is not None or high is not None
        ):
            raise ValueError("Thrust-to-weight/rate limits configure default body_rate bounds")
        defaults = {
            "acceleration": ([-6.0] * 3, [6.0] * 3),
            "horizontal_acceleration": ([-6.0] * 2, [6.0] * 2),
            "attitude_thrust": (
                [-0.6, -0.6, -np.pi, 0.4 * self.hover],
                [0.6, 0.6, np.pi, 1.6 * self.hover],
            ),
            "body_rate": (
                [thrust_low * self.hover, *([-body_rate_limit] * 3)],
                [thrust_high * self.hover, *([body_rate_limit] * 3)],
            ),
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
        size = 2 if level == "horizontal_acceleration" else 3 if level == "acceleration" else 4
        if (
            lower.shape != (size,)
            or upper.shape != (size,)
            or not np.isfinite([lower, upper]).all()
        ):
            raise ValueError("Action bounds must be finite vectors matching the control interface")
        if np.any(lower >= upper):
            raise ValueError("Each lower action limit must be smaller than its upper limit")
        expected_frame = (
            "world"
            if level in {"acceleration", "horizontal_acceleration", "attitude_thrust"}
            else "body"
        )
        self.frame = expected_frame if frame is None else frame
        if (
            self.frame != expected_frame and not (level == "acceleration" and self.frame == "body")
        ) or heading not in {"fixed", "target"}:
            raise ValueError("Unsupported action frame or heading convention")
        self.low, self.high, self.size = jnp.asarray(lower), jnp.asarray(upper), size
        self.output_level = (
            "attitude_thrust" if level in {"acceleration", "horizontal_acceleration"} else level
        )
        self._acceleration = None
        if level in {"acceleration", "horizontal_acceleration"}:
            from drone_playground.simulation.controllers import MellingerController

            self._acceleration = MellingerController(drone=drone, frequency=frequency)

    @property
    def contract(self):
        """Return the action meaning saved with network parameters."""
        result = {
            "level": self.level,
            "frame": self.frame,
            "heading": self.heading,
            "low": np.asarray(self.low).tolist(),
            "high": np.asarray(self.high).tolist(),
        }
        if self.level == "horizontal_acceleration":
            result["altitude_hold"] = {"kp": 2.0, "kd": 1.5, "z_acceleration_limit": 3.0}
        return result

    def decode(self, action):
        """Decode bounded policy output into the declared physical units."""
        if action.shape[-1] != self.size:
            raise ValueError("Policy output dimension does not match its action interface")
        scale = (self.high - self.low) / 2
        return jnp.clip(action, -1.0, 1.0) * scale + (self.high + self.low) / 2

    def world_acceleration(self, physics, physical):
        """Map commanded *net* acceleration to the world frame, before gravity compensation.

        Body acceleration is expressed in the current body basis; it is not a
        thrust-specific-force command. Rotate once when issuing the setpoint,
        not once per inner-loop physics tick. This preserves the existing native
        Mellinger/PointMass world-acceleration contract and delay semantics.
        """
        if self.level != "acceleration":
            raise ValueError("World acceleration conversion requires acceleration action level")
        if self.frame == "body":
            return Rotation.from_quat(physics.states.quat[:, 0]).apply(physical)
        return physical

    def setpoint(self, physics, physical, target=None):
        """Convert a physical request to the selected native control input."""
        if self.level == "body_rate":
            # Policy/model input is [thrust, wx, wy, wz]; Crazyflow stages rates first.
            return Setpoint(physical[..., jnp.array([1, 2, 3, 0])], level="body_rate", frame="body")
        if self._acceleration is None:
            return Setpoint(physical, level=self.level, frame=self.frame)
        if self.level == "horizontal_acceleration":
            if target is None:
                raise ValueError(
                    "Horizontal acceleration needs a target altitude for stabilization"
                )
            target = jnp.broadcast_to(target, physics.states.pos[:, 0].shape)
            altitude_error = target[:, 2] - physics.states.pos[:, 0, 2]
            vertical_speed = physics.states.vel[:, 0, 2]
            z_acceleration = jnp.clip(2.0 * altitude_error - 1.5 * vertical_speed, -3.0, 3.0)
            physical = jnp.concatenate([physical, z_acceleration[:, None]], axis=-1)
        yaw = jnp.zeros(physical.shape[:-1])
        if self.heading == "target":
            if target is None:
                raise ValueError("Target heading requires an explicit reference position")
            delta = target - physics.states.pos[:, 0]
            yaw = jnp.arctan2(delta[:, 1], delta[:, 0])
        if self.level == "acceleration":
            physical = self.world_acceleration(physics, physical)
        return self._acceleration.acceleration(physics, physical, yaw)


class PolicyMethod:
    """Policy execution with parameters and actor memory owned by the caller."""

    execution = "jax"
    trainable = True

    def initial_log_std(self, env, value):
        """Initialize this method's stochastic continuous-action parameters."""
        return jnp.full((env.action_size,), value, dtype=jnp.float32)

    def sample(self, rng, raw, log_std):
        """Return executable action, sampled policy variables and log probability."""
        from drone_playground.simulation.networks import sample_tanh_gaussian

        return sample_tanh_gaussian(rng, raw, log_std)

    def log_probability(self, sample, raw, log_std, observation):
        """Evaluate the probability of a previous sample under current weights."""
        from drone_playground.simulation.networks import tanh_gaussian_log_prob

        return tanh_gaussian_log_prob(sample, raw, log_std)

    def entropy(self, rng, raw, log_std):
        """Estimate transformed-action entropy using the same distribution."""
        return -jnp.mean(self.sample(rng, raw, log_std)[2])

    def deterministic(self, raw):
        """Decode the mean network output for evaluation and differentiable training."""
        return jnp.tanh(raw)

    def execute(self, env, state, action, step=None):
        """Execute this method's action through the selected JAX transition.

        The caller selects the forward or backward dynamics step. A learnable
        planner can supply a different physical action conversion here.
        """
        return (env.step if step is None else step)(state, action)

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
            from drone_playground.simulation.networks import reset_actor_memory

            raw, memory, _ = actor.apply(parameters, env.observe(state), memory)
            state = self.execute(env, state, self.deterministic(raw))
            memory = reset_actor_memory(memory, state.done)
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
        if hasattr(self.controller, "set_course"):
            self.controller.set_course(env.task)
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
            reference, terminal_hold = sample_planner_horizon(trajectory, times)
            if terminal_hold > 1e-9:
                if decision is None:
                    decision = {"time": time, "status": "TERMINAL_REFERENCE_HOLD"}
                decision["terminal_hold_seconds"] = terminal_hold
                decision["valid_until"] = trajectory.valid_until
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
