"""Pure batched episodes around the official Crazyflow control and step functions."""

import jax
import jax.numpy as jnp
from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim import functional as cf
from crazyflow.sim.data import SimData
from crazyflow.utils import pytree_replace, world_mask
from flax import struct
from jax import Array
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.methods import MellingerController, Setpoint
from drone_playground.simulation.observation import ObservationState, SensorObservation
from drone_playground.simulation.scene import Scene
from drone_playground.simulation.sensors import SensorConfig
from drone_playground.simulation.tasks import Event, Task, TaskState


@struct.dataclass
class EnvState:
    """Physics and per-episode task state; no trainer or hidden mutable counters."""

    physics: SimData
    task: TaskState
    previous_action: Array
    observation: ObservationState | None = None

    @property
    def time(self):
        """Return elapsed physical simulation time in seconds."""
        return self.physics.core.steps[:, 0] / self.physics.core.freq

    @property
    def terminated(self):
        """Identify episodes that reached a terminal task outcome."""
        return (self.task.event != Event.RUNNING) & (self.task.event != Event.TIMEOUT)

    @property
    def truncated(self):
        """Identify episodes that reached the task deadline."""
        return self.task.event == Event.TIMEOUT

    @property
    def done(self):
        """Identify episodes requiring a reset before further execution."""
        return self.terminated | self.truncated


class Environment:
    """Single-drone, multi-world execution shared by training and frozen methods."""

    def __init__(
        self,
        task: str = "tracking",
        scene: str = "empty",
        num_envs: int = 1,
        device: str = "cpu",
        dynamics: str = "first_principles",
        drone: str = "cf21B_500",
        physics_hz: int = 500,
        method_hz: int = 50,
        reference: str = "figure_eight",
        duration: float | None = None,
        reference_seed: int = 0,
        sensor: str = "state",
        sensor_config: dict | None = None,
        point_count: int = 1024,
        navigation_goal_observation: bool = False,
    ):
        """Construct the configured task, scene, sensor and Crazyflow physics."""
        if num_envs <= 0 or method_hz <= 0 or physics_hz % method_hz:
            raise ValueError(
                "Positive batch and method frequency dividing physics frequency required"
            )
        if task == "navigation" and scene == "empty":
            raise ValueError("Navigation requires an explicitly selected scene")
        if task == "racing" and scene != "racing":
            raise ValueError("Racing requires the LSY Level0 scene")
        self.scene = Scene(scene)
        self.task = Task(task, self.scene, reference, duration, reference_seed)
        if not isinstance(navigation_goal_observation, bool):
            raise ValueError("navigation_goal_observation must be boolean")
        if navigation_goal_observation and task != "navigation":
            raise ValueError("Goal observation requires Navigation")
        self.navigation_goal_observation = navigation_goal_observation
        if task == "navigation" and float(self.scene.clearance(self.task.start, 0.0)) < 0.22:
            raise ValueError("Navigation nominal reset position lacks 0.15 m body clearance")
        self.num_envs, self.method_hz = num_envs, method_hz
        self.substeps = physics_hz // method_hz
        self.dt = 1 / method_hz
        self.sim = Sim(
            n_worlds=num_envs,
            drone=Drone(drone),
            dynamics=Dynamics(dynamics),
            control=Control.attitude,
            freq=physics_hz,
            device=device,
            attitude_freq=physics_hz,
            force_torque_freq=physics_hz,
        )
        self.physics_step = self.sim.build_step_fn()
        self.physics_reset = self.sim.build_reset_fn()
        self.controller = MellingerController(self.sim, method_hz)
        self.action_size = 3 if task == "navigation" else 4
        self.sensor = None
        if sensor == "depth":
            config = SensorConfig.d435i("training64x48", **(sensor_config or {}))
            self.sensor = SensorObservation(self.scene, config, physics_hz)
        elif sensor in {"lidar", "uniform_lidar_paper"}:
            factory = SensorConfig.mid360 if sensor == "lidar" else SensorConfig.uniform_lidar_paper
            self.sensor = SensorObservation(
                self.scene, factory(**(sensor_config or {})), physics_hz, point_count
            )
        elif sensor != "state":
            raise ValueError(f"Unknown observation profile: {sensor}")
        self.reset = jax.jit(self._reset, static_argnames=("randomize_position",))
        self.step = jax.jit(self._step)
        self.step_setpoint = jax.jit(self._step_setpoint)
        self.observe_state = jax.jit(self._observe_state)
        self.observe = jax.jit(self._observe)

    def _reset(
        self, key, state: EnvState | None = None, mask=None, *, randomize_position=False
    ) -> EnvState:
        """Reset selected worlds; wide position sampling is opt-in Navigation training."""
        if randomize_position and self.task.name != "navigation":
            raise ValueError("randomize_position requires Navigation")
        batch = self.num_envs
        pos_key, vel_key, yaw_key = jax.random.split(key, 3)
        width = (
            jnp.array([0.25, 0.25, 0.10]) if self.task.name == "navigation" else jnp.full(3, 0.05)
        )

        def sample_positions(sample_key):
            if randomize_position:
                return jax.random.uniform(
                    sample_key,
                    (batch, 3),
                    minval=jnp.array([2.0, -18.0, 1.0]),
                    maxval=jnp.array([94.0, 18.0, 5.5]),
                )
            return self.task.start + jax.random.uniform(
                sample_key, (batch, 3), minval=-width, maxval=width
            )

        pos = sample_positions(pos_key)
        if self.task.name == "navigation":

            def resample(carry):
                positions, random_key = carry
                random_key, sample_key = jax.random.split(random_key)
                candidate = sample_positions(sample_key)
                invalid = self.scene.clearance(positions, 0.0) - self.task.radius < 0.15
                return jnp.where(invalid[:, None], candidate, positions), random_key

            pos, _ = jax.lax.while_loop(
                lambda carry: jnp.any(
                    self.scene.clearance(carry[0], 0.0) - self.task.radius < 0.15
                ),
                resample,
                (pos, pos_key),
            )
        vel = jax.random.uniform(vel_key, (batch, 3), minval=-0.1, maxval=0.1)
        yaw = jax.random.uniform(yaw_key, (batch,), minval=-jnp.pi / 36, maxval=jnp.pi / 36)
        quat = jnp.stack(
            [jnp.zeros_like(yaw), jnp.zeros_like(yaw), jnp.sin(yaw / 2), jnp.cos(yaw / 2)], -1
        )
        initial = self.physics_reset(self.sim.data, self.sim.default_data)
        states = initial.states.replace(pos=pos[:, None], vel=vel[:, None], quat=quat[:, None])
        if self.sim.dynamics == Dynamics.first_principles:
            rpm = motor_force2rotor_vel(
                jnp.full((batch, 1, 4), 1.0) * initial.params.mass * 9.81 / 4,
                initial.params.rpm2thrust,
            )
            states = states.replace(rotor_vel=rpm)
        initial = initial.replace(states=states)
        observation = None if self.sensor is None else self.sensor.reset(initial)
        fresh = EnvState(
            initial, self.task.initial(batch), jnp.zeros((batch, self.action_size)), observation
        )
        if state is None:
            return fresh
        if mask is None:
            mask = state.done
        physics = self.physics_reset(state.physics, initial, mask)
        task = jax.tree.map(lambda old, new: jnp.where(mask, new, old), state.task, fresh.task)
        action = jnp.where(mask[:, None], fresh.previous_action, state.previous_action)
        if observation is not None:
            observation = jax.tree.map(
                lambda old, new: jnp.where(
                    mask.reshape(mask.shape + (1,) * (new.ndim - 1)), new, old
                ),
                state.observation,
                fresh.observation,
            )
        return EnvState(physics, task, action, observation)

    def action_setpoint(self, state: EnvState, action: Array) -> Setpoint:
        """Decode the shared bounded action contract; all algorithms use this conversion."""
        if self.task.name == "navigation":
            delta = self.task.goal - state.physics.states.pos[:, 0]
            yaw = jnp.arctan2(delta[:, 1], delta[:, 0])
            return self.controller.acceleration(state.physics, action * 6.0, yaw)
        hover = state.physics.params.mass * 9.81
        value = action * jnp.array([0.6, 0.6, jnp.pi, 1.0])
        value = value.at[:, 3].set(hover * (1 + 0.6 * action[:, 3]))
        return Setpoint(value)

    def _step(self, state: EnvState, action: Array) -> EnvState:
        action = jnp.clip(action, -1.0, 1.0)
        result = self._step_setpoint(state, self.action_setpoint(state, action))
        return result.replace(
            previous_action=jnp.where(state.done[:, None], state.previous_action, action)
        )

    def _step_setpoint(self, state: EnvState, setpoint: Setpoint) -> EnvState:
        if setpoint.level != "attitude_thrust" or setpoint.frame != "world":
            raise ValueError("Crazyflow attitude control requires world RPY radians and thrust N")
        staged = cf.attitude_control(state.physics, setpoint.value[:, None, :])
        staged = pytree_replace(staged, state.physics, world_mask(staged), state.done)
        state = state.replace(physics=staged)

        def advance(carry, _):
            physics = self.physics_step(carry.physics, 1)
            time = physics.core.steps[:, 0] / self.sim.freq
            clearance = self.scene.clearance(physics.states.pos[:, 0], time)
            task_state = self.task.update(
                carry.task, carry.physics.states, physics.states, time, clearance
            )
            physics = pytree_replace(physics, carry.physics, world_mask(physics), carry.done)
            observation = carry.observation
            if self.sensor is not None:
                observation = self.sensor.update(observation, physics, ~carry.done)
            return carry.replace(physics=physics, task=task_state, observation=observation), None

        result, _ = jax.lax.scan(advance, state, None, length=self.substeps)
        return result

    def _observe_state(self, state: EnvState) -> dict[str, Array]:
        physics = state.physics.states
        p, v, q, w = physics.pos[:, 0], physics.vel[:, 0], physics.quat[:, 0], physics.ang_vel[:, 0]
        rot = Rotation.from_quat(q)
        if self.task.name == "navigation":
            delta = self.task.goal - p
            goal_velocity = delta / jnp.maximum(
                jnp.linalg.norm(delta, axis=-1, keepdims=True), 1e-6
            )
            goal_velocity = (
                goal_velocity * jnp.minimum(3.0, jnp.linalg.norm(delta, axis=-1))[:, None]
            )
            body_v, body_target = rot.apply(v, inverse=True), rot.apply(goal_velocity, inverse=True)
            up = rot.apply(jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), p.shape), inverse=True)
            features = jnp.concatenate([body_v, body_target, up, jnp.full((len(p), 1), 0.07)], -1)
            if self.navigation_goal_observation:
                height_error = (self.task.goal[2] - p[:, 2]) / (
                    self.task.high[2] - self.task.low[2]
                )
                remaining = jnp.linalg.norm(delta, axis=-1) / jnp.linalg.norm(
                    self.task.goal - self.task.start
                )
                features = jnp.concatenate(
                    [features, height_error[:, None], remaining[:, None]], -1
                )
        else:
            target, tv, ta = self.task.target(state.time)
            future = jnp.stack(
                [self.task.target(state.time + offset)[0] - p for offset in (0.2, 0.5, 1.0)], -2
            ).reshape(len(p), -1)
            features = jnp.concatenate([target - p, tv - v, ta, q, w, v, future], -1)
            if self.task.name == "racing":
                gate = self.task.gate_order[jnp.minimum(state.task.gates, 4)]
                features = jnp.concatenate(
                    [features, self.task.gate_positions[gate] - p, (state.task.gates / 5)[:, None]],
                    -1,
                )
        return {"state": features}

    def _observe(self, state: EnvState) -> dict[str, Array]:
        observation = self._observe_state(state)
        if self.sensor is not None:
            observation.update(self.sensor.encode(state.observation))
        return observation

    def observe_privileged(self, state: EnvState) -> Array:
        """Return true Navigation state and instantaneous geometry for training only.

        Twenty-six fixed world-frame rays cover the nonzero directions in
        {-1,0,1} cubed. These ideal ranges have no sensor delay, noise or field
        of view restriction. Physical scales are fixed across all scenes.
        """
        if self.task.name != "navigation":
            raise ValueError("Privileged observation requires Navigation")
        physics = state.physics.states
        position = physics.pos[:, 0]
        directions = jnp.array(
            [
                (x, y, z)
                for x in (-1, 0, 1)
                for y in (-1, 0, 1)
                for z in (-1, 0, 1)
                if (x, y, z) != (0, 0, 0)
            ],
            dtype=position.dtype,
        )
        ranges = self.scene.raycast(position, directions, state.time, max_range=40.0)
        clearance = self.scene.clearance(position, state.time) - self.task.radius
        return jnp.concatenate(
            [
                self._observe_state(state)["state"],
                position / jnp.array([100.0, 20.0, 6.0]),
                (self.task.goal - position) / jnp.array([100.0, 40.0, 6.0]),
                physics.quat[:, 0],
                physics.vel[:, 0] / 3.0,
                physics.ang_vel[:, 0] / 10.0,
                state.previous_action,
                (state.time / self.task.duration)[:, None],
                (jnp.clip(clearance, -40.0, 40.0) / 40.0)[:, None],
                ranges / 40.0,
            ],
            axis=-1,
        )

    def cost(self, before: EnvState, after: EnvState, action: Array) -> Array:
        """Dense task costs and first-event penalties; same terms for all algorithms."""
        p, v = after.physics.states.pos[:, 0], after.physics.states.vel[:, 0]
        target, tv, _ = self.task.target(after.time)
        if self.task.name == "navigation":
            direction = self.task.goal - p
            distance = jnp.linalg.norm(direction, axis=-1, keepdims=True)
            desired = direction / jnp.maximum(distance, 1e-6) * jnp.minimum(distance, 3.0)
            error = jnp.sum((v - desired) ** 2, -1)
            clearance = self.scene.clearance(p, after.time) - self.task.radius
            error = error + 5 * jax.nn.softplus((0.5 - clearance) * 8) / 8
        else:
            error = 5 * jnp.sum((p - target) ** 2, -1) + 0.2 * jnp.sum((v - tv) ** 2, -1)
        smooth = 0.01 * jnp.sum(action**2, -1)
        smooth = smooth + 0.01 * jnp.sum((action - before.previous_action) ** 2, -1)
        failure = after.done & (after.task.event != Event.SUCCESS) & ~before.done
        success = (after.task.event == Event.SUCCESS) & ~before.done
        return jnp.where(before.done, 0.0, (error + smooth) * self.dt + 20 * failure - 5 * success)
