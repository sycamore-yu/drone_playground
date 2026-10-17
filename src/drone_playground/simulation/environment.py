"""Batched episodes with explicit control inputs, delivery state and native physics."""

import math
from functools import partial

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

from drone_playground.simulation.delay import CommandState, delay_range
from drone_playground.simulation.methods import Action, Setpoint
from drone_playground.simulation.observation import ObservationState, SensorObservation
from drone_playground.simulation.procedural_scene import ProceduralScene, SceneState
from drone_playground.simulation.randomization import install_randomization
from drone_playground.simulation.scene import Scene, SceneBatch
from drone_playground.simulation.sensors import SensorConfig
from drone_playground.simulation.task_rewards import motion_navigation_reward, shaped_task_cost
from drone_playground.simulation.tasks import Event, Task, TaskState, first_event


@struct.dataclass
class EnvState:
    """Physics and per-episode task state; no trainer or hidden mutable counters."""

    physics: SimData
    task: TaskState
    previous_action: Array
    observation: ObservationState | None = None
    commands: CommandState | None = None
    acceleration: Array | None = None
    scene_state: SceneState | None = None
    finite_horizon: bool = struct.field(pytree_node=False, default=False)

    @property
    def time(self):
        """Return elapsed physical simulation time in seconds."""
        return self.physics.core.steps[:, 0] / self.physics.core.freq

    @property
    def terminated(self):
        """Identify episodes that reached a terminal task outcome."""
        return (self.task.event != Event.RUNNING) & (
            (self.task.event != Event.TIMEOUT) | self.finite_horizon
        )

    @property
    def truncated(self):
        """Identify an external time limit that still permits value bootstrapping."""
        return (self.task.event == Event.TIMEOUT) & (not self.finite_horizon)

    @property
    def done(self):
        """Identify episodes requiring a reset before further execution."""
        return self.terminated | self.truncated


class BaseEnv:
    """Shared Crazyflow physics, sensors, clocks and action execution.

    Concrete task environments supply observations, rewards and success rules.
    """

    task_name: str = ""
    allows_goal_observation: bool = False

    def __init__(
        self,
        task: str | None = None,
        scene: str | list[str] | tuple[str, ...] = "empty",
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
        navigation_goal_observation: bool = False,
        action: Action | dict | None = None,
        control_level: str | None = None,
        action_delay_s=0.0,
        randomization: dict | None = None,
        disturbance: dict | None = None,
        procedural: dict | None = None,
        observation_config: dict | None = None,
        task_config: dict | None = None,
    ):
        """Construct one physical environment with the selected concrete task."""
        if task is None:
            task = self.task_name
        if task != self.task_name:
            raise ValueError(f"{type(self).__name__} requires task={self.task_name}")
        if num_envs <= 0 or method_hz <= 0 or physics_hz % method_hz:
            raise ValueError(
                "Positive batch and method frequency dividing physics frequency required"
            )
        if isinstance(scene, (list, tuple)):
            if task != "navigation" or procedural is not None:
                raise ValueError("Parallel scene batches require fixed Navigation geometry")
            self.scene = SceneBatch(scene, num_envs)
        else:
            self.scene = (
                ProceduralScene(procedural) if scene == "train/procedural" else Scene(scene)
            )
        if isinstance(self.scene, SceneBatch) and (task_config or {}).get("reward_profile") in {
            "fan2025",
            "fan2025_arrival",
            "visfly2025",
        }:
            raise ValueError("Paper per-object reward geometry does not support scene batches")
        self.task = Task(
            task, self.scene, reference, duration, reference_seed, **(task_config or {})
        )
        self._validate_task_setup(scene)
        if not isinstance(navigation_goal_observation, bool):
            raise ValueError("navigation_goal_observation must be boolean")
        if navigation_goal_observation and not self.allows_goal_observation:
            raise ValueError("Goal observation requires Navigation")
        self.navigation_goal_observation = navigation_goal_observation
        self.observation_config = dict(observation_config or {})
        relative_goal = self.observation_config.get("navigation_relative_goal", False)
        if type(relative_goal) is not bool or (relative_goal and self.task.name != "navigation"):
            raise ValueError("navigation_relative_goal requires boolean Navigation configuration")
        world_rotation = self.observation_config.get("navigation_world_rotation", False)
        if type(world_rotation) is not bool:
            raise ValueError("navigation_world_rotation must be boolean")
        if world_rotation and self.task.name != "navigation":
            raise ValueError("navigation_world_rotation requires Navigation")
        self.num_envs, self.method_hz = num_envs, method_hz
        self.substeps = physics_hz // method_hz
        self.dt = 1 / method_hz
        self.action = (
            action
            if isinstance(action, Action)
            else Action(drone=drone, frequency=method_hz, **(action or {}))
        )
        if (
            self.task.reward_profile == "legacy_motion_navigation_v2"
            and self.action.level != "velocity_yaw"
        ):
            raise ValueError("Motion-aware-v2 task reward requires velocity_yaw action contract")
        self.control_level = control_level or self.action.output_level
        self.delay_bounds = delay_range(action_delay_s)
        self.ideal_samples = math.ceil(self.delay_bounds[1] * physics_hz) + self.substeps
        native_control = {
            "attitude_thrust": Control.attitude,
            "body_rate": Control.body_rate,
            "force_torque": Control.force_torque,
            "rotor_vel": Control.rotor_vel,
            "ideal": Control.attitude,
        }
        if self.control_level not in native_control:
            raise ValueError(f"Unsupported execution control: {self.control_level}")
        if self.control_level == "ideal" and (randomization or disturbance):
            raise ValueError("Ideal tracking has no mass or external-force response")
        self.sim = Sim(
            n_worlds=num_envs,
            drone=Drone(drone),
            dynamics=Dynamics(dynamics),
            control=native_control[self.control_level],
            freq=physics_hz,
            device=device,
            attitude_freq=physics_hz,
            body_rate_freq=physics_hz,
            force_torque_freq=physics_hz,
        )
        install_randomization(self.sim, randomization, disturbance)
        self.physics_step = self.sim.build_step_fn()
        self.physics_reset = self.sim.build_reset_fn()
        self.action_size = self.action.size
        self._stage = {
            "attitude_thrust": cf.attitude_control,
            "body_rate": cf.body_rate_control,
            "force_torque": cf.force_torque_control,
            "rotor_vel": cf.rotor_vel_control,
            "ideal": None,
        }[self.control_level]
        self.sensor = None
        if self.observation_config.get("visfly_quaternion_order", "xyzw") not in {"xyzw", "wxyz"}:
            raise ValueError("VisFly quaternion order must be xyzw or wxyz")
        position_scale = self.observation_config.get("visfly_position_scale", 1.0)
        if (
            isinstance(position_scale, bool)
            or not isinstance(position_scale, (float, int))
            or not math.isfinite(position_scale)
            or position_scale <= 0
        ):
            raise ValueError("VisFly position scale must be finite and positive")
        sensor_config = dict(sensor_config or {})
        latency_bounds = delay_range(sensor_config.get("latency", 0.0))
        if "latency" in sensor_config:
            sensor_config["latency"] = latency_bounds[1]
        if sensor == "depth":
            config = SensorConfig.d435i(**sensor_config)
            self.sensor = SensorObservation(
                self.scene,
                config,
                physics_hz,
                latency_bounds=latency_bounds,
                observation_config=self.observation_config,
            )
        elif sensor in {"lidar", "uniform_lidar_paper"}:
            factory = SensorConfig.mid360 if sensor == "lidar" else SensorConfig.uniform_lidar_paper
            self.sensor = SensorObservation(
                self.scene,
                factory(**sensor_config),
                physics_hz,
                latency_bounds=latency_bounds,
                observation_config=self.observation_config,
            )
        elif sensor != "state":
            raise ValueError(f"Unknown observation profile: {sensor}")
        # A transport-delayed or slower-than-policy sensor can produce a stale
        # *current-frame-compensated* observation. Give the Actor the normalized
        # sample age so identical compensated geometry is not treated as equally
        # certain at t=0 and t=0.09. For synchronous zero-delay OA keep the old
        # state width, preserving the accepted model's input identity.
        self.sensor_age_in_state = bool(
            self.sensor is not None
            and self.sensor.motion_compensate
            and (
                self.sensor.config.frequency_hz < self.method_hz
                or self.sensor.latency_bounds[1] > 0
            )
        )
        self.reset = jax.jit(self._reset, static_argnames=("randomize_position",))
        self.step = jax.jit(self._step)
        self.step_setpoint = jax.jit(self._step_setpoint)
        self.observe_state = jax.jit(self._observe_state)
        self.observe = jax.jit(self._observe)

    def _validate_task_setup(self, scene_name):
        """Concrete environments may validate their scene and reference."""

    def _sample_reset_positions(self, key, count, *, randomize_position):
        """Tracking and Racing use the original local initial-position distribution."""
        if randomize_position:
            raise ValueError("randomize_position requires Navigation")
        return self.task.start + jax.random.uniform(key, (count, 3), minval=-0.05, maxval=0.05)

    def _reset(
        self,
        key,
        state: EnvState | None = None,
        mask=None,
        *,
        randomize_position=False,
        sampling_progress=None,
    ) -> EnvState:
        """Reset selected worlds; wide position sampling is opt-in Navigation training."""
        batch = self.num_envs
        pos_key, vel_key, yaw_key = jax.random.split(key, 3)
        physics_key, delay_key = jax.random.fold_in(key, 100), jax.random.fold_in(key, 101)
        scene_state = (
            self.scene.sample(
                jax.random.fold_in(key, 99),
                batch,
                None if state is None else state.scene_state,
                sampling_progress=sampling_progress,
            )
            if isinstance(self.scene, ProceduralScene)
            else None
        )
        pos = (
            scene_state.start
            if scene_state is not None
            else self._sample_reset_positions(pos_key, batch, randomize_position=randomize_position)
        )
        vel = jax.random.uniform(vel_key, (batch, 3), minval=-0.1, maxval=0.1)
        yaw = jax.random.uniform(yaw_key, (batch,), minval=-jnp.pi / 36, maxval=jnp.pi / 36)
        quat = jnp.stack(
            [jnp.zeros_like(yaw), jnp.zeros_like(yaw), jnp.sin(yaw / 2), jnp.cos(yaw / 2)], -1
        )
        default = self.sim.default_data.replace(
            core=self.sim.default_data.core.replace(rng_key=physics_key)
        )
        initial = self.physics_reset(self.sim.data, default)
        states = initial.states.replace(pos=pos[:, None], vel=vel[:, None], quat=quat[:, None])
        if self.sim.dynamics == Dynamics.first_principles:
            rpm = motor_force2rotor_vel(
                jnp.full((batch, 1, 4), 1.0) * initial.params.mass * 9.81 / 4,
                initial.params.rpm2thrust,
            )
            states = states.replace(rotor_vel=rpm)
        initial = initial.replace(states=states)
        observation = (
            None
            if self.sensor is None
            else self.sensor.reset(
                initial,
                scene_state=scene_state,
                mask=(None if state is None else (state.done if mask is None else mask)),
            )
        )
        inputs = self.action.decode(jnp.zeros((batch, self.action_size)))
        initial_target = (
            scene_state.goal if scene_state is not None else self.task.target(jnp.zeros(batch))[0]
        )
        neutral = self.action.setpoint(initial, inputs, initial_target).value
        if self.action.level == "acceleration":
            # PointMassLag's differentiable physical input is always world-frame,
            # including the command queued on the first physics tick.
            inputs = self.action.world_acceleration(initial, inputs)
        if self.control_level == "ideal":
            neutral = jnp.concatenate([pos, vel, jnp.zeros_like(pos), yaw[:, None]], -1)
            neutral = jnp.broadcast_to(neutral[:, None], (batch, self.ideal_samples, 10)).reshape(
                batch, -1
            )
        commands = CommandState.create(
            delay_key, neutral, inputs, self.delay_bounds, self.sim.freq, self.method_hz
        )
        fresh = EnvState(
            initial,
            self.task.initial(batch),
            jnp.zeros((batch, self.action_size)),
            observation,
            commands,
            jnp.zeros_like(pos),
            scene_state,
            finite_horizon=self.task.finite_horizon,
        )
        # Persisted arrays have explicit dtypes; avoid a different JIT specialization on restore.
        fresh = jax.tree.map(
            lambda value: jnp.asarray(value, dtype=jnp.asarray(value).dtype), fresh
        )
        if state is None:
            return fresh
        if mask is None:
            mask = state.done
        physics = pytree_replace(state.physics, initial, world_mask(initial), mask)
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
        commands = jax.tree.map(
            lambda old, new: jnp.where(
                mask.reshape((len(mask),) + (1,) * (new.ndim - 1)), new, old
            ),
            state.commands,
            fresh.commands,
        )
        acceleration = jnp.where(mask[:, None], fresh.acceleration, state.acceleration)
        if scene_state is not None:
            scene_state = jax.tree.map(
                lambda old, new: jnp.where(
                    mask.reshape(mask.shape + (1,) * (new.ndim - 1)), new, old
                ),
                state.scene_state,
                scene_state,
            )
        return EnvState(
            physics,
            task,
            action,
            observation,
            commands,
            acceleration,
            scene_state,
            finite_horizon=self.task.finite_horizon,
        )

    def goal_for(self, state: EnvState) -> Array:
        """Return static task goal or procedural per-world target."""
        return self.task.goal if state.scene_state is None else state.scene_state.goal

    def scene_clearance(self, position, time, scene_state=None):
        """Use precisely the same geometry as the trained sensor."""
        if scene_state is not None:
            return self.scene.clearance(position, time, scene_state=scene_state)
        return self.scene.clearance(position, time)

    def action_setpoint(self, state: EnvState, action: Array) -> Setpoint:
        """Decode the shared bounded action contract; all algorithms use this conversion."""
        return self.action.setpoint(
            state.physics,
            self.action.decode(action),
            (
                self.goal_for(state)
                if self.task.name == "navigation"
                else self.task.target(state.time)[0]
            ),
        )

    def _step(self, state: EnvState, action: Array, *, physics_step=None) -> EnvState:
        action = jnp.clip(action, -1.0, 1.0)
        if self.sensor is not None and self.sensor.latency_bounds[1] > 0:
            action = jnp.where(state.observation.available[:, None], action, 0.0)
        physical = self.action.decode(action)
        target = (
            self.goal_for(state)
            if self.task.name == "navigation"
            else self.task.target(state.time)[0]
        )
        setpoint = self.action.setpoint(state.physics, physical, target)
        # The Actor's declared frame may be body, but the delayed PointMassLag
        # surrogate consumes world-frame net acceleration. Rotate at issuance,
        # so subsequent ticks do not silently reinterpret an old queued command.
        model_input = (
            self.action.world_acceleration(state.physics, physical)
            if self.action.level == "acceleration"
            else physical
        )
        result = self._step_setpoint(state, setpoint, model_input, physics_step=physics_step)
        return result.replace(
            previous_action=jnp.where(state.done[:, None], state.previous_action, action)
        )

    def build_step(self, physics_step):
        """Bind one differentiable physics transfer without changing the frozen environment."""
        return jax.jit(partial(self._step, physics_step=physics_step))

    def native_step(self, physics, acceleration, command, physical_input):
        """Execute the official pipeline for one physical tick."""
        staged = self._stage(physics, command[:, None, :])
        result = self.physics_step(staged, 1)
        actual_acceleration = (result.states.vel[:, 0] - physics.states.vel[:, 0]) * self.sim.freq
        return result, actual_acceleration

    def _step_setpoint(
        self, state: EnvState, setpoint: Setpoint, inputs=None, *, physics_step=None
    ) -> EnvState:
        if setpoint.level != self.control_level:
            raise ValueError("Controller output and configured physics input do not match")
        frame = "world" if self.control_level in {"attitude_thrust", "ideal"} else "body"
        if setpoint.frame != frame:
            raise ValueError("Control coordinate frame does not match the native interface")
        if inputs is None:
            inputs = jnp.zeros((self.num_envs, self.action_size))
        commands = state.commands.push(
            setpoint.value, inputs, state.physics.core.steps[:, 0], ~state.done
        )
        state = state.replace(commands=commands)
        advance_physics = self.native_step if physics_step is None else physics_step

        def advance(carry, _):
            tick = carry.physics.core.steps[:, 0]
            command, physical, issued = carry.commands.at(tick)
            if self.control_level == "ideal":
                index = jnp.clip(tick - issued, 0, self.ideal_samples - 1)
                sample = command.reshape(self.num_envs, self.ideal_samples, 10)[
                    jnp.arange(self.num_envs), index
                ]
                pos, vel, acc = sample[:, :3], sample[:, 3:6], sample[:, 6:9]
                up = acc + jnp.array([0.0, 0.0, 9.80])
                up = up / jnp.maximum(jnp.linalg.norm(up, axis=-1, keepdims=True), 1e-6)
                heading = jnp.stack(
                    [jnp.cos(sample[:, 9]), jnp.sin(sample[:, 9]), jnp.zeros(self.num_envs)], -1
                )
                side = jnp.cross(up, heading)
                side = side / jnp.maximum(jnp.linalg.norm(side, axis=-1, keepdims=True), 1e-6)
                quat = Rotation.from_matrix(
                    jnp.stack([jnp.cross(side, up), side, up], -1)
                ).as_quat()
                physics = carry.physics.replace(
                    states=carry.physics.states.replace(
                        pos=pos[:, None], vel=vel[:, None], quat=quat[:, None]
                    ),
                    core=carry.physics.core.replace(
                        steps=carry.physics.core.steps + 1, mjx_synced=jnp.array(False)
                    ),
                )
            else:
                physics, acc = advance_physics(carry.physics, carry.acceleration, command, physical)
            time = physics.core.steps[:, 0] / self.sim.freq
            clearance = self.scene_clearance(physics.states.pos[:, 0], time, carry.scene_state)
            task_state = self.update_task(
                carry.task,
                carry.physics.states,
                physics.states,
                time,
                clearance,
                scene_state=carry.scene_state,
            )
            physics = pytree_replace(physics, carry.physics, world_mask(physics), carry.done)
            observation = carry.observation
            if self.sensor is not None:
                observation = self.sensor.update(
                    observation, physics, ~carry.done, scene_state=carry.scene_state
                )
            commands = carry.commands.replace(
                applied=jnp.where(carry.done[:, None], carry.commands.applied, command),
                applied_input=jnp.where(
                    carry.done[:, None], carry.commands.applied_input, physical
                ),
            )
            acc = jnp.where(carry.done[:, None], carry.acceleration, acc)
            return carry.replace(
                physics=physics,
                task=task_state,
                observation=observation,
                commands=commands,
                acceleration=acc,
            ), None

        result, _ = jax.lax.scan(advance, state, None, length=self.substeps)
        return result

    def _reference_observation(self, state: EnvState) -> dict[str, Array]:
        """Encode tracking reference consistently for Tracking and Racing."""
        physics = state.physics.states
        p, v, q, w = (
            physics.pos[:, 0],
            physics.vel[:, 0],
            physics.quat[:, 0],
            physics.ang_vel[:, 0],
        )
        target, tv, ta = self.task.target(state.time)
        future = jnp.stack(
            [self.task.target(state.time + offset)[0] - p for offset in (0.2, 0.5, 1.0)],
            -2,
        ).reshape(len(p), -1)
        features = jnp.concatenate([target - p, tv - v, ta, q, w, v, future], -1)
        return {"state": features}

    def _observe_state(self, state: EnvState) -> dict[str, Array]:
        """Concrete environments define the task observation."""
        raise NotImplementedError

    def _observe(self, state: EnvState) -> dict[str, Array]:
        observation = self._observe_state(state)
        observation["velocity_body"] = Rotation.from_quat(state.physics.states.quat[:, 0]).apply(
            state.physics.states.vel[:, 0], inverse=True
        )
        if self.sensor is not None:
            observation.update(self.sensor.encode(state.observation, physics=state.physics))
            if self.sensor_age_in_state:
                age_in_sensor_periods = jnp.clip(
                    observation["sensor_age_s"] * self.sensor.config.frequency_hz, 0.0, 10.0
                )
                observation["state"] = jnp.concatenate(
                    [observation["state"], age_in_sensor_periods[:, None]], axis=-1
                )
        if self.observation_config.get("fan_lidar") is not None:
            physics = state.physics.states
            observation["fan_state"] = jnp.concatenate(
                [physics.vel[:, 0, :2], state.previous_action[:, :2] * 6.0], axis=-1
            )
            goal = jnp.broadcast_to(self.goal_for(state), physics.pos[:, 0].shape)
            observation["fan_command"] = goal[:, :2] - physics.pos[:, 0, :2]
        if self.observation_config.get("visfly", False):
            physics = state.physics.states
            quaternion = physics.quat[:, 0]
            if self.observation_config.get("visfly_quaternion_order", "xyzw") == "wxyz":
                quaternion = quaternion[:, jnp.array([3, 0, 1, 2])]
            position_scale = self.observation_config.get("visfly_position_scale", 1.0)
            observation["visfly_state"] = jnp.concatenate(
                [
                    physics.pos[:, 0] / position_scale,
                    quaternion,
                    physics.vel[:, 0],
                    physics.ang_vel[:, 0],
                ],
                axis=-1,
            )
            observation["visfly_target"] = (
                jnp.broadcast_to(self.goal_for(state), physics.pos[:, 0].shape) / position_scale
            )
        return observation

    def observe_privileged(self, state: EnvState) -> Array:
        """Only Navigation exposes the optional training-only privileged state."""
        raise ValueError("Privileged observation requires Navigation")

    def _shared_cost(self, before: EnvState, after: EnvState, action: Array, error):
        """Apply the same physical smoothness and first-event penalties to each task."""
        smooth = 0.01 * jnp.sum(action**2, -1)
        smooth += 0.01 * jnp.sum((action - before.previous_action) ** 2, -1)
        failure = after.done & (after.task.event != Event.SUCCESS) & ~before.done
        success = (after.task.event == Event.SUCCESS) & ~before.done
        return jnp.where(before.done, 0.0, (error + smooth) * self.dt + 20 * failure - 5 * success)

    def task_reward_components(
        self, before: EnvState, after: EnvState, action: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Supply the configured Task reward and its terms to every learner."""
        if not self.task.reward_shaping:
            return self._base_task_reward_components(before, after, action)
        return shaped_task_cost(self, before, after, action, self.task.reward_shaping)

    def _base_task_reward_components(
        self, before: EnvState, after: EnvState, action: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Return the unshaped physical Task reward before configured coefficients."""
        raise NotImplementedError

    def _tracking_error(self, after: EnvState):
        """Compute physical reference errors without encoding a network action."""
        p, v = after.physics.states.pos[:, 0], after.physics.states.vel[:, 0]
        target, tv, _ = self.task.target(after.time)
        return 5 * jnp.sum((p - target) ** 2, -1) + 0.2 * jnp.sum((v - tv) ** 2, -1)

    def update_task(
        self, state: TaskState, before, after, time, clearance, *, scene_state=None
    ) -> TaskState:
        """Apply common terminal-event priority with concrete task success semantics."""
        p = after.pos[:, 0]
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate([p, after.vel[:, 0], after.quat[:, 0], after.ang_vel[:, 0]], -1)
            ),
            -1,
        )
        collision = (clearance <= self.task.radius) | (p[:, 2] <= self.task.radius)
        in_bounds = jnp.all((p >= self.task.low) & (p <= self.task.high), -1)
        success, gates, missed = self._task_progress(
            state, before, after, time, scene_state=scene_state
        )
        event = first_event(
            collision, finite, in_bounds, success, time >= self.task.duration - 1e-5
        )
        event = jnp.where((event == Event.RUNNING) & missed, Event.MISSED_GATE, event)
        active = state.event == Event.RUNNING
        target, _, _ = self.task.target(time)
        error = jnp.sum((p - target) ** 2, -1)
        return TaskState(
            event=jnp.where(active, event, state.event),
            gates=jnp.where(active, gates, state.gates),
            error_sum=state.error_sum + jnp.where(active, error, 0.0),
            samples=state.samples + active.astype(jnp.int32),
            min_clearance=jnp.where(
                active,
                jnp.minimum(state.min_clearance, clearance - self.task.radius),
                state.min_clearance,
            ),
        )

    def _task_progress(self, state: TaskState, before, after, time, *, scene_state=None):
        """Concrete task decides what counts as success and a missed gate."""
        raise NotImplementedError

    def cost(self, before: EnvState, after: EnvState, action: Array) -> Array:
        """Return the same configured Task cost consumed by the Brax adapter."""
        return self.task_reward_components(before, after, action)[0]


class TrackingEnv(BaseEnv):
    """Quadrotor trajectory tracking with a time-indexed reference."""

    task_name = "tracking"

    def _observe_state(self, state: EnvState) -> dict[str, Array]:
        """Expose tracking position, velocity and future reference errors."""
        return self._reference_observation(state)

    def _task_progress(self, state: TaskState, before, after, time, *, scene_state=None):
        """Complete tracking at the configured trajectory duration."""
        return (
            time >= self.task.duration - 1e-5,
            state.gates,
            jnp.zeros_like(time, dtype=bool),
        )

    def _base_task_reward_components(
        self, before: EnvState, after: EnvState, action: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Evaluate reference tracking accuracy and common physical penalties."""
        return self._shared_cost(before, after, action, self._tracking_error(after)), {}


class RacingEnv(BaseEnv):
    """Quadrotor racing with directional gate events and tracking-based shaping."""

    task_name = "racing"

    def _observe_state(self, state: EnvState) -> dict[str, Array]:
        """Add the active gate and completion progress to reference observations."""
        base = self._reference_observation(state)
        count = len(self.task.gate_order)
        gate = self.task.gate_order[jnp.minimum(state.task.gates, count - 1)]
        base["state"] = jnp.concatenate(
            [
                base["state"],
                self.task.gate_positions[gate] - state.physics.states.pos[:, 0],
                (state.task.gates / count)[:, None],
            ],
            -1,
        )
        return base

    def _task_progress(self, state: TaskState, before, after, time, *, scene_state=None):
        """Count only forward crossings inside the original gate aperture."""
        gates = state.gates
        count = len(self.task.gate_order)
        gate = self.task.gate_order[jnp.minimum(gates, count - 1)]
        center, rot = self.task.gate_positions[gate], self.task.gate_rotations[gate]
        prev = jnp.einsum("bij,bj->bi", jnp.swapaxes(rot, -1, -2), before.pos[:, 0] - center)
        curr = jnp.einsum("bij,bj->bi", jnp.swapaxes(rot, -1, -2), after.pos[:, 0] - center)
        crossed = (prev[:, 0] < 0) & (curr[:, 0] >= 0) & (gates < count)
        fraction = -prev[:, 0] / jnp.maximum(curr[:, 0] - prev[:, 0], 1e-8)
        hit = prev + fraction[:, None] * (curr - prev)
        aperture = jnp.all(jnp.abs(hit[:, 1:]) <= 0.2 - self.task.radius, -1)
        gates = gates + (crossed & aperture).astype(jnp.int32)
        missed = crossed & ~aperture & jnp.all(jnp.abs(hit[:, 1:]) < 0.36, -1)
        return gates == count, gates, missed

    def _base_task_reward_components(
        self, before: EnvState, after: EnvState, action: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Retain the racing reward used by existing checkpoints."""
        return self._shared_cost(before, after, action, self._tracking_error(after)), {}


class NavigationEnv(BaseEnv):
    """Obstacle-aware goal navigation, including privileged training observations."""

    task_name = "navigation"
    allows_goal_observation = True

    def _validate_task_setup(self, scene_name):
        """Navigation requires a reachable initial location and an explicit scene."""
        if scene_name == "empty":
            raise ValueError("Navigation requires an explicitly selected scene")
        if float(self.scene.clearance(self.task.start, 0.0)) < 0.22:
            raise ValueError("Navigation nominal reset position lacks 0.15 m body clearance")

    def _sample_reset_positions(self, key, count, *, randomize_position):
        """Sample navigation starts and reject collisions in the selected scene."""
        width = jnp.array([0.25, 0.25, 0.10])

        def sample_positions(sample_key):
            if randomize_position:
                return jax.random.uniform(
                    sample_key,
                    (count, 3),
                    minval=jnp.array([2.0, -18.0, 1.0]),
                    maxval=jnp.array([94.0, 18.0, 5.5]),
                )
            return self.task.start + jax.random.uniform(
                sample_key, (count, 3), minval=-width, maxval=width
            )

        positions = sample_positions(key)

        def resample(carry):
            previous, random_key = carry
            random_key, sample_key = jax.random.split(random_key)
            candidate = sample_positions(sample_key)
            invalid = self.scene.clearance(previous, 0.0) - self.task.radius < 0.15
            return jnp.where(invalid[:, None], candidate, previous), random_key

        positions, _ = jax.lax.while_loop(
            lambda carry: jnp.any(self.scene.clearance(carry[0], 0.0) - self.task.radius < 0.15),
            resample,
            (positions, key),
        )
        return positions

    def _observe_state(self, state: EnvState) -> dict[str, Array]:
        """Encode egocentric velocity, target direction and upright orientation."""
        physics = state.physics.states
        p, v, q = physics.pos[:, 0], physics.vel[:, 0], physics.quat[:, 0]
        rot = Rotation.from_quat(q)
        goal = self.goal_for(state)
        delta = goal - p
        distance = jnp.linalg.norm(delta, axis=-1)
        if self.observation_config.get("navigation_relative_goal", False):
            goal_feature = delta / 100.0
        else:
            goal_feature = delta / jnp.maximum(distance[:, None], 1e-6)
            goal_feature = goal_feature * jnp.minimum(3.0, distance)[:, None]
        body_v = rot.apply(v, inverse=True)
        body_target = rot.apply(goal_feature, inverse=True)
        up = rot.apply(jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), p.shape), inverse=True)
        features = jnp.concatenate([body_v, body_target, up, jnp.full((len(p), 1), 0.07)], -1)
        if self.navigation_goal_observation:
            height_error = (goal[..., 2] - p[:, 2]) / (self.task.high[2] - self.task.low[2])
            start = self.task.start if state.scene_state is None else state.scene_state.start
            remaining = distance / jnp.linalg.norm(goal - start, axis=-1)
            features = jnp.concatenate([features, height_error[:, None], remaining[:, None]], -1)
        if self.observation_config.get("navigation_world_rotation", False):
            # World acceleration needs body-to-world attitude beyond body-frame up.
            features = jnp.concatenate([features, rot.as_matrix().reshape(len(p), 9)], -1)
        if self.task.finite_horizon:
            remaining_time = jnp.clip(1.0 - state.time / self.task.duration, 0.0, 1.0)
            features = jnp.concatenate([features, remaining_time[:, None]], -1)
        return {"state": features}

    def _task_progress(self, state: TaskState, before, after, time, *, scene_state=None):
        """Reach the physical goal sphere before the 300-second deadline."""
        goal = self.task.goal if scene_state is None else scene_state.goal
        success = jnp.linalg.norm(after.pos[:, 0] - goal, axis=-1) <= self.task.goal_radius
        return success, state.gates, jnp.zeros_like(time, dtype=bool)

    def observe_privileged(self, state: EnvState) -> Array:
        """Expose training-only ideal geometry rays without changing actor observations."""
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
        ranges = (
            self.scene.raycast(
                position,
                directions,
                state.time,
                max_range=40.0,
                scene_state=state.scene_state,
            )
            if state.scene_state is not None
            else self.scene.raycast(position, directions, state.time, max_range=40.0)
        )
        clearance = self.scene_clearance(position, state.time, state.scene_state) - self.task.radius
        return jnp.concatenate(
            [
                self._observe_state(state)["state"],
                position / jnp.array([100.0, 20.0, 6.0]),
                (self.goal_for(state) - position) / jnp.array([100.0, 40.0, 6.0]),
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

    def _base_task_reward_components(
        self, before: EnvState, after: EnvState, action: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Evaluate the selected Navigation reward before optional shaping."""
        if self.task.reward_profile == "goal_reaching":
            from drone_playground.simulation.task_rewards import goal_reaching_reward

            reward, terms = goal_reaching_reward(self, before, after, action)
            return -reward, terms
        if self.task.reward_profile in {"fan2025", "fan2025_arrival", "visfly2025"}:
            from drone_playground.simulation.task_rewards import paper_navigation_reward

            reward, terms = paper_navigation_reward(self, before, after, action)
            return -reward, terms
        if self.task.reward_profile == "legacy_motion_navigation_v2":
            reward, terms = motion_navigation_reward(self, before, after, action)
            return -reward, terms
        position = after.physics.states.pos[:, 0]
        velocity = after.physics.states.vel[:, 0]
        direction = self.goal_for(after) - position
        distance = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        if self.task.reward_profile == "minimum_time_navigation":
            error = jnp.ones(position.shape[0])
        else:
            desired = direction / jnp.maximum(distance, 1e-6) * jnp.minimum(distance, 3.0)
            error = jnp.sum((velocity - desired) ** 2, -1)
        clearance = self.scene_clearance(position, after.time, after.scene_state) - self.task.radius
        error = error + 5 * jax.nn.softplus((0.5 - clearance) * 8) / 8
        terms = (
            {"navigation_time_cost": jnp.where(before.done, 0.0, self.dt)}
            if self.task.reward_profile == "minimum_time_navigation"
            else {}
        )
        return self._shared_cost(before, after, action, error), terms


def make_environment(*, task: str = "tracking", **kwargs) -> BaseEnv:
    """Construct the concrete task environment from one runtime task selector."""
    classes = {"tracking": TrackingEnv, "racing": RacingEnv, "navigation": NavigationEnv}
    if task not in classes:
        raise ValueError(f"Unknown task: {task}")
    return classes[task](**kwargs)
