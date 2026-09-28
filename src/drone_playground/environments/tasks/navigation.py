"""Unified navigation task: SANDO-style scene, arrival, body collision, time limit.

One task protocol owns the navigation episode for every method. The frozen
rules are the user-confirmed 40 s limit, the 0.5 m arrival radius and the
Crazyflie body-collision failure, with collision taking priority over arrival
in the same step. Static and dynamic navigation share this implementation; the
difference is which scene families the scene bank contains, and the composition
refuses a mismatch.

Collision is evaluated at every 500 Hz physics substep against the analytic
obstacles and ground. This catches the tested 40 m/s thin-bar crossing; it is
discrete detection, not a continuous-collision guarantee at arbitrary speeds.
Body geometry is the 0.07 m sphere of the pinned Crazyflow model, offsets included.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from crazyflow.sim.data import SimData
from flax import struct

from drone_playground.environments.observations import NavigationObservation
from drone_playground.environments.scenes.navigation import (
    BODY_RADIUS_M,
    SceneBank,
    body_centre_from_state,
    clearance_and_collision,
    euclidean_norm,
)
from drone_playground.environments.sensors.depth import DepthCamera, cast_depth
from drone_playground.environments.sensors.lidar import Mid360Lidar, cast_lidar
from drone_playground.environments.tasks.tracking import numerically_valid_observation
from drone_playground.execution.controllers.crazyflow import AttitudeControl
from drone_playground.execution.transition import ExecutionTransition
from drone_playground.learning.objectives import NavigationObjective
from drone_playground.models.crazyflow import CrazyflowModel


@struct.dataclass
class NavigationData:
    """All evolving navigation state; the immutable scene bank belongs to the task."""

    sim_data: SimData
    scenario_id: jax.Array
    step_index: jax.Array
    previous_distance: jax.Array
    previous_action: jax.Array
    sensor_values: jax.Array
    sensor_time: jax.Array
    sensor_sequence: jax.Array


class NavigationEnv(Env):
    """Single-environment navigation interface; Brax owns the outer batch axis."""

    def __init__(
        self,
        scene_bank: SceneBank,
        task: str = "navigation",
        dynamics: str = "first_principles",
        drone: str = "cf2x_L250",
        freq: int = 50,
        duration: float = 40.0,
        goal_radius: float = 0.5,
        body_radius: float = BODY_RADIUS_M,
        device: str = "cpu",
        model=None,
        controller=None,
        observation=None,
        objective=None,
        sensor: DepthCamera | Mid360Lidar | None = None,
        stride: int | None = None,
    ):
        if freq <= 0 or 500 % freq:
            raise ValueError("Task frequency must be a positive divisor of 500 Hz")
        if duration <= 0:
            raise ValueError("Navigation duration must be positive")
        if goal_radius <= 0:
            raise ValueError("Goal radius must be positive")
        self.bank = scene_bank
        self.model = model or CrazyflowModel(dynamics, drone)
        self.controller = controller or AttitudeControl()
        self.observer = observation or NavigationObservation()
        self.objective = objective or NavigationObjective()
        self.task = task
        self.dynamics = self.model.forward
        self.drone = self.model.drone
        self.freq = freq
        self.duration = float(duration)
        self.goal_radius = float(goal_radius)
        self.body_radius = float(body_radius)
        self.episode_length = round(self.duration * freq)

        self.reference = self.model.create_navigation(
            self.duration, freq, device, tuple(np.asarray(scene_bank.start[0], np.float32))
        )
        self.sim = self.reference.sim
        self.default = self.sim.default_data
        self.reset_fn = self.sim.build_reset_fn()
        self.substeps = self.reference.n_substeps
        self.dt_physics = 1.0 / (self.freq * self.substeps)
        self.low = jnp.asarray(self.reference.single_action_space.low)
        self.high = jnp.asarray(self.reference.single_action_space.high)
        self.controller.bind(self.low, self.high)
        self.execution = ExecutionTransition(
            self.controller.apply, self.model.advance, self.substeps
        )
        hover = jnp.array([0.0, 0.0, 0.0, float(self.default.params.mass[0]) * 9.81])
        self.hover_action = 2 * (hover - self.low) / (self.high - self.low) - 1
        # The pinned model owns the rest attitude; Crazyflow uses xyzw order.
        self.identity_quat = jnp.asarray(self.default.states.quat[0, 0])
        self.bounds_low = jnp.asarray(scene_bank.world_low, jnp.float32)
        self.bounds_high = jnp.asarray(scene_bank.world_high, jnp.float32)

        self.sensor = sensor
        if sensor is None:
            self.points_per_frame = 0
            self.sensor_period = 1
            self.depth_stride = 1
        else:
            self.depth_stride = stride
            self.points_per_frame = sensor.points_per_frame
            self.sensor_period = sensor.period_steps(freq)
            expected = sensor.history * sensor.points_per_frame * sensor.channels
            if self.observer.sensor_size != expected:
                raise ValueError(
                    "the observation sensor block does not match the sensor frame size; "
                    f"observation expects {self.observer.sensor_size}, sensor produces "
                    f"{expected}"
                )
        self.sensor_calibration = None if sensor is None else sensor.calibration()

    # -- Brax Env interface ------------------------------------------------------------

    @property
    def observation_size(self) -> int:
        return self.observer.size

    @property
    def action_size(self) -> int:
        return 4

    @property
    def backend(self) -> str:
        return "crazyflow"

    @property
    def dt(self) -> float:
        return 1.0 / self.freq

    def scenario(self, scenario_id) -> dict:
        """Human-readable identity of one scenario instance."""
        index = int(np.asarray(scenario_id))
        return {
            "scenario_id": index,
            **self.bank.labels(index),
            "obstacles": self.bank.active_count(index),
            "start": [float(value) for value in np.asarray(self.bank.start[index])],
            "goal": [float(value) for value in np.asarray(self.bank.goal[index])],
        }

    def observation(self, data: NavigationData) -> jax.Array:
        states = data.sim_data.states
        goal = self.bank.goal[data.scenario_id]
        if self.sensor is None:
            return self.observer(states, goal, data.previous_action)
        return self.observer(states, goal, data.previous_action, {"values": data.sensor_values})

    def _sample_sensor(self, data: NavigationData) -> NavigationData:
        """Generate the due measurement and refresh the history at its own rate.

        The scene is advanced and the body pose synchronised before the
        measurement is generated, so a sample always describes the pose at its
        recorded capture time. Frames are refreshed on a fixed control-step
        divisor, which makes the realised availability exact and recordable
        instead of an aliased 50/30 ratio.
        """
        if self.sensor is None:
            return data
        states = data.sim_data.states
        time = data.step_index.astype(jnp.float32) * self.dt
        position = states.pos[0, 0]
        quat = states.quat[0, 0]
        if isinstance(self.sensor, Mid360Lidar):
            # The scan phase is episode state, so a reset restarts the horizon
            # instead of continuing a cursor owned by the generator.
            frame = cast_lidar(
                self.sensor,
                self.bank,
                data.scenario_id,
                position,
                quat,
                time,
                data.sensor_sequence,
            )
        else:
            frame = cast_depth(
                self.sensor,
                self.bank,
                data.scenario_id,
                position,
                quat,
                time,
                self.depth_stride,
            )
        values = self.sensor.frame_values(frame)
        due = (data.step_index % self.sensor_period) == 0
        history = jnp.where(
            due, jnp.concatenate([data.sensor_values[1:], values[None]]), data.sensor_values
        )
        times = jnp.where(
            due, jnp.concatenate([data.sensor_time[1:], frame.time[None]]), data.sensor_time
        )
        sequence = data.sensor_sequence + due.astype(jnp.int32)
        return data.replace(sensor_values=history, sensor_time=times, sensor_sequence=sequence)

    def proprioception(self, data: NavigationData) -> jax.Array:
        """Value input before auto-reset, independent of the ray-casting graph."""
        observer = NavigationObservation(
            include_goal=self.observer.include_goal,
            include_previous_action=self.observer.include_previous_action,
            action_size=self.observer.action_size,
        )
        return observer(
            data.sim_data.states, self.bank.goal[data.scenario_id], data.previous_action
        )

    def reset(self, rng: jax.Array, scenario_id: jax.Array | None = None) -> State:
        if rng.dtype == jnp.uint32:
            rng = jax.random.wrap_key_data(rng)
        key, scene_key = jax.random.split(rng)
        sim_data = self.default.replace(core=self.default.core.replace(rng_key=key))
        sim_data = self.reset_fn(sim_data, self.default)
        if scenario_id is None:
            scenario_id = jax.random.randint(
                scene_key, (), 0, self.bank.num_instances, dtype=jnp.int32
            )
        scenario_id = jnp.asarray(scenario_id, jnp.int32)
        start = self.bank.start[scenario_id]
        goal = self.bank.goal[scenario_id]
        # The scenario owns the initial pose; the reset pipeline only supplies
        # motor state and the physical default data.
        states = sim_data.states.replace(
            pos=start[None, None],
            quat=jnp.broadcast_to(self.identity_quat, sim_data.states.quat.shape),
            vel=jnp.zeros_like(sim_data.states.vel),
            ang_vel=jnp.zeros_like(sim_data.states.ang_vel),
        )
        sim_data = sim_data.replace(states=states)
        history = 1 if self.sensor is None else self.sensor.history
        channels = 1 if self.sensor is None else self.sensor.channels
        data = NavigationData(
            sim_data=sim_data,
            scenario_id=scenario_id,
            step_index=jnp.int32(0),
            previous_distance=euclidean_norm(start - goal),
            previous_action=self.hover_action,
            sensor_values=jnp.zeros((history, self.points_per_frame, channels), jnp.float32),
            sensor_time=jnp.zeros((history,), jnp.float32),
            sensor_sequence=jnp.int32(0),
        )
        data = self._sample_sensor(data)
        zero = jnp.float32(0)
        metrics = {
            name: zero
            for name in (
                "goal_distance",
                "clearance",
                "action_saturation",
                "physical_thrust",
                "arrived",
                "collision",
                "out_of_bounds",
                "numerical_failure",
                "failure",
                "sensor_sequence",
            )
        }
        metrics["sensor_sequence"] = zero
        return State(
            pipeline_state=data,
            obs=self.observation(data),
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                "terminated": zero,
                "outcome": jnp.int32(OUTCOME_RUNNING),
                "terminal_proprioception": self.proprioception(data),
            },
        )

    def physical_action(self, action: jax.Array) -> jax.Array:
        return self.controller.physical_action(action)

    def step(self, state: State, action: jax.Array) -> State:
        data = state.pipeline_state
        physical = self.physical_action(action)
        scenario_id = data.scenario_id
        first_substep = data.step_index * self.substeps

        def probe(current, offset):
            time = (first_substep + offset + 1) * self.dt_physics
            centre = body_centre_from_state(current.states.pos[0, 0], current.states.quat[0, 0])
            clearance, hit = clearance_and_collision(
                self.bank, scenario_id, time, centre, self.body_radius
            )
            return clearance, hit

        sim_data, clearance, collided = self.execution.step_with_evidence(
            data.sim_data, physical, probe
        )

        states = sim_data.states
        position = states.pos[0, 0]
        goal = self.bank.goal[scenario_id]
        distance = euclidean_norm(position - goal)
        arrived = distance <= self.goal_radius
        out_of_bounds = jnp.any(position < self.bounds_low) | jnp.any(position > self.bounds_high)
        data = data.replace(
            sim_data=sim_data,
            step_index=data.step_index + 1,
            previous_distance=distance,
            previous_action=action,
        )
        numerical_failure = ~numerically_valid_observation(self.observation(data))
        # Numerical invalidity is a failed trial. Retain the last finite pose for
        # terminal recording; healthy transitions pass through unchanged and no
        # physical limit is relaxed.
        states = jax.tree.map(
            lambda new, old: jnp.where(numerical_failure, jax.lax.stop_gradient(old), new),
            data.sim_data.states,
            state.pipeline_state.sim_data.states,
        )
        data = data.replace(sim_data=data.sim_data.replace(states=states))
        terminated = collided | arrived | out_of_bounds | numerical_failure
        reward = self.objective(
            arrived=arrived,
            collided=collided,
            out_of_bounds=out_of_bounds,
            numerical_failure=numerical_failure,
            previous_distance=state.pipeline_state.previous_distance,
            distance=distance,
            clearance=clearance,
            action=action,
            previous_action=state.pipeline_state.previous_action,
        )
        data = self._sample_sensor(data)
        outcome = _outcome(collided, arrived, out_of_bounds, numerical_failure)
        metrics = {
            **state.metrics,
            "goal_distance": distance,
            "clearance": clearance,
            "action_saturation": jnp.mean((jnp.abs(action) >= 0.99).astype(jnp.float32)),
            "physical_thrust": physical[3],
            "arrived": arrived.astype(jnp.float32),
            "collision": collided.astype(jnp.float32),
            "out_of_bounds": out_of_bounds.astype(jnp.float32),
            "numerical_failure": numerical_failure.astype(jnp.float32),
            "failure": (collided | out_of_bounds | numerical_failure).astype(jnp.float32),
        }
        return state.replace(
            pipeline_state=data,
            obs=self.observation(data),
            reward=reward,
            done=terminated.astype(jnp.float32),
            metrics=metrics,
            info={
                **state.info,
                "terminated": terminated.astype(jnp.float32),
                "outcome": outcome,
                "terminal_proprioception": self.proprioception(data),
            },
        )

    @property
    def realised_sensor_rate_hz(self) -> float:
        """Actual frame availability after snapping to the control-step divisor."""
        return self.freq / self.sensor_period

    def close(self) -> None:
        self.reference.close()

    def step_physical(self, state, physical):
        normalized = 2 * (physical - self.low) / (self.high - self.low) - 1
        return self.step(state, normalized)

    def controller_observation(self, state):
        data = state.pipeline_state.sim_data.states
        return {
            name: np.asarray(getattr(data, name)[0, 0])
            for name in ("pos", "quat", "vel", "ang_vel")
        }


OUTCOME_RUNNING = 0
OUTCOME_ARRIVED = 1
OUTCOME_COLLISION = 2
OUTCOME_OUT_OF_BOUNDS = 3
OUTCOME_NUMERICAL = 4
OUTCOME_TIMEOUT = 5

OUTCOME_NAMES = {
    OUTCOME_RUNNING: "running",
    OUTCOME_ARRIVED: "arrived",
    OUTCOME_COLLISION: "collision",
    OUTCOME_OUT_OF_BOUNDS: "out_of_bounds",
    OUTCOME_NUMERICAL: "numerical_failure",
    OUTCOME_TIMEOUT: "timeout",
}


def _outcome(collided, arrived, out_of_bounds, numerical_failure) -> jax.Array:
    """Collision wins over arrival in the same step, as the protocol requires."""
    return jnp.where(
        collided,
        OUTCOME_COLLISION,
        jnp.where(
            numerical_failure,
            OUTCOME_NUMERICAL,
            jnp.where(
                out_of_bounds,
                OUTCOME_OUT_OF_BOUNDS,
                jnp.where(arrived, OUTCOME_ARRIVED, OUTCOME_RUNNING),
            ),
        ),
    ).astype(jnp.int32)
