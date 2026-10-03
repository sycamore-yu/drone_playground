"""Unified navigation task: SANDO-style scene, arrival, body collision, time limit.

One task protocol owns the navigation episode for every method. The active
recipe uses a 300 s limit; the archived v1 keeps 40 s. Both use the 0.5 m arrival radius and the
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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State
from crazyflow.sim.data import SimData
from flax import struct

from drone_playground.environments.initialization import initialize_simulation
from drone_playground.environments.observations.state import (
    NavigationObservation,
    numerically_valid_observation,
)
from drone_playground.environments.scenes.geometry import (
    body_centre_from_state,
    euclidean_norm,
)
from drone_playground.environments.scenes.procedural_navigation import make_bank
from drone_playground.environments.sensors.depth import cast_depth
from drone_playground.environments.sensors.lidar import Mid360Lidar, cast_lidar
from drone_playground.environments.tasks.navigation.events import OUTCOME_RUNNING, NavigationEvents
from drone_playground.environments.tasks.rigid_body import RigidBodyTask


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
    goal: jax.Array
    scene_time_offset: jax.Array = 0.0


@dataclass
class NavigationTask(RigidBodyTask):
    """Navigation observations, first-event outcomes and the configured reward."""

    name: str
    freq: int
    physics_freq: int
    duration: float
    goal_radius: float
    body_radius: float
    reference_count: int
    time_limit_kind: str
    command_distribution: dict
    observation: object
    reward: object
    owns_reset_randomization = True

    def bind(self, env):
        self.events = NavigationEvents(self.goal_radius, self.body_radius)
        count = self.reference_count if env.role == "train" else env.count
        count = max(count, len(getattr(env.scene, "scene_ids", ())))
        env.bank, env.scene_manifest = make_bank(env.scene, env.reference_seed, count)
        env.reset_randomization = env.conditions.get("reset_randomization")
        env.training_collision_mode = env.conditions.get("navigation_collision_mode", "terminate")
        if env.training_collision_mode not in ("terminate", "continuous_loss"):
            raise ValueError("Unknown navigation collision mode")
        env.physics_freq = self.physics_freq
        env.goal_radius, env.body_radius = self.goal_radius, self.body_radius
        env.simulation = initialize_simulation(
            env.dynamics,
            env.duration,
            env.freq,
            env.device,
            tuple(np.asarray(env.bank.start[0], np.float32)),
            control_mode=env.controller.native_mode,
        )
        env.sim = env.simulation.sim
        if self.physics_freq != env.sim.freq:
            env.sim.close()
            raise ValueError("Requested physics frequency differs from the actual dynamics backend")
        env.default = env.sim.default_data
        env.reset_fn = env.sim.build_reset_fn()
        env.controller.bind(
            env.simulation.single_action_space.low,
            env.simulation.single_action_space.high,
            env.dynamics,
        )
        env.low, env.high = env.controller.low, env.controller.high
        hover = jnp.array([0.0, 0.0, 0.0, float(env.default.params.mass[0]) * 9.81])
        if hasattr(env.controller, "hover"):
            hover = env.controller.hover(env.default)
        env.hover_action = 2 * (hover - env.low) / (env.high - env.low) - 1
        env.identity_quat = jnp.asarray(env.default.states.quat[0, 0])
        env.bounds_low = jnp.asarray(env.bank.world_low, jnp.float32)
        env.bounds_high = jnp.asarray(env.bank.world_high, jnp.float32)
        sensor = env.sensor
        env.depth_stride = None
        env.points_per_frame = 0 if sensor is None else sensor.points_per_frame
        env.sensor_period = 1 if sensor is None else sensor.period_steps(env.freq)
        env.sensor_calibration = None if sensor is None else sensor.calibration()
        if (
            sensor is not None
            and self.observation.sensor_size
            != sensor.history * sensor.points_per_frame * sensor.channels
        ):
            raise ValueError("Sensor history and observation specification differ")

    def scenario(self, env, scenario_id) -> dict:
        """Human-readable identity of one scenario instance."""
        index = int(np.asarray(scenario_id))
        return {
            "scenario_id": index,
            **env.bank.labels(index),
            "obstacles": env.bank.active_count(index),
            "start": [float(value) for value in np.asarray(env.bank.start[index])],
            "goal": [float(value) for value in np.asarray(env.bank.goal[index])],
        }

    def observe(self, env, data: NavigationData) -> jax.Array:
        states = data.sim_data.states
        goal = data.goal
        if env.sensor is None:
            return env.task.observation(states, goal, data.previous_action)
        return env.task.observation(
            states, goal, data.previous_action, {"values": data.sensor_values}
        )

    def sample_sensor(self, env, data: NavigationData) -> NavigationData:
        """Generate the due measurement and refresh the history at its own rate.

        The scene is advanced and the body pose synchronised before the
        measurement is generated, so a sample always describes the pose at its
        recorded capture time. Frames are refreshed on a fixed control-step
        divisor, which makes the realised availability exact and recordable
        instead of an aliased 50/30 ratio.
        """
        if env.sensor is None:
            return data
        states = data.sim_data.states
        time = data.scene_time_offset + data.step_index.astype(jnp.float32) * env.dt
        position = states.pos[0, 0]
        quat = states.quat[0, 0]
        if isinstance(env.sensor, Mid360Lidar):
            # The scan phase is episode state, so a reset restarts the horizon
            # instead of continuing a cursor owned by the generator.
            frame = cast_lidar(
                env.sensor,
                env.bank,
                data.scenario_id,
                position,
                quat,
                time,
                data.sensor_sequence,
            )
        else:
            frame = cast_depth(
                env.sensor,
                env.bank,
                data.scenario_id,
                position,
                quat,
                time,
                env.depth_stride,
            )
        values = env.sensor.frame_values(frame)
        noise = getattr(env, "observation_noise", {})
        std, dropout = noise.get("sensor_std_m", 0.0), noise.get("sensor_dropout_probability", 0.0)
        if std or dropout:
            from drone_playground.environments.randomization import point_measurement_noise

            noise_key = jax.random.fold_in(data.sim_data.core.rng_key, data.sensor_sequence)
            valid = values[..., -1] > 0.5
            coordinates = values[..., :1] if env.sensor.channels == 2 else values[..., :3]
            noisy, valid = point_measurement_noise(coordinates, valid, noise_key, std, dropout)
            if env.sensor.channels == 2:
                values = values.at[..., 0].set(
                    jnp.clip(noisy[..., 0], env.sensor.near_m, env.sensor.far_m)
                )
            else:
                values = values.at[..., :3].set(noisy).at[..., 3].set(euclidean_norm(noisy))
            values = values.at[..., -1].set(valid.astype(jnp.float32))
        due = (data.step_index % env.sensor_period) == 0
        history = jnp.where(
            due,
            jnp.concatenate([data.sensor_values[1:], values[None]]),
            data.sensor_values,
        )
        times = jnp.where(
            due,
            jnp.concatenate([data.sensor_time[1:], frame.time[None]]),
            data.sensor_time,
        )
        sequence = data.sensor_sequence + due.astype(jnp.int32)
        return data.replace(sensor_values=history, sensor_time=times, sensor_sequence=sequence)

    def proprioception(self, env, data: NavigationData) -> jax.Array:
        """Value input before auto-reset, independent of the ray-casting graph."""
        observer = NavigationObservation(
            include_goal=env.task.observation.include_goal,
            include_previous_action=env.task.observation.include_previous_action,
            action_size=env.task.observation.action_size,
        )
        return observer(data.sim_data.states, data.goal, data.previous_action)

    def reset(
        self,
        env,
        rng: jax.Array,
        scenario_id: jax.Array | None = None,
        initial_state: dict | None = None,
    ) -> State:
        if rng.dtype == jnp.uint32:
            rng = jax.random.wrap_key_data(rng)
        key, scene_key = jax.random.split(rng)
        sim_data = env.default.replace(core=env.default.core.replace(rng_key=key))
        sim_data = env.reset_fn(sim_data, env.default)
        sim_data = env.dynamics.randomize(sim_data, jax.random.fold_in(key, 101))
        if scenario_id is None:
            scenario_id = jax.random.randint(
                scene_key, (), 0, env.bank.num_instances, dtype=jnp.int32
            )
        scenario_id = jnp.asarray(scenario_id, jnp.int32)
        start = env.bank.start[scenario_id]
        goal = env.bank.goal[scenario_id]
        from drone_playground.environments.randomization import sample_command

        goal = sample_command(
            goal,
            jax.random.fold_in(key, 110),
            getattr(env, "command_distribution", {}),
        )
        initial_velocity = jnp.zeros(3)
        initial_quaternion = env.identity_quat
        scene_phase = jnp.float32(0)
        if env.reset_randomization:
            from drone_playground.environments.tasks.navigation.initialization import (
                sample_initial_state,
            )

            start, initial_velocity, scene_phase = sample_initial_state(
                env.bank,
                scenario_id,
                jax.random.fold_in(key, 73),
                env.body_radius,
                float(getattr(env.task.reward, "target_speed", 2.0)),
                env.reset_randomization,
            )
            from jax.scipy.spatial.transform import Rotation

            width = jnp.asarray(
                env.reset_randomization.get("orientation_half_width_rad", [0, 0, 0])
            )
            angles = (
                jax.random.uniform(jax.random.fold_in(key, 74), (3,), minval=-1, maxval=1) * width
            )
            initial_quaternion = (
                Rotation.from_quat(initial_quaternion) * Rotation.from_euler("xyz", angles)
            ).as_quat()
        angular_velocity = jax.random.normal(jax.random.fold_in(key, 75), (3,)) * jnp.asarray(
            (env.reset_randomization or {}).get("angular_velocity_std_radps", 0.0)
        )
        if initial_state is not None:
            start = initial_state["position"]
            initial_velocity = initial_state["velocity"]
            initial_quaternion = initial_state["quaternion"]
        # The scenario owns the initial pose; the reset pipeline only supplies
        # motor state and the physical default data.
        states = sim_data.states.replace(
            pos=start[None, None],
            quat=jnp.broadcast_to(initial_quaternion, sim_data.states.quat.shape),
            vel=jnp.broadcast_to(initial_velocity, sim_data.states.vel.shape),
            ang_vel=jnp.broadcast_to(angular_velocity, sim_data.states.ang_vel.shape),
        )
        sim_data = sim_data.replace(states=states)
        history = 1 if env.sensor is None else env.sensor.history
        channels = 1 if env.sensor is None else env.sensor.channels
        data = NavigationData(
            sim_data=sim_data,
            scenario_id=scenario_id,
            step_index=jnp.int32(0),
            previous_distance=euclidean_norm(start - goal),
            previous_action=env.hover_action,
            sensor_values=jnp.zeros((history, env.points_per_frame, channels), jnp.float32),
            sensor_time=jnp.zeros((history,), jnp.float32),
            sensor_sequence=jnp.int32(0),
            goal=goal,
            scene_time_offset=scene_phase,
        )
        data = env.task.sample_sensor(env, data)
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
            obs=env.observation(data),
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                "terminated": zero,
                "physical_parameters": env.dynamics.physical_parameters(sim_data),
                "outcome": jnp.int32(OUTCOME_RUNNING),
                "terminal_proprioception": env.proprioception(data),
            },
        )

    def probe(self, env, state):
        data = state.pipeline_state
        scenario_id = data.scenario_id
        first_substep = data.step_index * env.substeps

        def probe(current, offset):
            time = data.scene_time_offset + (first_substep + offset + 1) * env.dt_physics
            centre = body_centre_from_state(
                current.sim_data.states.pos[0, 0], current.sim_data.states.quat[0, 0]
            )
            clearance, hit = env.task.events.clearance(env.bank, scenario_id, centre, time)
            return clearance, hit

        return probe

    def finish(self, env, state, data, action, physical, evidence):
        clearance, collided = evidence
        sim_data = data.sim_data
        states = sim_data.states
        position = states.pos[0, 0]
        goal = data.goal
        distance, arrived, out_of_bounds, _ = env.task.events.events(
            env.bank, position, goal, collided, jnp.array(False)
        )
        data = data.replace(
            sim_data=sim_data,
            step_index=data.step_index + 1,
            previous_distance=distance,
            previous_action=action,
        )
        numerical_failure = ~numerically_valid_observation(env.observation(data))
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
        if env.training_collision_mode == "continuous_loss":
            # The explicitly selected training surrogate continues through
            # geometric contact to retain penetration/escape loss gradients.
            # Collision labels below remain true. All evaluation instances are
            # constructed with the original hard terminal rule.
            terminated = arrived | out_of_bounds | numerical_failure
        reward = env.task.reward(
            arrived=arrived,
            collided=collided,
            out_of_bounds=out_of_bounds,
            numerical_failure=numerical_failure,
            previous_distance=state.pipeline_state.previous_distance,
            distance=distance,
            clearance=clearance,
            action=action,
            previous_action=state.pipeline_state.previous_action,
            velocity=states.vel[0, 0],
            goal_delta=goal - states.pos[0, 0],
            dt=env.dt,
        )
        data = env.task.sample_sensor(env, data)
        _, _, _, outcome = env.task.events.events(
            env.bank, position, goal, collided, numerical_failure
        )
        metrics = {
            **state.metrics,
            "goal_distance": distance,
            "clearance": clearance,
            "action_saturation": jnp.mean((jnp.abs(action) >= 0.99).astype(jnp.float32)),
            "physical_thrust": (
                data.sim_data.controls.attitude.staged_cmd[0, 0, 3]
                if env.controller.native_mode == "attitude" and env.controller.input_kind == "state"
                else physical[0]
                if env.controller.input_kind == "rates"
                else physical[3]
            ),
            "arrived": arrived.astype(jnp.float32),
            "collision": collided.astype(jnp.float32),
            "out_of_bounds": out_of_bounds.astype(jnp.float32),
            "numerical_failure": numerical_failure.astype(jnp.float32),
            "failure": (collided | out_of_bounds | numerical_failure).astype(jnp.float32),
        }
        return state.replace(
            pipeline_state=data,
            obs=env.observation(data),
            reward=reward,
            done=terminated.astype(jnp.float32),
            metrics=metrics,
            info={
                **state.info,
                "terminated": terminated.astype(jnp.float32),
                "outcome": outcome,
                "terminal_proprioception": env.proprioception(data),
            },
        )
