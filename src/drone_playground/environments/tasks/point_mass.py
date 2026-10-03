"""Brax lifecycle for acceleration tasks with native point-mass rollout kernels.

Optimized training kernels and the public environment share these components and
the same Dynamics.step call. Task memory lives in Brax info; the physical state
remains PointMassState. Recurrent network memory belongs to the running method.
"""

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State
from jax.scipy.spatial.transform import Rotation

from drone_playground.environments.tasks.navigation.events import NavigationEvents


class PointMassTask:
    """Shared task configuration and public state contract for acceleration inputs."""

    handles_transport_delay = True
    owns_environment_effects = True
    reset_info_fields = (
        "scenario_id",
        "clock",
        "speed",
        "previous_command",
        "delay_ticks",
        "outcome",
        "gates_passed",
    )
    extra_fields = frozenset(
        {
            "reference_count",
            "numerical_guard",
            "arrival_sampling",
            "max_speed",
            "reference_gain",
            "velocity_limit",
            "integration",
            "floor_rule",
            "provenance",
        }
    )

    def __init__(
        self,
        *,
        name,
        freq,
        physics_freq,
        duration,
        goal_radius,
        body_radius,
        time_limit_kind,
        command_distribution,
        observation,
        loss,
        **settings,
    ):
        unknown = set(settings) - self.extra_fields
        if unknown:
            raise TypeError(f"Unknown acceleration task fields: {sorted(unknown)}")
        if (
            not np.isfinite([freq, physics_freq, duration, goal_radius, body_radius]).all()
            or min(freq, physics_freq, duration, goal_radius, body_radius) <= 0
            or physics_freq % freq
        ):
            raise ValueError(
                "Task requires positive finite duration/radii and integral physics substeps"
            )
        speeds = command_distribution.get("speed_range_mps")
        if speeds is not None and settings.get("max_speed") is not None:
            if not 0 <= speeds[0] <= speeds[1] <= settings["max_speed"]:
                raise ValueError("Commanded speed range exceeds the task maximum speed")
        self.name, self.freq, self.physics_freq = name, freq, physics_freq
        self.duration, self.goal_radius, self.body_radius = duration, goal_radius, body_radius
        self.time_limit_kind, self.command_distribution = time_limit_kind, command_distribution
        self.observation, self.loss = observation, loss
        self.settings = dict(
            settings,
            name=name,
            freq=freq,
            physics_freq=physics_freq,
            duration=duration,
            goal_radius=goal_radius,
            body_radius=body_radius,
            time_limit_kind=time_limit_kind,
            command_distribution=command_distribution,
        )
        self.events = NavigationEvents(goal_radius, body_radius)
        self.arrival_sampling = settings.get("arrival_sampling", "policy")
        self.dt, self.physics_dt = 1.0 / freq, 1.0 / physics_freq
        self.substeps = physics_freq // freq
        self.episode_length = round(duration * freq)
        self.action_size = 3

    def bind(self, env):
        """Use the environment's resolved components and conditions directly."""
        if env.sensor is None or env.sensor.source_rate_hz != self.freq:
            raise ValueError(
                "Synchronous sensor frequency must match the recurrent policy frequency"
            )
        if env.conditions.get("action_delay_steps", 0):
            raise ValueError(
                "This task owns physical transport delay; use milliseconds, not a second integer action delay"
            )
        self.dynamics, self.controller = env.dynamics, env.controller
        self.sensor, self.scene, self.reference_source = env.sensor, env.scene, env.reference
        self.conditions, self.role = env.conditions, env.role
        self.observation_noise = env.observation_noise
        self.environment_effects = env.environment_effects
        self.drone = env.dynamics.drone
        self.sensor_calibration = self.sensor.calibration()
        self.physics_engine = f"PointMassLag JAX {self.physics_freq}Hz; MuJoCo geometry/replay"
        self.observation_size = {
            "state": (self.observation.proprioception_size,),
            "sensor": (self.sensor.points_per_frame,)
            if self.sensor.policy_value_channels == 1
            else (self.sensor.points_per_frame, self.sensor.policy_value_channels),
            "valid": (self.sensor.points_per_frame,),
        }
        env.physics_freq = self.physics_freq
        env.hover_action = jnp.zeros(3)
        env.handles_transport_delay = True
        env.sensor_calibration = self.sensor_calibration
        env.sensor_period = 1
        env.body_radius, env.goal_radius = self.body_radius, self.goal_radius

    @staticmethod
    def physics(data):
        return data

    @staticmethod
    def with_physics(data, physical):
        del data
        return physical

    @staticmethod
    def _batch(data):
        return jax.tree.map(lambda value: value[None], data)

    @staticmethod
    def _single(data):
        return jax.tree.map(lambda value: value[0], data)

    def probe(self, env, state):
        del env, state
        return None

    def _public_observation(self, data, info):
        """Return named network inputs from the same measurement kernel."""
        physical = self._batch(data)
        if self.name in ("tracking", "hovering", "racing"):
            points, valid, proprio = self.measure(physical, info["clock"][None])
        else:
            bank = self.bank.select(info["scenario_id"][None])
            points, valid, proprio, _ = self.measure(
                bank, physical, info["clock"][None], info["speed"][None]
            )
        return dict(state=proprio[0], sensor=points[0], valid=valid[0])

    def controller_observation(self, env, state):
        del env
        data = state.pipeline_state
        return dict(
            pos=np.asarray(data.pos),
            vel=np.asarray(data.vel),
            quat=np.asarray(Rotation.from_matrix(data.rotation).as_quat()),
        )

    def reset(self, env, rng, scenario_id=None, initial_state=None):
        """Create one episode; Brax wrappers provide vectorization and fresh reset."""
        key, scene_key, delay_key = jax.random.split(rng, 3)
        if self.name in ("tracking", "hovering", "racing"):
            data = self._single(self.initial(jax.random.split(key, 1)))
            scenario_id = jnp.int32(0)
            delay, _ = self.delays(delay_key, 1)
            speed = jnp.float32(0)
        else:
            scenario_id = (
                jax.random.randint(scene_key, (), 0, self.bank.num_instances)
                if scenario_id is None
                else jnp.asarray(scenario_id, jnp.int32)
            )
            bank = self.bank.select(scenario_id[None])
            data = self._single(self.initial_state(bank, key))
            delay = self.delays(delay_key, 1)
            speed_range = self.command_distribution.get("speed_range_mps", (0.0, 4.0))
            speed = jnp.float32(self.conditions.get("commanded_speed", speed_range[-1]))
        if initial_state is not None:
            data = data.replace(
                pos=jnp.asarray(initial_state["position"]),
                vel=jnp.asarray(initial_state["velocity"]),
            )
            if "quaternion" in initial_state:
                data = data.replace(
                    rotation=Rotation.from_quat(
                        jnp.asarray(initial_state["quaternion"])
                    ).as_matrix()
                )
        info = dict(
            scenario_id=scenario_id,
            clock=jnp.float32(0),
            speed=speed,
            previous_command=jnp.zeros(3),
            delay_ticks=delay[0],
            outcome=jnp.int32(0),
            gates_passed=jnp.int32(0),
            terminated=jnp.float32(0),
        )
        zero = jnp.float32(0)
        metrics = {
            name: zero
            for name in (
                "tracking_error",
                "squared_error",
                "goal_distance",
                "clearance",
                "arrived",
                "collision",
                "out_of_bounds",
                "numerical_failure",
                "failure",
                "gates_passed",
            )
        }
        return State(data, self._public_observation(data, info), zero, zero, metrics, info)

    def advance(self, env, state, physical, commands):
        """Execute the existing first-event kernel through the typed dynamics API."""
        if commands is not None:
            raise ValueError(
                "This execution already owns transport delay; do not add another schedule"
            )
        control = self.controller.apply(state.pipeline_state, physical)
        command = self.controller.input_values(control)[None]
        data, info = self._batch(state.pipeline_state), state.info
        args = (
            data,
            command,
            info["previous_command"][None],
            info["delay_ticks"][None],
            info["clock"][None],
            info["outcome"][None],
        )
        if self.name in ("tracking", "hovering", "racing"):
            from drone_playground.environments.tasks.tracking.events import advance_checked

            data, clock, outcome, gates, clearance = advance_checked(
                self, *args, info["gates_passed"][None]
            )
        else:
            bank = self.bank.select(info["scenario_id"][None])
            data, clock, outcome, clearance = self.advance_checked(bank, *args)
            gates = info["gates_passed"][None]
        return self._single(data), dict(
            clock=clock[0],
            outcome=outcome[0],
            clearance=clearance[0],
            gates_passed=gates[0],
            previous_command=command[0],
        )

    def finish(self, env, state, data, action, physical, evidence):
        """Use the original point-mass evaluation reward and shared event codes."""
        del action, physical
        info = {
            **state.info,
            **{
                name: evidence[name]
                for name in ("clock", "outcome", "gates_passed", "previous_command")
            },
        }
        outcome = evidence["outcome"]
        if self.name in ("tracking", "hovering", "racing"):
            target, _ = self.reference(info["clock"])
            error = jnp.linalg.norm(data.pos - target)
            reward = -jnp.square(error)
        else:
            goal = self.bank.goal[info["scenario_id"]]
            error = jnp.linalg.norm(data.pos - goal)
            reward = jnp.linalg.norm(state.pipeline_state.pos - goal) - error
        metrics = dict(
            state.metrics,
            tracking_error=error,
            squared_error=error**2,
            goal_distance=error,
            clearance=evidence["clearance"],
            arrived=(outcome == 1).astype(jnp.float32),
            collision=(outcome == 2).astype(jnp.float32),
            out_of_bounds=(outcome == 3).astype(jnp.float32),
            numerical_failure=(outcome == 4).astype(jnp.float32),
            failure=((outcome > 1) & (outcome < 5)).astype(jnp.float32),
            gates_passed=info["gates_passed"].astype(jnp.float32),
        )
        terminated = (outcome > 0).astype(jnp.float32)
        info["terminated"] = terminated
        return state.replace(
            pipeline_state=data,
            obs=self._public_observation(data, info),
            reward=reward,
            done=terminated,
            metrics=metrics,
            info=info,
        )

    def close(self, env):
        del env

    def measurement(self, points, valid, state, time):
        """Seeded noise in physical sensor units, shared by train and frozen eval."""
        noise = self.observation_noise
        std, dropout = noise.get("sensor_std_m", 0.0), noise.get("sensor_dropout_probability", 0.0)
        if not std and not dropout:
            return points, valid
        from drone_playground.environments.randomization import point_measurement_noise

        keys = jax.vmap(jax.random.fold_in)(
            state.measurement_key,
            jnp.broadcast_to(
                jnp.floor(jnp.asarray(time) / self.dt).astype(jnp.int32),
                state.pos.shape[:-1],
            ),
        )
        if self.sensor.policy_value_channels == 1:
            noisy, mask = jax.vmap(
                lambda x, v, k: point_measurement_noise(x[..., None], v, k, std, dropout)
            )(points, valid, keys)
            return noisy[..., 0], mask
        noisy, mask = jax.vmap(lambda x, v, k: point_measurement_noise(x, v, k, std, dropout))(
            points, valid, keys
        )
        return noisy, mask

    def uncertain_action(self, action, state):
        noise = getattr(self, "environment_effects", {}).get("action_noise") or {}
        if not noise:
            return action
        if set(noise) - {"std_physical", "bias_physical"}:
            raise ValueError("Acceleration action uncertainty uses physical m/s^2 units")
        keys = jax.vmap(jax.random.fold_in)(
            state.measurement_key,
            jnp.floor(state.elapsed_time / self.dt).astype(jnp.int32),
        )
        draws = jax.vmap(lambda key: jax.random.normal(key, (3,)))(keys)
        return (
            action
            + draws * jnp.asarray(noise.get("std_physical", 0.0))
            + jnp.asarray(noise.get("bias_physical", 0.0))
        )
