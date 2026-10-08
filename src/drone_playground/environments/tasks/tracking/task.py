"""Pure Crazyflow task adapter and fresh-reset wrapper for Brax.

Figure-eight equations, observations, physical actions, reward and termination
reuse learnsyslab/crazyflow at 36f584d114d9d331f0cee0fe4b9066f821c0fbfd.
The random spline construction follows LSY RandTrajEnv at
b1f5b36adb8e08e8e2adea85de790bd0e0a1d118 (MIT; see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import crazyflow  # noqa: F401 — configure SciPy before its first import
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State
from crazyflow.envs import FigureEightEnv
from crazyflow.sim.data import SimData
from flax import struct

from drone_playground.environments.observations.state import (
    numerically_valid_observation,
)
from drone_playground.environments.tasks.rigid_body import RigidBodyTask


@struct.dataclass
class TrackingData:
    """All evolving task state; the immutable reference bank belongs to the task."""

    sim_data: SimData
    reference_id: jax.Array
    command: jax.Array = struct.field(default_factory=lambda: jnp.zeros(3))
    reference_phase_ticks: jax.Array = struct.field(default_factory=lambda: jnp.int32(0))


@dataclass
class TrackingTask(RigidBodyTask):
    """Reference-tracking events, observation and reward, independent of the learner."""

    name: str
    duration: float
    reference_count: int
    numerical_guard: bool
    time_limit_kind: str
    command_distribution: dict
    observation: object
    reward: object

    def create_simulation(self, env):
        from drone_playground.environments.initialization import initialize_rigid_body

        return initialize_rigid_body(
            env,
            start=env.scene.takeoff,
            figure_eight=env.reference.name == "figure8",
            reset=True,
        )

    def bind(self, env):
        if self.name not in ("hovering", "tracking"):
            raise ValueError("TrackingTask requires a hovering or tracking task name")
        env.reference_kind = env.reference.name
        env.numerical_guard = self.numerical_guard
        count = self.reference_count if env.role == "train" else env.count
        trajectories = env.reference.build(env.reference_seed, count, env.duration, env.freq)
        env.trajectories = jax.device_put(
            jnp.asarray(np.asarray(trajectories, np.float32)), env.sim.device
        )
        env.offsets = jnp.asarray(
            np.arange(self.observation.n_samples) * env.freq * self.observation.interval,
            dtype=jnp.int32,
        )

    def index(self, env, data: SimData) -> jax.Array:
        # This includes the upstream reset observation's index of -1.
        return data.core.steps[0, 0] // env.substeps - 1

    def observe(self, env, data: TrackingData) -> jax.Array:
        indices = env.index(data.sim_data) + data.reference_phase_ticks + env.offsets
        if env.reference_kind == "figure8":
            indices = indices % env.episode_length
        else:
            indices = jnp.clip(indices, 0, env.episode_length - 1)
        refs = env.trajectories[data.reference_id, indices]
        distribution = getattr(env, "command_distribution", {})
        if distribution.get("kind") == "position":
            refs = jnp.broadcast_to(data.command, refs.shape)
        elif distribution.get("kind") == "velocity":
            refs = data.sim_data.states.pos[0, 0] + data.command * (env.offsets * env.dt)[:, None]
        else:
            refs = refs + data.command
        return env.task.observation(data.sim_data.states, refs)

    def reset(self, env, rng: jax.Array, reference_id: jax.Array | None = None) -> State:
        if rng.dtype == jnp.uint32:
            rng = jax.random.wrap_key_data(rng)
        key, ref_key = jax.random.split(rng)
        sim_data = env.default.replace(core=env.default.core.replace(rng_key=key))
        sim_data = env.reset_fn(sim_data, env.default)
        sim_data = env.dynamics.randomize(sim_data, jax.random.fold_in(key, 101))
        if reference_id is None:
            reference_id = jax.random.randint(ref_key, (), 0, env.trajectories.shape[0])
        from drone_playground.environments.randomization import sample_command

        distribution = getattr(env, "command_distribution", {})
        nominal = env.trajectories[reference_id, 0]
        command = sample_command(nominal, jax.random.fold_in(key, 110), distribution)
        offset = (
            command if distribution.get("kind") in ("velocity", "position") else command - nominal
        )
        data = TrackingData(sim_data, jnp.asarray(reference_id, jnp.int32), offset)
        zero = jnp.float32(0)
        metrics = {
            k: zero
            for k in (
                "tracking_error",
                "squared_error",
                "action_saturation",
                "physical_thrust",
                "failure",
            )
        }
        if env.numerical_guard:
            metrics["numerical_failure"] = zero
        return State(
            pipeline_state=data,
            obs=env.observation(data),
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                "terminated": zero,
                "physical_parameters": env.dynamics.physical_parameters(sim_data),
            },
        )

    def finish(self, env, state, data, action, physical, evidence):
        del evidence
        sim_data = data.sim_data
        numerical_failure = jnp.array(False)
        if env.numerical_guard:
            numerical_failure = ~numerically_valid_observation(env.observation(data))
            # A diverged integrator state is a failed trial. Retain the last valid
            # physical pose for terminal recording, and reset before another action.
            # Healthy transitions pass through unchanged; no physical limits are relaxed.
            states = jax.tree.map(
                lambda new, old: jnp.where(numerical_failure, jax.lax.stop_gradient(old), new),
                sim_data.states,
                state.pipeline_state.sim_data.states,
            )
            sim_data = sim_data.replace(states=states)
            data = data.replace(sim_data=sim_data)
        index = jnp.clip(
            env.index(sim_data) + data.reference_phase_ticks,
            0,
            env.episode_length - 1,
        )
        goal = env.trajectories[data.reference_id, index]
        if getattr(env, "command_distribution", {}).get("kind") == "position":
            goal = data.command
        elif getattr(env, "command_distribution", {}).get("kind") == "velocity":
            goal = state.pipeline_state.sim_data.states.pos[0, 0] + data.command * env.dt
        else:
            goal = goal + data.command
        pos = sim_data.states.pos
        if env.reference_kind == "figure8":
            terminated = FigureEightEnv._terminated(pos)[0]
        else:
            terminated = jnp.any(
                (pos[0, 0] < jnp.array([-4.0, -4.0, 0.0]))
                | (pos[0, 0] > jnp.array([4.0, 4.0, 4.0]))
            )
        obs = env.observation(data)
        # Preserve physical termination and explicitly surface numerical invalidity.
        terminated = terminated | ~jnp.all(jnp.isfinite(obs)) | numerical_failure
        reward = env.task.reward(terminated, pos[0, 0], goal)
        error = jnp.linalg.norm(pos[0, 0] - goal)
        metrics = {
            **state.metrics,
            "tracking_error": error,
            "squared_error": error**2,
            "action_saturation": jnp.mean((jnp.abs(action) >= 0.99).astype(jnp.float32)),
            "physical_thrust": physical[0] if env.controller.input_kind == "rates" else physical[3],
            "failure": terminated.astype(jnp.float32),
        }
        if env.numerical_guard:
            metrics["numerical_failure"] = numerical_failure.astype(jnp.float32)
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=terminated.astype(jnp.float32),
            metrics=metrics,
            info={**state.info, "terminated": terminated.astype(jnp.float32)},
        )
