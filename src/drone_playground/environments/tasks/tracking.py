"""Pure Crazyflow task adapter and fresh-reset wrapper for Brax.

Figure-eight equations, observations, physical actions, reward and termination
reuse learnsyslab/crazyflow at 36f584d114d9d331f0cee0fe4b9066f821c0fbfd.
The random spline construction follows LSY RandTrajEnv at
b1f5b36adb8e08e8e2adea85de790bd0e0a1d118 (MIT; see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import crazyflow  # noqa: F401 — configure SciPy before its first import
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from crazyflow.envs import FigureEightEnv
from crazyflow.sim.data import SimData
from flax import struct

from drone_playground.environments.observations import TrackingObservation
from drone_playground.environments.scenes import EmptyScene
from drone_playground.execution.controllers.crazyflow import AttitudeControl
from drone_playground.execution.transition import ExecutionTransition
from drone_playground.learning.objectives import TrackingObjective
from drone_playground.methods.planners.reference import TrajectoryPlan
from drone_playground.models.crazyflow import CrazyflowModel


def numerically_valid_observation(observation: jax.Array) -> jax.Array:
    """Require representable finite second moments, not just finite scalar entries."""
    return jnp.all(jnp.isfinite(observation)) & jnp.isfinite(jnp.sum(observation**2))


@struct.dataclass
class TrackingData:
    """All evolving task state; the immutable reference bank belongs to the task."""

    sim_data: SimData
    reference_id: jax.Array


class TrackingEnv(Env):
    """Single-environment interface; Brax owns the outer batch axis."""

    def __init__(
        self,
        task: str = "figure8",
        dynamics: str = "so_rpy",
        drone: str | None = None,
        freq: int = 50,
        device: str = "cpu",
        reference_seed: int = 10000,
        reference_count: int = 256,
        numerical_guard: bool = False,
        model=None,
        controller=None,
        planner=None,
        scene=None,
        observation=None,
        objective=None,
        duration: float | None = None,
    ):
        if task not in {"figure8", "random", "hovering"}:
            raise ValueError(f"Unknown tracking task: {task}")
        if freq <= 0 or 500 % freq:
            raise ValueError("Task frequency must be a positive divisor of 500 Hz")
        if reference_count < 1:
            raise ValueError("reference_count must be positive")
        self.model = model or CrazyflowModel(
            dynamics, drone or ("cf2x_L250" if task == "figure8" else "cf21B_500")
        )
        self.controller = controller or AttitudeControl()
        self.planner = planner or TrajectoryPlan(task)
        self.scene = scene or EmptyScene()
        self.observer = observation or TrackingObservation()
        self.objective = objective or TrackingObjective()
        self.task, self.dynamics, self.freq = task, self.model.forward, freq
        self.numerical_guard = numerical_guard
        self.drone = self.model.drone
        self.duration = duration if duration is not None else (10.0 if task == "figure8" else 15.0)
        self.episode_length = round(self.duration * freq)
        self.reference_seed = reference_seed
        self.reference = self.model.create_tracking(task, self.duration, freq, device, self.scene)
        trajectories = self.planner.build(reference_seed, reference_count, self.duration, freq)
        trajectories = np.asarray(trajectories, dtype=np.float32)
        self.sim = self.reference.sim
        self.sim.reset()
        self.default = self.sim.default_data
        self.reset_fn = self.sim.build_reset_fn()
        self.substeps = self.reference.n_substeps
        self.trajectories = jax.device_put(jnp.asarray(trajectories), self.sim.device)
        self.offsets = jnp.asarray(
            np.arange(self.observer.n_samples) * freq * self.observer.interval, dtype=jnp.int32
        )
        self.low = jnp.asarray(self.reference.single_action_space.low)
        self.high = jnp.asarray(self.reference.single_action_space.high)
        self.controller.bind(self.low, self.high)
        self.execution = ExecutionTransition(
            self.controller.apply, self.model.advance, self.substeps
        )
        hover = jnp.array([0.0, 0.0, 0.0, float(self.default.params.mass[0]) * 9.81])
        self.hover_action = 2 * (hover - self.low) / (self.high - self.low) - 1

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

    def index(self, data: SimData) -> jax.Array:
        # This includes the upstream reset observation's index of -1.
        return data.core.steps[0, 0] // self.substeps - 1

    def observation(self, data: TrackingData) -> jax.Array:
        indices = self.index(data.sim_data) + self.offsets
        if self.task == "figure8":
            indices = indices % self.episode_length
        else:
            indices = jnp.clip(indices, 0, self.episode_length - 1)
        refs = self.trajectories[data.reference_id, indices]
        return self.observer(data.sim_data.states, refs)

    def reset(self, rng: jax.Array, reference_id: jax.Array | None = None) -> State:
        if rng.dtype == jnp.uint32:
            rng = jax.random.wrap_key_data(rng)
        key, ref_key = jax.random.split(rng)
        sim_data = self.default.replace(core=self.default.core.replace(rng_key=key))
        sim_data = self.reset_fn(sim_data, self.default)
        if reference_id is None:
            reference_id = jax.random.randint(ref_key, (), 0, self.trajectories.shape[0])
        data = TrackingData(sim_data, jnp.asarray(reference_id, jnp.int32))
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
        if self.numerical_guard:
            metrics["numerical_failure"] = zero
        return State(
            pipeline_state=data,
            obs=self.observation(data),
            reward=zero,
            done=zero,
            metrics=metrics,
            info={"terminated": zero},
        )

    def physical_action(self, action: jax.Array) -> jax.Array:
        return self.controller.physical_action(action)

    def step(self, state: State, action: jax.Array) -> State:
        return self._transition(state, action)

    def step_schedule(self, state, commands):
        action = 2 * (commands[-1] - self.low) / (self.high - self.low) - 1
        return self._transition(state, action, commands)

    def _transition(self, state, action, commands=None):
        data = state.pipeline_state
        physical = self.physical_action(action)
        sim_data = (
            self.execution.step(data.sim_data, physical)
            if commands is None
            else self.execution.step_schedule(data.sim_data, commands)
        )
        data = data.replace(sim_data=sim_data)
        numerical_failure = jnp.array(False)
        if self.numerical_guard:
            numerical_failure = ~numerically_valid_observation(self.observation(data))
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
        index = jnp.clip(self.index(sim_data), 0, self.episode_length - 1)
        goal = self.trajectories[data.reference_id, index]
        pos = sim_data.states.pos
        if self.task == "figure8":
            terminated = FigureEightEnv._terminated(pos)[0]
        else:
            terminated = jnp.any(
                (pos[0, 0] < jnp.array([-4.0, -4.0, 0.0]))
                | (pos[0, 0] > jnp.array([4.0, 4.0, 4.0]))
            )
        obs = self.observation(data)
        # Preserve physical termination and explicitly surface numerical invalidity.
        terminated = terminated | ~jnp.all(jnp.isfinite(obs)) | numerical_failure
        reward = self.objective(terminated, pos[0, 0], goal)
        error = jnp.linalg.norm(pos[0, 0] - goal)
        metrics = {
            **state.metrics,
            "tracking_error": error,
            "squared_error": error**2,
            "action_saturation": jnp.mean((jnp.abs(action) >= 0.99).astype(jnp.float32)),
            "physical_thrust": physical[3],
            "failure": terminated.astype(jnp.float32),
        }
        if self.numerical_guard:
            metrics["numerical_failure"] = numerical_failure.astype(jnp.float32)
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=terminated.astype(jnp.float32),
            metrics=metrics,
            info={**state.info, "terminated": terminated.astype(jnp.float32)},
        )

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
