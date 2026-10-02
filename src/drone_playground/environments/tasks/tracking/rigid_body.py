"""Pure Crazyflow task adapter and fresh-reset wrapper for Brax.

Figure-eight equations, observations, physical actions, reward and termination
reuse learnsyslab/crazyflow at 36f584d114d9d331f0cee0fe4b9066f821c0fbfd.
The random spline construction follows LSY RandTrajEnv at
b1f5b36adb8e08e8e2adea85de790bd0e0a1d118 (MIT; see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from drone_playground.environments.observations.state import numerically_valid_observation

import crazyflow  # noqa: F401 — configure SciPy before its first import
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from crazyflow.envs import FigureEightEnv
from crazyflow.sim.data import SimData
from flax import struct

from drone_playground.actions.controllers.crazyflow import AttitudeControl
from drone_playground.actions.transition import ActionTransition
from drone_playground.dynamics.crazyflow import CrazyflowModel
from drone_playground.environments.observations.state import TrackingObservation
from drone_playground.environments.scenes.empty import EmptyScene
from drone_playground.environments.tasks.references import ReferenceGenerator
from drone_playground.environments.tasks.rewards import TrackingObjective




@struct.dataclass
class TrackingData:
    """All evolving task state; the immutable reference bank belongs to the task."""

    sim_data: SimData
    reference_id: jax.Array
    command: jax.Array = struct.field(default_factory=lambda: jnp.zeros(3))
    reference_phase_ticks: jax.Array = struct.field(
        default_factory=lambda: jnp.int32(0)
    )


class TrackingEnv(Env):
    """Single-environment interface; Brax owns the outer batch axis."""

    def __init__(
        self,
        task: str = "tracking",
        reference: str | None = None,
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
        if task not in {"tracking", "hovering"}:
            raise ValueError(f"Unknown tracking task: {task}")
        reference = reference or ("hover" if task == "hovering" else "figure8")
        if reference not in {"figure8", "random", "hover"}:
            raise ValueError(f"Unknown tracking reference: {reference}")
        if freq <= 0:
            raise ValueError("Task frequency must be positive")
        if reference_count < 1:
            raise ValueError("reference_count must be positive")
        self.model = model or CrazyflowModel(
            dynamics,
            drone or ("cf2x_L250" if reference == "figure8" else "cf21B_500"),
        )
        self.controller = controller or AttitudeControl()
        self.planner = planner or ReferenceGenerator(reference)
        self.scene = scene or EmptyScene()
        self.observer = observation or TrackingObservation()
        self.objective = objective or TrackingObjective()
        self.task, self.reference_kind, self.dynamics, self.freq = (
            task,
            reference,
            self.model.forward,
            freq,
        )
        self.numerical_guard = numerical_guard
        self.drone = self.model.drone
        self.duration = (
            duration
            if duration is not None
            else (10.0 if reference == "figure8" else 15.0)
        )
        self.episode_length = round(self.duration * freq)
        self.reference_seed = reference_seed
        self.reference = self.model.create_tracking(
            reference, self.duration, freq, device, self.scene
        )
        trajectories = self.planner.build(
            reference_seed, reference_count, self.duration, freq
        )
        trajectories = np.asarray(trajectories, dtype=np.float32)
        self.sim = self.reference.sim
        self.sim.reset()
        self.default = self.sim.default_data
        self.reset_fn = self.sim.build_reset_fn()
        self.substeps = self.reference.n_substeps
        self.trajectories = jax.device_put(
            jnp.asarray(trajectories), self.sim.device
        )
        self.offsets = jnp.asarray(
            np.arange(self.observer.n_samples) * freq * self.observer.interval,
            dtype=jnp.int32,
        )
        self.low = jnp.asarray(self.reference.single_action_space.low)
        self.high = jnp.asarray(self.reference.single_action_space.high)
        self.controller.bind(self.low, self.high)
        self.low, self.high = self.controller.low, self.controller.high
        self.transition = ActionTransition(
            self.controller.apply, self.model.advance, self.substeps
        )
        hover = jnp.array(
            [0.0, 0.0, 0.0, float(self.default.params.mass[0]) * 9.81]
        )
        if hasattr(self.controller, "hover"):
            hover = self.controller.hover(self.default)
        self.hover_action = 2 * (hover - self.low) / (self.high - self.low) - 1

    @property
    def observation_size(self) -> int:
        return self.observer.size

    @property
    def action_size(self) -> int:
        return 4

    @property
    def backend(self) -> str:
        return type(self.model).__name__

    @property
    def dt(self) -> float:
        return 1.0 / self.freq

    def index(self, data: SimData) -> jax.Array:
        # This includes the upstream reset observation's index of -1.
        return data.core.steps[0, 0] // self.substeps - 1

    def observation(self, data: TrackingData) -> jax.Array:
        indices = (
            self.index(data.sim_data)
            + data.reference_phase_ticks
            + self.offsets
        )
        if self.reference_kind == "figure8":
            indices = indices % self.episode_length
        else:
            indices = jnp.clip(indices, 0, self.episode_length - 1)
        refs = self.trajectories[data.reference_id, indices]
        distribution = getattr(self, "command_distribution", {})
        if distribution.get("kind") == "position":
            refs = jnp.broadcast_to(data.command, refs.shape)
        elif distribution.get("kind") == "velocity":
            refs = (
                data.sim_data.states.pos[0, 0]
                + data.command * (self.offsets * self.dt)[:, None]
            )
        else:
            refs = refs + data.command
        return self.observer(data.sim_data.states, refs)

    def reset(
        self, rng: jax.Array, reference_id: jax.Array | None = None
    ) -> State:
        if rng.dtype == jnp.uint32:
            rng = jax.random.wrap_key_data(rng)
        key, ref_key = jax.random.split(rng)
        sim_data = self.default.replace(
            core=self.default.core.replace(rng_key=key)
        )
        sim_data = self.reset_fn(sim_data, self.default)
        sim_data = self.model.randomize(sim_data, jax.random.fold_in(key, 101))
        if reference_id is None:
            reference_id = jax.random.randint(
                ref_key, (), 0, self.trajectories.shape[0]
            )
        from drone_playground.environments.randomization import sample_command

        distribution = getattr(self, "command_distribution", {})
        nominal = self.trajectories[reference_id, 0]
        command = sample_command(
            nominal, jax.random.fold_in(key, 110), distribution
        )
        offset = (
            command
            if distribution.get("kind") in ("velocity", "position")
            else command - nominal
        )
        data = TrackingData(
            sim_data, jnp.asarray(reference_id, jnp.int32), offset
        )
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
            info={
                "terminated": zero,
                "physical_parameters": self.model.physical_parameters(sim_data),
            },
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
            self.transition.step(data.sim_data, physical)
            if commands is None
            else self.transition.step_schedule(data.sim_data, commands)
        )
        data = data.replace(sim_data=sim_data)
        numerical_failure = jnp.array(False)
        if self.numerical_guard:
            numerical_failure = ~numerically_valid_observation(
                self.observation(data)
            )
            # A diverged integrator state is a failed trial. Retain the last valid
            # physical pose for terminal recording, and reset before another action.
            # Healthy transitions pass through unchanged; no physical limits are relaxed.
            states = jax.tree.map(
                lambda new, old: jnp.where(
                    numerical_failure, jax.lax.stop_gradient(old), new
                ),
                sim_data.states,
                state.pipeline_state.sim_data.states,
            )
            sim_data = sim_data.replace(states=states)
            data = data.replace(sim_data=sim_data)
        index = jnp.clip(
            self.index(sim_data) + data.reference_phase_ticks,
            0,
            self.episode_length - 1,
        )
        goal = self.trajectories[data.reference_id, index]
        if getattr(self, "command_distribution", {}).get("kind") == "position":
            goal = data.command
        elif (
            getattr(self, "command_distribution", {}).get("kind") == "velocity"
        ):
            goal = (
                state.pipeline_state.sim_data.states.pos[0, 0]
                + data.command * self.dt
            )
        else:
            goal = goal + data.command
        pos = sim_data.states.pos
        if self.reference_kind == "figure8":
            terminated = FigureEightEnv._terminated(pos)[0]
        else:
            terminated = jnp.any(
                (pos[0, 0] < jnp.array([-4.0, -4.0, 0.0]))
                | (pos[0, 0] > jnp.array([4.0, 4.0, 4.0]))
            )
        obs = self.observation(data)
        # Preserve physical termination and explicitly surface numerical invalidity.
        terminated = (
            terminated | ~jnp.all(jnp.isfinite(obs)) | numerical_failure
        )
        reward = self.objective(terminated, pos[0, 0], goal)
        error = jnp.linalg.norm(pos[0, 0] - goal)
        metrics = {
            **state.metrics,
            "tracking_error": error,
            "squared_error": error**2,
            "action_saturation": jnp.mean(
                (jnp.abs(action) >= 0.99).astype(jnp.float32)
            ),
            "physical_thrust": physical[0]
            if self.controller.input_kind == "thrust_bodyrates"
            else physical[3],
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
