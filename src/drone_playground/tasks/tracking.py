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
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
from crazyflow.envs import FigureEightEnv
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim import functional as F
from crazyflow.sim.data import SimData
from flax import struct
from scipy.interpolate import CubicSpline


def random_trajectory(seed: int, duration: float = 15.0, freq: int = 50) -> np.ndarray:
    """Build the author's ten-knot trajectory using a private NumPy RNG."""
    takeoff = np.array([-1.5, 1.0, 0.07])
    waypoints = np.random.RandomState(seed).uniform(-1, 1, (10, 3))
    waypoints = waypoints * [1.2, 1.2, 0.5] + 0.3 * takeoff + [0, 0, 0.7]
    waypoints[:3] = [[-1.5, 1.0, 0.07], [-1.0, 0.55, 0.4], [0.3, 0.35, 0.7]]
    spline = CubicSpline(
        np.linspace(0, duration, 10),
        waypoints,
        bc_type=((1, np.array([0.0, 0.0, 0.4])), "not-a-knot"),
    )
    return spline(np.linspace(0, duration, int(np.ceil(duration * freq)))).astype(np.float32)


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
    ):
        if task not in {"figure8", "random"}:
            raise ValueError(f"Unknown tracking task: {task}")
        if freq <= 0 or 500 % freq:
            raise ValueError("Task frequency must be a positive divisor of 500 Hz")
        if reference_count < 1:
            raise ValueError("reference_count must be positive")
        self.task, self.dynamics, self.freq = task, dynamics, freq
        self.drone = drone or ("cf2x_L250" if task == "figure8" else "cf21B_500")
        self.duration = 10.0 if task == "figure8" else 15.0
        self.episode_length = round(self.duration * freq)
        self.reference_seed = reference_seed
        if task == "figure8":
            self.reference = FigureEightEnv(
                num_envs=1, freq=freq, dynamics=dynamics, drone=self.drone, device=device
            )
            trajectories = np.asarray(self.reference.trajectory, dtype=np.float32)[None]
        else:
            # The source reset hook predates the current three-argument pipeline.
            # Adapt only that signature; preserve the author's rotor initialization.
            def random_reset(data, default, mask):
                del default
                from crazyflow.utils import leaf_replace

                speed = 10000.0 if dynamics == "first_principles" else 0.05
                rotor = jnp.full_like(data.states.rotor_vel, speed)
                return data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor))

            self.reference = DroneEnv(
                num_envs=1,
                freq=freq,
                max_episode_time=self.duration,
                dynamics=dynamics,
                drone=self.drone,
                device=device,
                reset_randomization=random_reset,
            )
            sim = self.reference.sim
            sim.data = sim.data.replace(
                states=sim.data.states.replace(
                    pos=sim.data.states.pos.at[0, 0].set(jnp.array([-1.5, 1.0, 0.07]))
                )
            )
            sim.build_default_data()
            trajectories = np.stack(
                [
                    random_trajectory(reference_seed + i, self.duration, freq)
                    for i in range(reference_count)
                ]
            )
        self.sim = self.reference.sim
        self.sim.reset()
        self.default = self.sim.default_data
        self.step_fn, self.reset_fn = self.sim.build_step_fn(), self.sim.build_reset_fn()
        self.substeps = self.reference.n_substeps
        self.trajectories = jax.device_put(jnp.asarray(trajectories), self.sim.device)
        self.offsets = jnp.asarray(np.arange(10) * freq * 0.1, dtype=jnp.int32)
        self.low = jnp.asarray(self.reference.single_action_space.low)
        self.high = jnp.asarray(self.reference.single_action_space.high)
        hover = jnp.array([0.0, 0.0, 0.0, float(self.default.params.mass[0]) * 9.81])
        self.hover_action = 2 * (hover - self.low) / (self.high - self.low) - 1

    @property
    def observation_size(self) -> int:
        return 43

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
        state = data.sim_data.states
        indices = self.index(data.sim_data) + self.offsets
        if self.task == "figure8":
            indices = indices % self.episode_length
        else:
            indices = jnp.clip(indices, 0, self.episode_length - 1)
        refs = self.trajectories[data.reference_id, indices]
        local = (refs - state.pos[0, 0]).reshape(-1)
        return jnp.concatenate(
            (state.pos[0, 0], state.quat[0, 0], state.vel[0, 0], state.ang_vel[0, 0], local)
        )

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
        return State(
            pipeline_state=data,
            obs=self.observation(data),
            reward=zero,
            done=zero,
            metrics=metrics,
            info={"terminated": zero},
        )

    def physical_action(self, action: jax.Array) -> jax.Array:
        return self.low + (jnp.clip(action, -1.0, 1.0) + 1) * 0.5 * (self.high - self.low)

    def step(self, state: State, action: jax.Array) -> State:
        data = state.pipeline_state
        physical = self.physical_action(action)
        sim_data = F.attitude_control(data.sim_data, physical[None, None])
        sim_data = self.step_fn(sim_data, self.substeps)
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
        terminated = terminated | ~jnp.all(jnp.isfinite(obs))
        reward = FigureEightEnv._reward(terminated[None], pos, goal)[0]
        error = jnp.linalg.norm(pos[0, 0] - goal)
        metrics = {
            **state.metrics,
            "tracking_error": error,
            "squared_error": error**2,
            "action_saturation": jnp.mean((jnp.abs(action) >= 0.99).astype(jnp.float32)),
            "physical_thrust": physical[3],
            "failure": terminated.astype(jnp.float32),
        }
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


class FreshAutoReset(Wrapper):
    """Brax same-step reset with fresh initial states and saved terminal observations.

    EpisodeWrapper owns truncation and metrics. Unlike Brax's cached-reset wrapper,
    a completed trial consumes its own new random key. Resets disconnect gradients.
    """

    def reset(self, rng: jax.Array) -> State:
        children = jax.vmap(jax.random.split)(rng)
        state = self.env.reset(children[:, 1])
        return state.replace(
            info={
                **state.info,
                "reset_key": children[:, 0],
                "terminal_observation": state.obs,
                "time_out": jnp.zeros_like(state.done),
            }
        )

    def step(self, state: State, action: jax.Array) -> State:
        info = {**state.info, "steps": jnp.where(state.done, 0, state.info["steps"])}
        state = state.replace(done=jnp.zeros_like(state.done), info=info)
        terminal = self.env.step(state, action)
        done = terminal.done.astype(bool)
        keys = jax.vmap(jax.random.split)(terminal.info["reset_key"])

        def choose(fresh, current):
            mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
            return jnp.where(mask, jax.lax.stop_gradient(fresh), current)

        def reset_done(_):
            fresh = self.env.reset(keys[:, 1])
            data = jax.tree.map(choose, fresh.pipeline_state, terminal.pipeline_state)
            obs = choose(fresh.obs, terminal.obs)
            return data, obs

        data, obs = jax.lax.cond(
            jnp.any(done),
            reset_done,
            lambda _: (terminal.pipeline_state, terminal.obs),
            operand=None,
        )
        info = {
            **terminal.info,
            "terminal_observation": terminal.obs,
            "time_out": terminal.info["truncation"],
            "reset_key": jnp.where(done[:, None], keys[:, 0], terminal.info["reset_key"]),
        }
        return terminal.replace(pipeline_state=data, obs=obs, info=info)


def wrap_for_training(
    env: TrackingEnv, episode_length: int, action_repeat: int = 1, randomization_fn=None
) -> FreshAutoReset:
    """Compose native Brax batching/statistics with the task's fresh-reset contract."""
    if randomization_fn is not None:
        raise ValueError("Randomization belongs to this Crazyflow task's reset, not a Brax System")
    if action_repeat != 1:
        raise ValueError("Action repeat is fixed at one; task frequency owns physical substeps")
    return FreshAutoReset(EpisodeWrapper(VmapWrapper(env), episode_length, action_repeat))
