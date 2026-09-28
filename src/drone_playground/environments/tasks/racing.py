"""LSY Level-0 race, using its native geometry/events and a named learning reward."""

from __future__ import annotations

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State

from drone_playground.environments.observations import TrackingObservation
from drone_playground.environments.scenes import LSYScene
from drone_playground.execution.controllers.crazyflow import AttitudeControl
from drone_playground.execution.transition import ExecutionTransition
from drone_playground.learning.objectives import TrackingObjective
from drone_playground.methods.planners.reference import TrajectoryPlan
from drone_playground.models.crazyflow import CrazyflowModel

from .lsy_upstream import race_core
from .lsy_upstream.utils import gate_passed as gate_passed


class RacingEnv(Env):
    """One race world; the existing Brax wrapper owns vectorization and reset.

    Gate crossing, collision masks, action/dynamics disturbances and bounds come
    from pinned LSY. Since this is one drone per world, terminal poses are retained
    rather than warped underground; completed worlds reset at the Brax boundary.
    Learning uses reference-tracking-v1, while success uses native gate events.
    """

    def __init__(
        self,
        task="racing",
        dynamics="first_principles",
        drone=None,
        freq=50,
        device="cpu",
        reference_seed=10000,
        reference_count=256,
        model=None,
        controller=None,
        planner=None,
        scene=None,
        observation=None,
        objective=None,
    ):
        if task != "racing" or freq != 50:
            raise ValueError("LSY Level0 protocol fixes task=racing and 50 Hz control")
        self.model = model or CrazyflowModel(dynamics, drone or "cf21B_500")
        self.controller = controller or AttitudeControl()
        self.planner = planner or TrajectoryPlan("lsy_course")
        self.scene = scene or LSYScene()
        self.observer = observation or TrackingObservation()
        self.objective = objective or TrackingObjective()
        cfg = self.scene.config(self.model)
        self.task, self.dynamics, self.drone, self.freq = (
            "racing",
            self.model.forward,
            self.model.drone,
            freq,
        )
        self.config = cfg
        self.reference_seed = reference_seed
        self.duration, self.episode_length = 30.0, 1500
        self.core = race_core.RaceCoreEnv(
            n_envs=1,
            n_drones=1,
            freq=freq,
            sim_config=cfg.sim,
            sensor_range=cfg.env.sensor_range,
            track=cfg.env.track,
            control_mode="attitude",
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=0,
            max_episode_steps=self.episode_length,
            device=device,
        )
        self.sim = self.core.sim
        self.model.bind(self.sim)
        # MjSpec attachment retains source resources at compile time, but to_xml()
        # emits their basenames. Preserve the known source directory for export.
        self.sim.spec.compiler.texturedir = str(self.core.gate_spec_path.parent)
        self.core.settings = self.core.settings.replace(autoreset=False)
        self.default = self.core.data.replace(sim_data=self.sim.default_data)
        self.reset_fn = self.core.build_reset_fn()
        self.apply_action_fn = self.core.build_apply_action_fn()
        self.contact_fn = self.core.build_contact_check_fn()
        self.substeps = self.sim.freq // freq
        start = np.asarray(self.sim.default_data.states.pos[0, 0])
        self.trajectories = jnp.asarray(
            self.planner.build(reference_seed, reference_count, self.duration, freq, start)
        )
        self.offsets = jnp.arange(self.observer.n_samples, dtype=jnp.int32) * int(
            freq * self.observer.interval
        )
        action_space = race_core.build_action_space("attitude", self.drone)
        self.low, self.high = jnp.asarray(action_space.low), jnp.asarray(action_space.high)
        self.controller.bind(self.low, self.high)
        self.execution = ExecutionTransition(
            lambda data, physical: self.apply_action_fn(physical[None, None], data),
            lambda data, steps: data.replace(sim_data=self.model.advance(data.sim_data, steps)),
            self.substeps,
        )
        hover = jnp.array(
            [
                0.0,
                0.0,
                0.0,
                float(np.asarray(self.sim.default_data.params.mass).reshape(-1)[0]) * 9.81,
            ]
        )
        self.hover_action = (hover - self.low) / (self.high - self.low) * 2 - 1
        self.required_gates = len(cfg.env.track.gate_order)
        # Initialize scene mocap buffers for export; static course stays identical.
        self.core.data, _ = self.reset_fn(self.default)
        self.core.data, self.sim.mjx_data = self.core._render_sync(
            self.core.data, self.sim.mjx_data
        )

    @property
    def observation_size(self):
        return self.observer.size

    @property
    def action_size(self):
        return 4

    @property
    def backend(self):
        return "crazyflow"

    @property
    def dt(self):
        return 1 / self.freq

    def index(self, data):
        return jnp.clip(data.steps[0] - 1, 0, self.episode_length - 1)

    def observation(self, data):
        refs = self.trajectories[
            0, jnp.clip(data.steps[0] + self.offsets, 0, self.episode_length - 1)
        ]
        return self.observer(data.sim_data.states, refs)

    def reset(self, rng, reference_id=None):
        if rng.dtype == jnp.uint32:
            rng = jax.random.wrap_key_data(rng)
        initial = self.default.replace(
            sim_data=self.default.sim_data.replace(
                core=self.default.sim_data.core.replace(rng_key=rng)
            )
        )
        data, _ = self.reset_fn(initial)
        zero = jnp.float32(0)
        metrics = {
            name: zero
            for name in (
                "tracking_error",
                "squared_error",
                "action_saturation",
                "physical_thrust",
                "failure",
                "collision",
                "success",
                "gates_passed",
                "native_reward",
                "reference_x",
                "reference_y",
                "reference_z",
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

    def physical_action(self, action):
        return self.controller.physical_action(action)

    def step_physical(self, state, physical):
        data = self.execution.step(state.pipeline_state, physical)
        contacts = self.contact_fn(jax.tree.map(jax.lax.stop_gradient, data))
        data = race_core._update_disabled_drones(data, contacts)
        # Original core warps dead drones to -1 to avoid multi-drone interference.
        # Our single-drone boundary retains the real terminal pose for replay/bootstrap.
        data = race_core._update_visited_objects(data)
        data = race_core._update_target_gates(data)
        data = race_core._mark_drones_for_reset(data)
        data = data.replace(steps=data.steps + 1)
        obs = self.observation(data)
        failed = data.disabled_drones[0, 0] | ~jnp.isfinite(obs).all()
        success = (data.n_gates_passed[0, 0] >= self.required_gates) & ~failed
        goal = self.trajectories[0, self.index(data)]
        error = jnp.linalg.norm(data.sim_data.states.pos[0, 0] - goal)
        reward = self.objective(failed, data.sim_data.states.pos[0, 0], goal)
        normalized = (physical - self.low) / (self.high - self.low) * 2 - 1
        metrics = {
            **state.metrics,
            "tracking_error": error,
            "squared_error": error**2,
            "action_saturation": jnp.mean((jnp.abs(normalized) >= 0.99).astype(jnp.float32)),
            "physical_thrust": physical[3],
            "failure": failed.astype(jnp.float32),
            "collision": contacts[0, 0].astype(jnp.float32),
            "success": success.astype(jnp.float32),
            "gates_passed": data.n_gates_passed[0, 0].astype(jnp.float32),
            "native_reward": race_core.reward(data)[0, 0],
            "reference_x": goal[0],
            "reference_y": goal[1],
            "reference_z": goal[2],
        }
        terminal = (failed | success).astype(jnp.float32)
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=terminal,
            metrics=metrics,
            info={**state.info, "terminated": terminal},
        )

    def step(self, state, action):
        return self.step_physical(state, self.physical_action(action))

    def controller_observation(self, state):
        raw = race_core.obs(state.pipeline_state)
        return {k: np.asarray(v[0, 0]) for k, v in raw.items()}

    def close(self):
        self.core.close()
