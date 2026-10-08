"""LSY Level-0 race, using its native geometry/events and a named learning reward."""

from __future__ import annotations

from dataclasses import dataclass

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State

from drone_playground.environments.tasks.lsy_upstream import race_core
from drone_playground.environments.tasks.lsy_upstream.utils import gate_passed as gate_passed
from drone_playground.environments.tasks.rigid_body import RigidBodyTask


@dataclass
class RacingTask(RigidBodyTask):
    """Pinned LSY gate events and the existing reference-tracking reward."""

    name: str
    duration: float
    reference_count: int
    time_limit_kind: str
    command_distribution: dict
    observation: object
    reward: object

    def create_simulation(self, env):
        if self.name != "racing" or env.freq != 50 or self.duration != 30.0:
            raise ValueError("LSY Level0 fixes racing at 50 Hz for 30 seconds")
        self.config = cfg = env.scene.config(env.dynamics)
        core = env.scene.create_core(
            n_envs=1,
            n_drones=1,
            freq=env.freq,
            sim_config=cfg.sim,
            sensor_range=cfg.env.sensor_range,
            track=cfg.env.track,
            control_mode="attitude",
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=0,
            max_episode_steps=env.episode_length,
            device=env.device,
        )
        try:
            env.dynamics.bind(core.sim, control_mode=env.controller.native_mode)
            core.sim.spec.compiler.texturedir = str(core.gate_spec_path.parent)
            core.settings = core.settings.replace(autoreset=False)
            return core, race_core.build_action_space("attitude", cfg.sim.drone)
        except BaseException:
            core.sim.close()
            raise

    def bind(self, env):
        env.core = env.simulation
        env.config = self.config
        cfg = self.config
        env.default = env.core.data.replace(sim_data=env.sim.default_data)
        env.reset_fn = env.core.build_reset_fn()
        env.contact_fn = env.core.build_contact_check_fn()
        start = np.asarray(env.sim.default_data.states.pos[0, 0])
        count = self.reference_count if env.role == "train" else env.count
        env.trajectories = jnp.asarray(
            env.reference.build(env.reference_seed, count, env.duration, env.freq, start)
        )
        env.offsets = jnp.arange(self.observation.n_samples, dtype=jnp.int32) * int(
            env.freq * self.observation.interval
        )
        env.required_gates = len(cfg.env.track.gate_order)
        env.core.data, _ = env.reset_fn(env.default)
        env.core.data, env.sim.mjx_data = env.core._render_sync(env.core.data, env.sim.mjx_data)

    def index(self, env, data):
        return jnp.clip(data.steps[0] - 1, 0, env.episode_length - 1)

    def observe(self, env, data):
        refs = env.trajectories[
            0,
            jnp.clip(data.steps[0] + env.offsets, 0, env.episode_length - 1),
        ]
        return env.task.observation(data.sim_data.states, refs)

    def reset(self, env, rng, reference_id=None):
        if rng.dtype == jnp.uint32:
            rng = jax.random.wrap_key_data(rng)
        initial = env.default.replace(
            sim_data=env.default.sim_data.replace(
                core=env.default.sim_data.core.replace(rng_key=rng)
            )
        )
        data, _ = env.reset_fn(initial)
        data = data.replace(
            sim_data=env.dynamics.randomize(data.sim_data, jax.random.fold_in(rng, 101))
        )
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
            obs=env.observation(data),
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                "terminated": zero,
                "physical_parameters": env.dynamics.physical_parameters(data.sim_data),
            },
        )

    def controller_observation(self, env, state):
        raw = race_core.obs(state.pipeline_state)
        return {k: np.asarray(v[0, 0]) for k, v in raw.items()}

    def finish(self, env, state, data, action, physical, evidence):
        del action, evidence
        contacts = env.contact_fn(jax.tree.map(jax.lax.stop_gradient, data))
        data = race_core._update_disabled_drones(data, contacts)
        # Original core warps dead drones to -1 to avoid multi-drone interference.
        # Our single-drone boundary retains the real terminal pose for replay/bootstrap.
        data = race_core._update_visited_objects(data)
        data = race_core._update_target_gates(data)
        data = race_core._mark_drones_for_reset(data)
        data = data.replace(steps=data.steps + 1)
        obs = env.observation(data)
        failed = data.disabled_drones[0, 0] | ~jnp.isfinite(obs).all()
        success = (data.n_gates_passed[0, 0] >= env.required_gates) & ~failed
        goal = env.trajectories[0, env.index(data)]
        error = jnp.linalg.norm(data.sim_data.states.pos[0, 0] - goal)
        reward = env.task.reward(failed, data.sim_data.states.pos[0, 0], goal)
        normalized = (physical - env.low) / (env.high - env.low) * 2 - 1
        metrics = {
            **state.metrics,
            "tracking_error": error,
            "squared_error": error**2,
            "action_saturation": jnp.mean((jnp.abs(normalized) >= 0.99).astype(jnp.float32)),
            "physical_thrust": physical[0] if env.controller.input_kind == "rates" else physical[3],
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
