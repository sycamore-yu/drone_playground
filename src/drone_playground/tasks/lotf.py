"""Original LOTF state tasks with injected control/dynamics, objective and observations."""

from __future__ import annotations

import copy

import jax
from lotf.envs import HoveringStateEnv, TrajTrackingStateEnv
from lotf.envs.wrappers import LogWrapper, MinMaxObservationWrapper, VecEnv

from drone_playground.dynamics.lotf import LOTFModel


class LOTFTask:
    def __init__(self, config, device="cpu", split="train"):
        self.config = copy.deepcopy(config)
        self.task = config["task"]["name"]
        self.model = LOTFModel(**config["dynamics"])
        self.drone = self.model.drone
        self.dynamics = self.model.forward
        self.freq = config["task"]["freq"]
        self.dt = 1 / self.freq
        self.duration = config["task"]["duration"]
        self.episode_length = int(round(self.duration * self.freq))
        self.reference_seed = {"train": 10000, "dev": 20000, "heldout": 30000}[split]
        self.device = jax.devices(device)[0]
        task_kwargs = {
            k: v
            for k, v in config["task"].items()
            if k not in ("name", "freq", "duration", "reference", "reference_count")
        }
        task_kwargs.update(
            quad_obj=self.model.execution, dt=self.dt, max_steps_in_episode=self.episode_length
        )
        if self.task == "lotf_hover":
            task_kwargs.update(
                reward_sharpness=config["objective"]["reward_sharpness"],
                action_penalty_weight=config["objective"]["action_penalty_weight"],
            )
            self.raw = HoveringStateEnv(**task_kwargs)
        elif self.task == "lotf_tracking":
            task_kwargs["ref_traj_name"] = config["policy"]["planning"]["name"]
            self.raw = TrajTrackingStateEnv(**task_kwargs)
        else:
            raise ValueError(f"Unknown LOTF task: {self.task}")
        # Scaling is a configured objective choice; the sole reward implementation
        # remains the corresponding upstream task function.
        scale = config["objective"].get("scale", 1.0)
        if scale != 1.0:
            original = self.raw._get_reward
            self.raw._get_reward = lambda old, new: scale * original(old, new)
        if config["observation"]["normalization"] == "min_max":
            self.env = MinMaxObservationWrapper(self.raw)
        elif config["observation"]["normalization"] == "none":
            self.env = self.raw
        else:
            raise ValueError("LOTF observation normalization must be min_max or none")
        self.training_env = VecEnv(LogWrapper(self.env))
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.low, self.high = self.raw.action_space.low, self.raw.action_space.high
        self.hover_action = self.raw.hovering_action
        self._render_model = None

    def reset(self, key):
        return self.env.reset(key, None)

    def raw_step(self, state, action, key):
        return self.env._step(state, action, None, key)

    @property
    def sim(self):
        if self._render_model is None:
            from drone_playground.runs.lotf_scene import create_replay_model

            self._render_model = create_replay_model(self)
        return self._render_model

    def goal(self, state):
        if self.task == "lotf_hover":
            return self.raw.goal
        import jax.numpy as jnp
        from lotf.objects.reference_traj_obj import TrajColumns

        index = jnp.minimum(
            state.init_ref_traj_idx + state.step_idx, self.raw.num_ref_traj_points - 1
        )
        return self.raw.ref_traj[index, TrajColumns.POS.slice]

    def close(self):
        self._render_model = None
