"""One recoverable update interface for batched PPO, APG and SHAC."""

import inspect
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from drone_playground.learning.losses import liu_loss, temporal_gradient_decay, zhang_loss
from drone_playground.learning.objective import EpisodeObjective, ObjectiveState
from drone_playground.simulation.methods import PolicyMethod
from drone_playground.simulation.networks import Actor, Critic


@struct.dataclass
class TrainingState:
    """All mutable learning state, including environment and objective history."""

    params: Any
    critic_params: Any
    target_critic_params: Any
    optimizer_state: Any
    rng: jax.Array
    env_state: Any
    recurrent_memory: Any
    objective_state: ObjectiveState
    updates: int
    interactions: int
    log_std: jax.Array


_DEFAULTS = {
    "horizon": 32,
    "lr": 3e-4,
    "critic_lr": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_epochs": 4,
    "minibatches": 1,
    "critic_epochs": 4,
    "clip_epsilon": 0.2,
    "ppo_target_kl": 0.0,
    "ppo_value_clip": 0.0,
    "bootstrap_weight": 1.0,
    "report_terminal_critic_grad": False,
    "entropy_weight": 0.0,
    "max_grad_norm": 1.0,
    "target_alpha": 0.995,
    "temporal_gradient_alpha": 0.0,
    "initial_log_std": -0.5,
    "weight_decay": 0.0,
    "task_weight": 1.0,
    "altitude_weight": 0.0,
    "height_boundary_weight": 0.0,
    "failure_cost": 0.0,
    "progress_reward_scale": 0.0,
    "critic_uses_sensor": False,
    "critic_uses_privileged": False,
    "perception_weight": 1.0,
    "velocity_aux_weight": 0.0,
    "perception_loss": {},
    "randomize_navigation_start": False,
}


class Trainer:
    """Own the update machinery; the caller owns budgets and evaluation schedules.

    Args:
        env: Shared Simulation Environment. Batch size and device belong to it.
        kind: Shared actor architecture: state, depth or lidar.
        algorithm: ppo, apg or shac.
        seed: Initialization seed, also recorded in the resolved configuration.
        config: Overrides of the options documented in docs/training.md.
    """

    def __init__(
        self,
        env,
        kind="state",
        algorithm="ppo",
        seed=0,
        config=None,
        *,
        actor_config=None,
        loss=None,
        backward_model=None,
        backward_options=None,
        actor=None,
        critic=None,
        method=None,
        objective=None,
    ):
        """Configure the shared actor, critic, optimizers and compiled update."""
        if algorithm not in ("ppo", "apg", "shac"):
            raise ValueError("Expected algorithm ppo/apg/shac")
        if actor is None and kind not in ("state", "depth", "lidar"):
            raise ValueError("Default Actor kind must be state/depth/lidar")
        if actor is not None and actor_config is not None:
            raise ValueError("Pass an Actor or actor_config, not both")
        options = dict(config or {})
        unknown = options.keys() - _DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown learning options: {sorted(unknown)}")
        self.config = {**_DEFAULTS, **options}
        self.env, self.kind, self.algorithm, self.seed = env, kind, algorithm, seed
        self.loss_name = loss
        self.backward_model, self.backward_options = backward_model, dict(backward_options or {})
        if not isinstance(self.config["randomize_navigation_start"], bool):
            raise ValueError("randomize_navigation_start must be boolean")
        if not isinstance(self.config["critic_uses_sensor"], bool):
            raise ValueError("critic_uses_sensor must be boolean")
        if self.config["critic_uses_sensor"] and (algorithm != "ppo" or env.sensor is None):
            raise ValueError("Sensor-aware critic requires PPO and a sensor observation")
        if not isinstance(self.config["critic_uses_privileged"], bool):
            raise ValueError("critic_uses_privileged must be boolean")
        if self.config["critic_uses_privileged"]:
            if algorithm != "ppo" or env.sensor is None or env.task.name != "navigation":
                raise ValueError("Privileged critic requires sensor-based Navigation PPO")
            if self.config["critic_uses_sensor"]:
                raise ValueError("Choose either sensor-aware or privileged critic")
        if self.config["progress_reward_scale"] and env.task.name not in ("navigation", "racing"):
            raise ValueError("progress_reward_scale requires Navigation or Racing")
        self._reset_options = {}
        if self.config["randomize_navigation_start"]:
            if env.task.name != "navigation":
                raise ValueError("randomize_navigation_start requires Navigation")
            self._reset_options["randomize_position"] = True
        for key in ("horizon", "ppo_epochs", "minibatches", "critic_epochs"):
            value = self.config[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if env.num_envs % self.config["minibatches"]:
            raise ValueError("minibatches must divide env.num_envs (whole recurrent sequences)")
        for key, value in self.config.items():
            if key != "perception_loss" and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in ("lr", "critic_lr", "max_grad_norm", "clip_epsilon"):
            if self.config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if self.config["ppo_target_kl"] < 0 or self.config["ppo_value_clip"] < 0:
            raise ValueError("PPO target KL and value clipping must be nonnegative")
        if algorithm != "ppo" and (
            self.config["ppo_target_kl"] > 0 or self.config["ppo_value_clip"] > 0
        ):
            raise ValueError("PPO target KL and value clipping require PPO")
        if not 0 <= self.config["bootstrap_weight"] <= 1:
            raise ValueError("SHAC bootstrap_weight must be in [0,1]")
        if algorithm != "shac" and (
            self.config["bootstrap_weight"] != 1 or self.config["report_terminal_critic_grad"]
        ):
            raise ValueError("Terminal bootstrap ablation requires SHAC")
        if not isinstance(self.config["report_terminal_critic_grad"], bool):
            raise ValueError("report_terminal_critic_grad must be boolean")
        for key in ("gamma", "gae_lambda", "target_alpha"):
            if not 0 <= self.config[key] <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        for key in (
            "temporal_gradient_alpha",
            "weight_decay",
            "task_weight",
            "altitude_weight",
            "height_boundary_weight",
            "failure_cost",
            "progress_reward_scale",
            "perception_weight",
            "entropy_weight",
            "velocity_aux_weight",
        ):
            if self.config[key] < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.config["velocity_aux_weight"] and loss != "zhang":
            raise ValueError("velocity_aux_weight requires the Zhang velocity prediction objective")
        if self.config["altitude_weight"] and env.task.name != "navigation":
            raise ValueError("altitude_weight requires a Navigation task")
        if self.config["height_boundary_weight"] and env.task.name != "navigation":
            raise ValueError("height_boundary_weight requires a Navigation task")
        if loss not in (None, "zhang", "liu"):
            raise ValueError("Loss must be null, zhang or liu")
        if loss is not None and env.action.level != "acceleration":
            raise ValueError("Named acceleration/jerk losses require acceleration commands")
        settings = dict(actor_config or {})
        if actor is not None:
            self.actor = actor
        elif "_target_" in settings:
            from hydra.utils import instantiate

            self.actor = instantiate(settings, action_size=env.action_size)
        else:
            self.actor = Actor(kind=kind, action_size=env.action_size, **settings)
        critic_kind = "state"
        if self.config["critic_uses_sensor"]:
            critic_kind = "depth" if env.sensor.config.profile == "d435i" else "lidar"
        self.critic = (
            critic
            if critic is not None
            else Critic(kind="privileged" if self.config["critic_uses_privileged"] else critic_kind)
        )
        self.method = method if method is not None else PolicyMethod()
        if not self.method.trainable or self.method.execution != "jax":
            raise ValueError("Training requires a JAX trainable Method")
        self._observe = (
            (
                lambda state: {
                    **env.observe(state),
                    "privileged_state": env.observe_privileged(state),
                }
            )
            if self.config["critic_uses_privileged"]
            else env.observe
        )
        recipe = {"zhang": zhang_loss, "liu": liu_loss}.get(loss)
        defaults = {
            key: parameter.default
            for key, parameter in (inspect.signature(recipe).parameters.items() if recipe else ())
            if parameter.default is not inspect.Parameter.empty
            and key not in ("velocity_prediction", "clearance_mask", "velocity_aux_weight")
        }
        invalid = self.config["perception_loss"].keys() - defaults.keys()
        if invalid:
            raise ValueError(f"Unsupported perception loss options: {sorted(invalid)}")
        defaults.update(self.config["perception_loss"])
        if any(not math.isfinite(value) or value < 0 for value in defaults.values()):
            raise ValueError("Perception loss options must be finite and nonnegative")
        if recipe and (
            not isinstance(defaults["velocity_window"], int) or defaults["velocity_window"] < 1
        ):
            raise ValueError("velocity_window must be a positive integer")
        for key in (
            ("huber_delta", "collision_beta" if loss == "zhang" else "beta2") if recipe else ()
        ):
            if defaults[key] <= 0:
                raise ValueError(f"{key} must be positive")
        self.config["perception_loss"] = defaults
        self.objective = (
            objective
            if objective is not None
            else EpisodeObjective(env, self.config, recipe=recipe, loss_name=loss)
        )
        from drone_playground.learning.dynamics import training_step

        self._differentiable_step = (
            training_step(env, backward_model, self.backward_options)
            if algorithm != "ppo"
            else env.step
        )
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config["max_grad_norm"]),
            optax.adamw(self.config["lr"], weight_decay=self.config["weight_decay"]),
        )
        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config["max_grad_norm"]),
            optax.adam(self.config["critic_lr"]),
        )
        from drone_playground.learning import apg, ppo, shac

        implementation = {"ppo": ppo.update, "apg": apg.update, "shac": shac.update}[algorithm]
        self._update = jax.jit(lambda state: implementation(self, state))

    @property
    def resolved_config(self):
        """Learning identity; callers add the resolved Simulation configuration."""
        return {
            **self.config,
            "kind": self.kind,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "num_envs": self.env.num_envs,
            "dt": self.env.dt,
            "actor": self.actor.specification,
            "action": self.env.action.contract,
            "loss": self.loss_name,
            "backward_model": self.backward_model,
            "backward_options": self.backward_options,
        }

    def initialize(self) -> TrainingState:
        """Initialize variables, optimizers, physical episodes and recurrent history."""
        rng, reset_key, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(self.seed), 4)
        env_state = self.env.reset(reset_key, **self._reset_options)
        observation = self._observe(env_state)
        memory = self.actor.initialize_memory(self.env.num_envs)
        params = self.actor.init(actor_key, observation, memory)
        critic_params = self.critic.init(critic_key, observation)
        log_std = self.method.initial_log_std(self.env, self.config["initial_log_std"])
        actor_parameters = (
            {"params": params, "log_std": log_std} if self.algorithm == "ppo" else params
        )
        history = self.objective.initialize_state(self.env.num_envs)
        return TrainingState(
            params=params,
            critic_params=critic_params,
            target_critic_params=critic_params,
            optimizer_state={
                "actor": self.actor_optimizer.init(actor_parameters),
                "critic": self.critic_optimizer.init(critic_params),
            },
            rng=rng,
            env_state=env_state,
            recurrent_memory=memory,
            objective_state=history,
            updates=0,
            interactions=0,
            log_std=log_std,
        )

    def reset_episodes(self, state: TrainingState) -> TrainingState:
        """Truncate sampling into this environment, keeping all shared learning state."""
        rng, reset_key = jax.random.split(state.rng)
        return state.replace(
            rng=rng,
            env_state=self.env.reset(reset_key, **self._reset_options),
            recurrent_memory=jnp.zeros_like(state.recurrent_memory),
            objective_state=jax.tree.map(jnp.zeros_like, state.objective_state),
        )

    def update(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
        """Run one compiled update; scalar metrics remain on device until consumed."""
        # Host integers avoid device int32 counter overflow without enabling global float64.
        result, metrics = self._update(state.replace(updates=0, interactions=0))
        result = result.replace(
            updates=state.updates + 1,
            interactions=state.interactions + self.config["horizon"] * self.env.num_envs,
        )
        return result, {**metrics, "updates": result.updates, "interactions": result.interactions}

    def save_state(self, path, state, *, provenance, config=None):
        """Atomically save a resumable checkpoint with caller-supplied run provenance."""
        from drone_playground.learning.checkpoint import save_state

        save_state(
            path,
            state,
            config={**(config or {}), "learning": self.resolved_config},
            provenance=provenance,
        )

    def load_state(self, path):
        """Restore against this trainer's structure and reject a mismatched recipe."""
        from drone_playground.learning.checkpoint import load_state

        state, metadata = load_state(path, self.initialize())
        saved_config = metadata["config"]["learning"]
        if saved_config != self.resolved_config:
            raise ValueError("Checkpoint learning configuration does not match this Trainer")
        return state, metadata

    def save_inference(self, path: str | Path, state: TrainingState, *, provenance, config=None):
        """Export standalone actor variables without optimizer or environment state."""
        from drone_playground.learning.checkpoint import save_inference

        save_inference(
            path,
            state.params,
            kind=self.kind,
            config={**(config or {}), "learning": self.resolved_config},
            provenance=provenance,
            actor_spec=self.actor.specification,
        )

    def _step(self, before, action, *, differentiable):
        if differentiable:
            before = before.replace(
                physics=temporal_gradient_decay(
                    before.physics, alpha=self.config["temporal_gradient_alpha"], dt=self.env.dt
                )
            )
        step = self._differentiable_step if differentiable else self.env.step
        return self.method.execute(self.env, before, action, step=step)

    def _reset(self, key, after, memory, history):
        from drone_playground.simulation.networks import reset_actor_memory

        done = after.done
        memory = reset_actor_memory(memory, done)
        history = jax.tree.map(
            lambda value: jnp.where(
                done.reshape((done.shape[0],) + (1,) * (value.ndim - 1)), 0, value
            ),
            history,
        )
        env_state = jax.lax.cond(
            jnp.any(done),
            lambda: self.env.reset(key, after, **self._reset_options),
            lambda: after,
        )
        return env_state, memory, history

    def _apply_actor(self, state, gradients):
        variables = (
            {"params": state.params, "log_std": state.log_std}
            if self.algorithm == "ppo"
            else state.params
        )
        updates, optimizer = self.actor_optimizer.update(
            gradients, state.optimizer_state["actor"], variables
        )
        variables = optax.apply_updates(variables, updates)
        return state.replace(
            params=variables["params"] if self.algorithm == "ppo" else variables,
            log_std=variables["log_std"] if self.algorithm == "ppo" else state.log_std,
            optimizer_state={**state.optimizer_state, "actor": optimizer},
        )

    def _fit_critic(self, state, observations, targets, epochs, *, old_values=None):
        observations = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), observations)
        targets = jax.lax.stop_gradient(targets.reshape(-1))
        clipped_values = (
            None if old_values is None else jax.lax.stop_gradient(old_values.reshape(-1))
        )

        def fit(state, _):
            def objective(params):
                values = self.critic.apply(params, observations)
                errors = (values - targets) ** 2
                if clipped_values is not None and self.config["ppo_value_clip"] > 0:
                    old = clipped_values
                    clipped = old + jnp.clip(
                        values - old,
                        -self.config["ppo_value_clip"],
                        self.config["ppo_value_clip"],
                    )
                    errors = jnp.maximum(errors, (clipped - targets) ** 2)
                return 0.5 * jnp.mean(errors)

            loss, gradients = jax.value_and_grad(objective)(state.critic_params)
            updates, optimizer = self.critic_optimizer.update(
                gradients, state.optimizer_state["critic"], state.critic_params
            )
            state = state.replace(
                critic_params=optax.apply_updates(state.critic_params, updates),
                optimizer_state={**state.optimizer_state, "critic": optimizer},
            )
            return state, {"critic_loss": loss, "critic_grad_norm": optax.tree.norm(gradients)}

        state, metrics = jax.lax.scan(fit, state, None, length=epochs)
        return state, jax.tree.map(jnp.mean, metrics)

    def _finish(self, state, metrics):
        return jax.lax.stop_gradient(state), metrics
