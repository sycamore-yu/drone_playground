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
from drone_playground.simulation.networks import Actor, Critic


@struct.dataclass
class ObjectiveState:
    """Episode-local history required by the common causal perception cost."""

    velocity: jax.Array
    count: jax.Array
    acceleration: jax.Array
    jerk_mean: jax.Array
    jerk_m2: jax.Array


@struct.dataclass
class TrainingState:
    """All mutable learning state, including environment and objective history."""

    params: Any
    critic_params: Any
    target_critic_params: Any
    optimizer_state: Any
    rng: jax.Array
    env_state: Any
    recurrent_memory: jax.Array
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
    ):
        """Configure the shared actor, critic, optimizers and compiled update."""
        if kind not in ("state", "depth", "lidar") or algorithm not in ("ppo", "apg", "shac"):
            raise ValueError("Expected kind state/depth/lidar and algorithm ppo/apg/shac")
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
        if self.config["critic_uses_sensor"] and (algorithm != "ppo" or kind == "state"):
            raise ValueError("Sensor-aware critic requires Depth/LiDAR PPO")
        if not isinstance(self.config["critic_uses_privileged"], bool):
            raise ValueError("critic_uses_privileged must be boolean")
        if self.config["critic_uses_privileged"]:
            if algorithm != "ppo" or kind == "state" or env.task.name != "navigation":
                raise ValueError("Privileged critic requires Depth/LiDAR Navigation PPO")
            if self.config["critic_uses_sensor"]:
                raise ValueError("Choose either sensor-aware or privileged critic")
        if self.config["progress_reward_scale"] and env.task.name != "navigation":
            raise ValueError("progress_reward_scale requires Navigation")
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
        if self.config["velocity_aux_weight"] and kind != "depth":
            raise ValueError("velocity_aux_weight requires the depth actor")
        if self.config["altitude_weight"] and env.task.name != "navigation":
            raise ValueError("altitude_weight requires a Navigation task")
        if self.config["height_boundary_weight"] and env.task.name != "navigation":
            raise ValueError("height_boundary_weight requires a Navigation task")
        if loss not in (None, "zhang", "liu"):
            raise ValueError("Loss must be null, zhang or liu")
        if loss is not None and env.action.level != "acceleration":
            raise ValueError("Named acceleration/jerk losses require acceleration commands")
        settings = dict(actor_config or {})
        self.actor = Actor(kind=kind, action_size=env.action_size, **settings)
        critic_kind = kind if self.config["critic_uses_sensor"] else "state"
        self.critic = Critic(
            kind="privileged" if self.config["critic_uses_privileged"] else critic_kind
        )
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
        self._recipe = recipe
        self._window = defaults.get("velocity_window", 1)
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
        log_std = jnp.full(
            (self.env.action_size,), self.config["initial_log_std"], dtype=memory.dtype
        )
        actor_parameters = (
            {"params": params, "log_std": log_std} if self.algorithm == "ppo" else params
        )
        history = ObjectiveState(
            velocity=jnp.zeros((self.env.num_envs, self._window, 3)),
            count=jnp.zeros(self.env.num_envs, jnp.int32),
            acceleration=jnp.zeros((self.env.num_envs, 3)),
            jerk_mean=jnp.zeros(self.env.num_envs),
            jerk_m2=jnp.zeros(self.env.num_envs),
        )
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
        return (
            self._differentiable_step(before, action)
            if differentiable
            else self.env.step(before, action)
        )

    def _reset(self, key, after, memory, history):
        done = after.done
        memory = jnp.where(done[:, None], 0, memory)
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

    def _cost(self, before, after, action, history):
        """Identical task and named perception reward costs for all three algorithms."""
        from drone_playground.simulation.tasks import Event

        task_cost = self.env.cost(before, after, action)
        progress_reward = jnp.zeros_like(task_cost)
        if self.config["progress_reward_scale"]:
            goal = self.env.task.goal
            previous_distance = jnp.linalg.norm(goal - before.physics.states.pos[:, 0], axis=-1)
            distance = jnp.linalg.norm(goal - after.physics.states.pos[:, 0], axis=-1)
            progress_reward = self.config["progress_reward_scale"] * (previous_distance - distance)
            progress_reward = jnp.where(before.done, 0.0, progress_reward)
            task_cost -= progress_reward
        failure = after.done & (after.task.event != Event.SUCCESS) & ~before.done
        extra_failure_cost = self.config["failure_cost"] * failure
        task_cost += extra_failure_cost
        height_boundary_cost = jnp.zeros_like(task_cost)
        if self.config["altitude_weight"]:
            altitude_error = after.physics.states.pos[:, 0, 2] - self.env.task.goal[2]
            task_cost += self.config["altitude_weight"] * self.env.dt * altitude_error**2
        if self.config["height_boundary_weight"]:
            height = after.physics.states.pos[:, 0, 2]
            edge_distance = jnp.minimum(
                height - self.env.task.low[2], self.env.task.high[2] - height
            )
            height_boundary_cost = (
                self.config["height_boundary_weight"]
                * self.env.dt
                * jax.nn.relu(1.0 - edge_distance) ** 2
            )
            height_boundary_cost = jnp.where(before.done, 0.0, height_boundary_cost)
            task_cost += height_boundary_cost
        cost = self.config["task_weight"] * task_cost
        terms = {
            "task_cost": task_cost,
            "failure_cost": extra_failure_cost,
            "height_boundary_cost": height_boundary_cost,
            "progress_reward": progress_reward,
            "perception_cost": jnp.zeros_like(cost),
        }
        if self._recipe is None:
            return cost, history, terms
        velocity = after.physics.states.vel[:, 0]
        position = after.physics.states.pos[:, 0]
        delta = self.env.task.goal - position
        distance = jnp.linalg.norm(delta, axis=-1, keepdims=True)
        target = delta / jnp.maximum(distance, 1e-6) * jnp.minimum(distance, 3.0)
        velocities = jnp.concatenate((history.velocity[:, 1:], velocity[:, None]), axis=1)
        count = history.count + 1
        smoothed = velocities.sum(axis=1) / jnp.minimum(count, self._window)[:, None]
        acceleration = self.env.action.decode(action)
        previous = jnp.where((history.count > 0)[:, None], history.acceleration, acceleration)
        clearance = self.env.scene.clearance(position, after.time) - self.env.task.radius
        old_clearance = self.env.scene.clearance(before.physics.states.pos[:, 0], before.time)
        old_clearance = old_clearance - self.env.task.radius
        valid = jnp.isfinite(clearance) & jnp.isfinite(old_clearance)
        approach = (
            jnp.where(valid, old_clearance, 0) - jnp.where(valid, clearance, 0)
        ) / self.env.dt
        kwargs = {**self.config["perception_loss"], "velocity_window": 1}

        def named_cost(velocity, target, previous, acceleration, clearance, approach, valid):
            total, terms = self._recipe(
                velocity[None, None],
                target[None, None],
                acceleration[None, None],
                clearance[None, None],
                approach[None, None],
                clearance_mask=valid[None, None],
                dt=self.env.dt,
                **kwargs,
            )

            def repeat(value):
                return jnp.broadcast_to(value, (2, 1, *value.shape))

            # Extract only the two-sample jerk, retaining the current acceleration cost.
            _, pair_terms = self._recipe(
                repeat(velocity),
                repeat(target),
                jnp.stack((previous, acceleration))[:, None],
                repeat(clearance),
                repeat(approach),
                clearance_mask=repeat(valid),
                dt=self.env.dt,
                **kwargs,
            )
            terms["jerk"] = pair_terms["jerk"]
            if self.loss_name == "liu":
                terms["jerk_mean"] = pair_terms["jerk_mean"]
            return total + kwargs["jerk_weight"] * pair_terms["jerk"], terms

        named, named_terms = jax.vmap(named_cost)(
            smoothed, target, previous, acceleration, clearance, approach, valid
        )
        squared = jnp.sum(((acceleration - previous) / self.env.dt) ** 2, axis=-1)
        jerk = jnp.where(squared > 0, jnp.sqrt(jnp.where(squared > 0, squared, 1)), 0)
        jerk_count = jnp.maximum(count - 1, 1)
        difference = jerk - history.jerk_mean
        jerk_mean = history.jerk_mean + difference / jerk_count
        jerk_m2 = history.jerk_m2 + difference * (jerk - jerk_mean)
        if self.loss_name == "liu":
            variance = jerk_m2 / jerk_count
            named = named + kwargs["jerk_weight"] * kwargs["jerk_variance_weight"] * variance
            named_terms["jerk_variance"] = variance
            named_terms["jerk"] += kwargs["jerk_variance_weight"] * variance
        history = ObjectiveState(velocities, count, acceleration, jerk_mean, jerk_m2)
        cost = cost + self.config["perception_weight"] * named * self.env.dt
        terms.update({f"named_{name}": value for name, value in named_terms.items()})
        terms["perception_cost"] = named * self.env.dt
        return cost, history, terms

    def _auxiliary_cost(self, prediction, observation):
        if not self.config["velocity_aux_weight"]:
            return jnp.zeros(prediction.shape[:-1])
        target = jax.lax.stop_gradient(observation["velocity_body"])
        return self.config["velocity_aux_weight"] * jnp.mean((prediction - target) ** 2, axis=-1)

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

    def _fit_critic(self, state, observations, targets, epochs):
        observations = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), observations)
        targets = jax.lax.stop_gradient(targets.reshape(-1))

        def fit(state, _):
            def objective(params):
                return 0.5 * jnp.mean((self.critic.apply(params, observations) - targets) ** 2)

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
