"""Episode costs shared by PPO, BPTT and SHAC, independent of the trainer."""

import jax
import jax.numpy as jnp
from flax import struct

from drone_playground.simulation.task_rewards import shaped_task_cost


@struct.dataclass
class ObjectiveState:
    """Episode-local history for causal velocity and jerk terms."""

    velocity: jax.Array
    count: jax.Array
    acceleration: jax.Array
    jerk_mean: jax.Array
    jerk_m2: jax.Array


class EpisodeObjective:
    """Bind task reward, optional named perception cost and auxiliary targets.

    This is a configured objective, not a training algorithm. Both PPO rewards and
    BPTT/SHAC differentiable costs call the same function with the same history.
    """

    def __init__(self, env, config, *, recipe=None, loss_name=None):
        """Keep one immutable configuration for training and replay."""
        self.env = env
        self.config = config
        self.recipe = recipe
        self.loss_name = loss_name
        self.window = config["perception_loss"].get("velocity_window", 1)
        keys = (
            "task_weight",
            "progress_reward_scale",
            "failure_cost",
            "altitude_weight",
            "height_boundary_weight",
        )
        # Old checkpoints record task coefficients under Learning. New task
        # presets can own the coefficients explicitly; they take precedence.
        self.task_reward_shaping = {
            **{key: config[key] for key in keys},
            **env.task.reward_shaping,
        }

    def initialize_state(self, batch: int) -> ObjectiveState:
        """Initialize causal statistics without assuming an Actor memory layout."""
        return ObjectiveState(
            velocity=jnp.zeros((batch, self.window, 3)),
            count=jnp.zeros(batch, jnp.int32),
            acceleration=jnp.zeros((batch, 3)),
            jerk_mean=jnp.zeros(batch),
            jerk_m2=jnp.zeros(batch),
        )

    def cost(self, before, after, action, history):
        """Compose Task cost with optional learning-specific perceptual losses."""
        cost, terms = shaped_task_cost(self.env, before, after, action, self.task_reward_shaping)
        terms["perception_cost"] = jnp.zeros_like(cost)
        if self.recipe is None:
            return cost, history, terms
        velocity = after.physics.states.vel[:, 0]
        position = after.physics.states.pos[:, 0]
        delta = self.env.goal_for(after) - position
        distance = jnp.linalg.norm(delta, axis=-1, keepdims=True)
        target = delta / jnp.maximum(distance, 1e-6) * jnp.minimum(distance, 3.0)
        velocities = jnp.concatenate((history.velocity[:, 1:], velocity[:, None]), axis=1)
        count = history.count + 1
        smoothed = velocities.sum(axis=1) / jnp.minimum(count, self.window)[:, None]
        physical = self.env.action.decode(action)
        acceleration = (
            self.env.action.world_acceleration(before.physics, physical)
            if self.env.action.level == "acceleration"
            else physical
        )
        previous = jnp.where((history.count > 0)[:, None], history.acceleration, acceleration)
        clearance = (
            self.env.scene_clearance(position, after.time, after.scene_state) - self.env.task.radius
        )
        old_clearance = self.env.scene_clearance(
            before.physics.states.pos[:, 0], before.time, before.scene_state
        )
        old_clearance = old_clearance - self.env.task.radius
        valid = jnp.isfinite(clearance) & jnp.isfinite(old_clearance)
        approach = (
            jnp.where(valid, old_clearance, 0) - jnp.where(valid, clearance, 0)
        ) / self.env.dt
        kwargs = {**self.config["perception_loss"], "velocity_window": 1}

        def named_cost(velocity, target, previous, acceleration, clearance, approach, valid):
            total, terms = self.recipe(
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

            _, pair_terms = self.recipe(
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

    def auxiliary_cost(self, prediction, observation):
        """Train optional velocity estimation without modifying task reward."""
        if not self.config["velocity_aux_weight"]:
            return jnp.zeros(prediction.shape[:-1])
        target = jax.lax.stop_gradient(observation["velocity_body"])
        return self.config["velocity_aux_weight"] * jnp.mean((prediction - target) ** 2, axis=-1)
