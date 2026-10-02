"""Paper Eq. (4)-(8), with every unreported coefficient exposed in configuration."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.geometry import euclidean_norm


def causal_average(values, window):
    """Mean of available current/past samples, never using a future measurement."""
    if window < 1:
        raise ValueError("Velocity averaging window must be positive")
    cumulative = jnp.concatenate(
        [jnp.zeros_like(values[:1]), jnp.cumsum(values, axis=0)]
    )
    end = jnp.arange(1, values.shape[0] + 1)
    start = jnp.maximum(0, end - window)
    counts = (end - start).reshape((-1,) + (1,) * (values.ndim - 1))
    return (cumulative[end] - cumulative[start]) / counts


def huber(error, delta=1.0):
    absolute = jnp.abs(error)
    return jnp.where(
        absolute <= delta, 0.5 * error * error, delta * (absolute - 0.5 * delta)
    )


@dataclass(frozen=True)
class VelocityTrackingAvoidanceObjective:
    name: str = "pointcloud_loss"
    velocity_weight: float = 1.0
    collision_weight: float = 1.5
    acceleration_weight: float = 0.01
    jerk_weight: float = 0.001
    velocity_norm_weight: float = 0.8
    velocity_component_weight: float = 0.6
    velocity_window: int = 30
    huber_delta: float = 1.0
    collision_beta1: float = 4 / 3
    collision_beta2: float = 32.0
    jerk_mean_weight: float = 1.0
    jerk_variance_weight: float = 0.1

    def __call__(self, trajectory, dt):
        """Return scalar mean loss and named components for a [time,batch,...] trace."""
        velocity = causal_average(trajectory["velocity"], self.velocity_window)
        delta = trajectory["target_velocity"] - velocity
        velocity_loss = jnp.mean(
            self.velocity_norm_weight
            * huber(euclidean_norm(delta), self.huber_delta)
            + self.velocity_component_weight
            * jnp.sum(huber(delta, self.huber_delta), axis=-1)
        )
        clearance = trajectory["clearance"]
        approaching = jax.lax.stop_gradient(
            jnp.maximum(trajectory["approaching_speed"], 0.0)
        )
        penalty = jnp.maximum(1.0 - clearance, 0.0) ** 2
        penalty += self.collision_beta1 * jax.nn.softplus(
            -self.collision_beta2 * clearance
        )
        collision_loss = jnp.mean(approaching * penalty)
        acceleration = trajectory["acceleration"]
        acceleration_loss = jnp.mean(
            jnp.sum(acceleration * acceleration, axis=-1)
        )
        previous = jnp.concatenate(
            [jnp.zeros_like(acceleration[:1]), acceleration[:-1]]
        )
        jerk_magnitude = euclidean_norm((acceleration - previous) / dt)
        jerk_mean = jnp.mean(jerk_magnitude)
        jerk_variance = jnp.mean(jnp.var(jerk_magnitude, axis=0))
        jerk_loss = (
            self.jerk_mean_weight * jerk_mean
            + self.jerk_variance_weight * jerk_variance
        )
        parts = dict(
            velocity=velocity_loss,
            collision=collision_loss,
            acceleration=acceleration_loss,
            jerk=jerk_loss,
            jerk_mean=jerk_mean,
            jerk_variance=jerk_variance,
        )
        total = (
            self.velocity_weight * velocity_loss
            + self.collision_weight * collision_loss
            + self.acceleration_weight * acceleration_loss
            + self.jerk_weight * jerk_loss
        )
        return total, parts
