"""Snapshot timing, delayed delivery and policy preprocessing."""

import math

import jax
import jax.numpy as jnp
from flax import struct

from drone_playground.simulation.delay import delay_range
from drone_playground.simulation.sensors import (
    Measurement,
    SensorConfig,
    measure,
    preprocess_depth,
    reduce_points,
)


@struct.dataclass
class ObservationState:
    """Latest snapshot and instance-owned state for sampling and delivery."""

    measurement: Measurement
    frame: jax.Array
    previous_pose: jax.Array
    acquisition_pose: jax.Array
    delay_s: jax.Array
    available: jax.Array
    pending: Measurement | None = None
    pending_poses: jax.Array | None = None
    pending_valid: jax.Array | None = None


class SensorObservation:
    """Capture one named device on its own clock and provide actor-ready observations."""

    def __init__(
        self,
        scene,
        config: SensorConfig,
        physics_hz: int,
        point_count: int = 1024,
        latency_bounds=None,
        encoding=None,
    ):
        """Initialize SensorObservation state for the declared contract."""
        if config.frequency_hz > physics_hz:
            raise ValueError("Sensor frequency cannot exceed physical pose sampling frequency")
        self.latency_bounds = delay_range(
            config.latency if latency_bounds is None else latency_bounds
        )
        self.capacity = math.ceil(self.latency_bounds[1] * config.frequency_hz) + 2
        self.encoding = dict(encoding or {})
        self.scene, self.config = scene, config
        self.point_count = point_count

    @property
    def specification(self) -> dict:
        """Identify measurement and preprocessing semantics for runs and checkpoints."""
        return {
            "config": self.config.specification,
            "point_count": self.point_count,
            "encoding": self.encoding,
            "latency_bounds": list(self.latency_bounds),
        }

    @staticmethod
    def _pose_at(previous, current, time):
        """Interpolate adjacent physical poses at a noninteger-rate frame time."""
        alpha = jnp.clip(
            (time - previous[:, 0]) / jnp.maximum(current[:, 0] - previous[:, 0], 1e-8), 0, 1
        )[:, None]
        position = previous[:, 1:4] + alpha * (current[:, 1:4] - previous[:, 1:4])
        first, last = previous[:, 4:], current[:, 4:]
        last = jnp.where(jnp.sum(first * last, axis=-1, keepdims=True) < 0, -last, last)
        quat = first + alpha * (last - first)
        quat = quat / jnp.maximum(jnp.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
        return position, quat

    def reset(self, physics) -> ObservationState:
        """Depth samples at reset; LiDAR publishes its first snapshot after one period."""
        p, q = physics.states.pos[:, 0], physics.states.quat[:, 0]
        batch = len(p)
        pose = jnp.concatenate([jnp.zeros((batch, 1)), p, q], -1)
        if self.config.profile == "d435i":
            measurement = measure(self.scene, self.config, p, q, jnp.zeros(batch))
        else:
            shape = (batch, self.config.points_per_frame)
            measurement = Measurement(
                jnp.zeros(shape),
                jnp.zeros(shape, bool),
                jnp.zeros((*shape, 3)),
                jnp.zeros((*shape, 3)),
                jnp.zeros(shape),
                jnp.zeros(batch),
                jnp.zeros(batch),
            )
        low, high = self.latency_bounds
        delay = jax.random.uniform(
            jax.random.fold_in(physics.core.rng_key, 201), (batch,), minval=low, maxval=high
        )
        measurement = measurement._replace(available_time=measurement.acquisition_time + delay)
        acquisition_pose = jnp.concatenate([p, q], -1)
        state = ObservationState(
            measurement=measurement,
            frame=jnp.zeros(batch, jnp.int32),
            previous_pose=pose,
            acquisition_pose=acquisition_pose,
            delay_s=delay,
            available=jnp.full(batch, self.config.profile == "d435i"),
        )
        if high == 0:
            return jax.tree.map(jax.lax.stop_gradient, state)
        empty = jax.tree.map(jnp.zeros_like, measurement)
        pending = jax.tree.map(
            lambda x: jnp.broadcast_to(x[:, None], (batch, self.capacity, *x.shape[1:])), empty
        )
        pending = jax.tree.map(lambda x, y: x.at[:, -1].set(y), pending, measurement)
        valid = jnp.zeros((batch, self.capacity), bool)
        valid = valid.at[:, -1].set(self.config.profile == "d435i")
        pending_poses = jnp.broadcast_to(acquisition_pose[:, None], (batch, self.capacity, 7))
        visible = delay == 0
        delivered = jax.tree.map(
            lambda old, new: jnp.where(visible.reshape((batch,) + (1,) * (new.ndim - 1)), new, old),
            empty,
            measurement,
        )
        return jax.tree.map(
            jax.lax.stop_gradient,
            state.replace(
                measurement=delivered,
                available=state.available & visible,
                pending=pending,
                pending_poses=pending_poses,
                pending_valid=valid & ~visible[:, None],
            ),
        )

    def update(self, state: ObservationState, physics, active) -> ObservationState:
        """Interpolate physical poses to exact, including noninteger-rate, capture times."""
        time = physics.core.steps[:, 0] / physics.core.freq
        p, q = physics.states.pos[:, 0], physics.states.quat[:, 0]
        pose = jnp.concatenate([time[:, None], p, q], -1)
        previous_pose = jnp.where(active[:, None], pose, state.previous_pose)
        next_time = (state.frame + 1) / self.config.frequency_hz
        due = (time + 1e-7 >= next_time) & active

        def choose(old, new):
            mask = due.reshape(due.shape + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)

        def capture(_):
            capture_time = jnp.where(due, next_time, time)

            positions, quaternions = self._pose_at(state.previous_pose, pose, capture_time)
            measurement = measure(
                self.scene,
                self.config,
                positions,
                quaternions,
                capture_time,
                frame_index=state.frame,
            )
            measurement = measurement._replace(
                available_time=measurement.acquisition_time + state.delay_s
            )
            acquisition_pose = jnp.concatenate([positions, quaternions], -1)
            return (
                jax.tree.map(choose, state.measurement, measurement),
                choose(state.acquisition_pose, acquisition_pose),
            )

        measurement, acquisition_pose = jax.lax.cond(
            jnp.any(due),
            capture,
            lambda _: (state.measurement, state.acquisition_pose),
            None,
        )
        pending, pending_poses, pending_valid = (
            state.pending,
            state.pending_poses,
            state.pending_valid,
        )
        available = state.available | due
        if pending is not None:

            def append(old, new):
                return choose(old, jnp.concatenate([old[:, 1:], new[:, None]], axis=1))

            pending, pending_poses, pending_valid = jax.lax.cond(
                jnp.any(due),
                lambda: (
                    jax.tree.map(append, pending, measurement),
                    append(pending_poses, acquisition_pose),
                    append(pending_valid, jnp.ones_like(due)),
                ),
                lambda: (pending, pending_poses, pending_valid),
            )
            delivered = (
                pending_valid & (pending.available_time <= time[:, None] + 1e-7) & active[:, None]
            )
            index = jnp.max(jnp.where(delivered, jnp.arange(self.capacity)[None], 0), axis=1)
            publish = jnp.any(delivered, axis=1) & active
            row = jnp.arange(len(time))

            def take(old, new):
                new = new[row, index]
                mask = publish.reshape((len(time),) + (1,) * (new.ndim - 1))
                return jnp.where(mask, new, old)

            measurement, acquisition_pose = jax.lax.cond(
                jnp.any(publish),
                lambda: (
                    jax.tree.map(take, state.measurement, pending),
                    take(state.acquisition_pose, pending_poses),
                ),
                lambda: (state.measurement, state.acquisition_pose),
            )
            pending_valid = pending_valid & ~delivered
            available = state.available | publish
        result = state.replace(
            measurement=measurement,
            acquisition_pose=acquisition_pose,
            frame=state.frame + due.astype(jnp.int32),
            previous_pose=previous_pose,
            pending=pending,
            pending_poses=pending_poses,
            pending_valid=pending_valid,
            available=available,
        )
        # Image/point acquisition is not part of the pathwise dynamics derivative.
        return jax.tree.map(jax.lax.stop_gradient, result)

    def encode(self, state: ObservationState) -> dict[str, jax.Array]:
        """Keep device calibration separate from the named policy preprocessing."""
        if self.config.profile == "d435i":
            depth, mask = preprocess_depth(state.measurement, **self.encoding)
            return {
                "depth": depth[..., None],
                "depth_mask": mask,
                "acquisition_time": state.measurement.acquisition_time,
            }
        measurement = reduce_points(state.measurement, self.point_count)
        return {
            "points": measurement.points_body,
            "mask": measurement.mask,
            "point_times": measurement.times,
            "acquisition_time": measurement.acquisition_time,
        }
