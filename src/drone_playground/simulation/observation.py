"""Sensor timing, pose history and policy preprocessing, independent of learning."""

import math

import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.spatial.transform import Rotation

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
    """Latest completed measurement and physical poses needed to sample it."""

    measurement: Measurement
    frame: jax.Array
    pose_history: jax.Array
    points_at_completion: jax.Array
    acquisition_pose: jax.Array
    delay_s: jax.Array
    available: jax.Array
    pending: Measurement | None = None
    pending_points: jax.Array | None = None
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
        self.history_length = math.ceil(physics_hz / config.frequency_hz) + 2
        self.point_count = point_count

    @staticmethod
    def _pose_at(history, times):
        def interpolate(samples, query):
            return jax.vmap(
                lambda column: jnp.interp(query, samples[:, 0], column), in_axes=1, out_axes=-1
            )(samples[:, 1:])

        poses = jax.vmap(interpolate)(history, times)
        quat = poses[..., 3:]
        quat = quat / jnp.maximum(jnp.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
        return poses[..., :3], quat

    def reset(self, physics) -> ObservationState:
        """Depth is available at reset; a scanning LiDAR needs one full scan period."""
        p, q = physics.states.pos[:, 0], physics.states.quat[:, 0]
        batch = len(p)
        pose = jnp.concatenate([jnp.zeros((batch, 1)), p, q], -1)
        history = jnp.broadcast_to(pose[:, None], (batch, self.history_length, 8))
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
        if high == 0:
            return ObservationState(
                measurement,
                jnp.zeros(batch, jnp.int32),
                history,
                measurement.points_body,
                acquisition_pose,
                delay,
                jnp.full(batch, self.config.profile == "d435i"),
            )
        empty = jax.tree.map(jnp.zeros_like, measurement)
        pending = jax.tree.map(
            lambda x: jnp.broadcast_to(x[:, None], (batch, self.capacity, *x.shape[1:])), empty
        )
        pending = jax.tree.map(lambda x, y: x.at[:, -1].set(y), pending, measurement)
        valid = jnp.zeros((batch, self.capacity), bool)
        valid = valid.at[:, -1].set(self.config.profile == "d435i")
        pending_points = jnp.broadcast_to(
            measurement.points_body[:, None],
            (batch, self.capacity, *measurement.points_body.shape[1:]),
        )
        pending_poses = jnp.broadcast_to(acquisition_pose[:, None], (batch, self.capacity, 7))
        visible = delay == 0
        delivered = jax.tree.map(
            lambda old, new: jnp.where(visible.reshape((batch,) + (1,) * (new.ndim - 1)), new, old),
            empty,
            measurement,
        )
        return ObservationState(
            delivered,
            jnp.zeros(batch, jnp.int32),
            history,
            delivered.points_body,
            acquisition_pose,
            delay,
            visible & (self.config.profile == "d435i"),
            pending,
            pending_points,
            pending_poses,
            valid,
        )

    def update(self, state: ObservationState, physics, active) -> ObservationState:
        """Interpolate physical poses to exact, including noninteger-rate, capture times."""
        time = physics.core.steps[:, 0] / physics.core.freq
        p, q = physics.states.pos[:, 0], physics.states.quat[:, 0]
        pose = jnp.concatenate([time[:, None], p, q], -1)
        history = jnp.concatenate([state.pose_history[:, 1:], pose[:, None]], 1)
        history = jnp.where(active[:, None, None], history, state.pose_history)
        next_time = (state.frame + 1) / self.config.frequency_hz
        due = (time + 1e-7 >= next_time) & active

        def choose(old, new):
            mask = due.reshape(due.shape + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)

        def capture(_):
            capture_time = jnp.where(due, next_time, time)

            def pose_at(t):
                return self._pose_at(history, t)

            positions, quaternions = pose_at(capture_time[:, None])
            measurement = measure(
                self.scene,
                self.config,
                positions[:, 0],
                quaternions[:, 0],
                capture_time,
                frame_index=state.frame + 1,
                pose_at=pose_at,
            )
            measurement = measurement._replace(
                available_time=measurement.acquisition_time + state.delay_s
            )
            if self.config.profile != "d435i":
                ray_position, ray_quaternion = pose_at(measurement.times)
                points = (
                    Rotation.from_quat(ray_quaternion.reshape(-1, 4))
                    .apply(measurement.points_body.reshape(-1, 3))
                    .reshape(measurement.points_body.shape)
                )
                points = points + ray_position - positions
                points = jax.vmap(
                    lambda quat, cloud: Rotation.from_quat(quat).apply(cloud, inverse=True)
                )(quaternions[:, 0], points)
                points = jnp.where(measurement.mask[..., None], points, 0.0)
            else:
                points = measurement.points_body
            # Select full frames only on capture ticks, preserving worlds that are not due.
            return measurement, points, jnp.concatenate([positions[:, 0], quaternions[:, 0]], -1)

        measurement, points, pose = jax.lax.cond(
            jnp.any(due),
            capture,
            lambda _: (state.measurement, state.points_at_completion, state.acquisition_pose),
            None,
        )
        if state.pending is None:
            result = state.replace(
                measurement=jax.tree.map(choose, state.measurement, measurement),
                points_at_completion=choose(state.points_at_completion, points),
                acquisition_pose=choose(state.acquisition_pose, pose),
                available=state.available | due,
                frame=state.frame + due.astype(jnp.int32),
                pose_history=history,
            )
        else:

            def append(old, new):
                return choose(old, jnp.concatenate([old[:, 1:], new[:, None]], axis=1))

            pending = jax.tree.map(append, state.pending, measurement)
            pending_points = append(state.pending_points, points)
            pending_poses = append(state.pending_poses, pose)
            valid = append(state.pending_valid, jnp.ones_like(due))
            delivered = valid & (pending.available_time <= time[:, None] + 1e-7)
            index = jnp.max(jnp.where(delivered, jnp.arange(self.capacity)[None], 0), axis=1)
            available = jnp.any(delivered, axis=1) & active
            row = jnp.arange(len(time))

            def take(old, new):
                new = new[row, index]
                mask = available.reshape((len(time),) + (1,) * (new.ndim - 1))
                return jnp.where(mask, new, old)

            result = state.replace(
                measurement=jax.tree.map(take, state.measurement, pending),
                points_at_completion=take(state.points_at_completion, pending_points),
                acquisition_pose=take(state.acquisition_pose, pending_poses),
                frame=state.frame + due.astype(jnp.int32),
                pose_history=history,
                pending=pending,
                pending_points=pending_points,
                pending_poses=pending_poses,
                pending_valid=valid,
                available=state.available | available,
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
        measurement = reduce_points(
            state.measurement._replace(points_body=state.points_at_completion), self.point_count
        )
        return {
            "points": measurement.points_body,
            "mask": measurement.mask,
            "point_times": measurement.times,
            "acquisition_time": measurement.acquisition_time,
        }
