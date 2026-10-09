"""Sensor timing, pose history and policy preprocessing, independent of learning."""

import math

import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.spatial.transform import Rotation

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


class SensorObservation:
    """Capture one named device on its own clock and provide actor-ready observations."""

    def __init__(self, scene, config: SensorConfig, physics_hz: int, point_count: int = 1024):
        """Initialize SensorObservation state for the declared contract."""
        if config.frequency_hz > physics_hz:
            raise ValueError("Sensor frequency cannot exceed physical pose sampling frequency")
        if config.latency != 0:
            raise ValueError("Buffered sensor observations currently require zero delivery latency")
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
        return ObservationState(
            measurement, jnp.zeros(batch, jnp.int32), history, measurement.points_body
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
            return (
                jax.tree.map(choose, state.measurement, measurement),
                choose(state.points_at_completion, points),
            )

        measurement, points = jax.lax.cond(
            jnp.any(due), capture, lambda _: (state.measurement, state.points_at_completion), None
        )

        result = ObservationState(
            measurement,
            state.frame + due.astype(jnp.int32),
            history,
            points,
        )
        # Image/point acquisition is not part of the pathwise dynamics derivative.
        return jax.tree.map(jax.lax.stop_gradient, result)

    def encode(self, state: ObservationState) -> dict[str, jax.Array]:
        """Keep device calibration separate from the named policy preprocessing."""
        if self.config.profile == "d435i":
            depth, mask = preprocess_depth(state.measurement)
            if depth.shape[-2:] != (12, 16):
                raise ValueError("The Zhang actor requires the declared 64x48 acquisition mode")
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
