"""Snapshot timing, delayed delivery and policy preprocessing."""

import math

import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.delay import delay_range
from drone_playground.simulation.paper_fan_lidar import (
    FanLidarConfig,
    empty_history,
    encode_range_ring,
    push_history,
    push_point_window,
)
from drone_playground.simulation.procedural_scene import ProceduralScene
from drone_playground.simulation.sensors import Measurement, SensorConfig, measure


def preprocess_depth(
    measurement: Measurement,
    *,
    near: float = 0.3,
    far: float = 10.0,
    scale: float = 3.0,
    offset: float = -0.6,
    pool: int = 4,
) -> tuple[jax.Array, jax.Array]:
    """Inverse axial depth and masked max-pooling for the depth policy."""
    depth, mask = measurement.values, measurement.mask
    height, width = depth.shape[-2:]
    if pool < 1 or height % pool or width % pool or not 0 < near < far:
        raise ValueError("Depth dimensions must be divisible by pool and 0 < near < far")
    inverse = scale / jnp.clip(jnp.where(mask, depth, far), near, far) + offset
    blocks = (*depth.shape[:-2], height // pool, pool, width // pool, pool)
    pooled = jnp.max(jnp.where(mask, inverse, -jnp.inf).reshape(blocks), axis=(-3, -1))
    valid = jnp.any(mask.reshape(blocks), axis=(-3, -1))
    return jnp.where(valid, pooled, 0), valid


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
    paper_fan_history: jax.Array | dict | None = None


class SensorObservation:
    """Capture one named device on its own clock and provide actor-ready observations."""

    def __init__(
        self,
        scene,
        config: SensorConfig,
        physics_hz: int,
        latency_bounds=None,
        observation_config: dict | None = None,
    ):
        """Initialize SensorObservation state for the declared contract."""
        if config.frequency_hz > physics_hz:
            raise ValueError("Sensor frequency cannot exceed physical pose sampling frequency")
        self.latency_bounds = delay_range(
            config.latency if latency_bounds is None else latency_bounds
        )
        self.capacity = math.ceil(self.latency_bounds[1] * config.frequency_hz) + 2
        observation_config = observation_config or {}
        self.encoding = dict(observation_config.get("depth", {}))
        self.scene, self.config = scene, config
        self.point_count = None
        if config.profile != "d435i":
            self.point_count = observation_config.get("lidar", {}).get("point_count")
            if (
                type(self.point_count) is not int
                or not 0 < self.point_count <= config.points_per_frame
            ):
                raise ValueError(
                    "observation.lidar.point_count must be a positive integer "
                    "no larger than the scan"
                )
        paper_fan = observation_config.get("fan_lidar")
        if paper_fan is not None and (config.profile != "mid360" or self.latency_bounds[1] > 0):
            raise ValueError("Fan LiDAR requires MID360 with zero measurement latency")
        self.paper_fan = None if paper_fan is None else FanLidarConfig(**paper_fan)

    @property
    def specification(self) -> dict:
        """Identify measurement and preprocessing semantics for runs and checkpoints."""
        result = {
            "config": self.config.specification,
            "encoding": self.encoding,
            "latency_bounds": list(self.latency_bounds),
        }
        if self.point_count is not None:
            result["point_count"] = self.point_count
        return result

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

    def _reset_depth_measurement(self, position, quaternion, mask, scene_state):
        """Capture only restarted procedural worlds, using fixed-shape GPU batches."""
        batch = position.shape[0]
        time = jnp.zeros(batch)

        def capture_all():
            return measure(
                self.scene, self.config, position, quaternion, time, scene_state=scene_state
            )

        if mask is None or not isinstance(self.scene, ProceduralScene):
            return capture_all()

        def capture_masked():
            chunk = min(batch, 32)
            indices = jnp.nonzero(mask, size=batch, fill_value=0)[0]
            indices = jnp.pad(indices, (0, (-batch) % chunk))
            active_count = jnp.count_nonzero(mask)
            height, width = self.config.height, self.config.width
            shape = (batch + 1, height, width)
            output = Measurement(
                jnp.zeros(shape),
                jnp.zeros(shape, dtype=bool),
                jnp.zeros((*shape, 3)),
                jnp.zeros(batch + 1),
                jnp.zeros(batch + 1),
            )

            def capture_chunk(index, accumulated):
                start = index * chunk
                selected = jax.lax.dynamic_slice_in_dim(indices, start, chunk)
                selected_state = jax.tree.map(
                    lambda array: jnp.take(array, selected, axis=0), scene_state
                )
                captured = measure(
                    self.scene,
                    self.config,
                    jnp.take(position, selected, axis=0),
                    jnp.take(quaternion, selected, axis=0),
                    jnp.zeros(chunk),
                    scene_state=selected_state,
                )
                # Extra row absorbs the padded indices without overwriting a live world.
                target = jnp.where(start + jnp.arange(chunk) < active_count, selected, batch)
                return jax.tree.map(lambda old, new: old.at[target].set(new), accumulated, captured)

            chunks = (active_count + chunk - 1) // chunk
            output = jax.lax.fori_loop(0, chunks, capture_chunk, output)
            return jax.tree.map(lambda array: array[:batch], output)

        return jax.lax.cond(jnp.all(mask), capture_all, capture_masked)

    def reset(self, physics, *, scene_state=None, mask=None) -> ObservationState:
        """Depth samples at reset; LiDAR publishes its first snapshot after one period."""
        p, q = physics.states.pos[:, 0], physics.states.quat[:, 0]
        batch = len(p)
        pose = jnp.concatenate([jnp.zeros((batch, 1)), p, q], -1)
        if self.config.profile == "d435i":
            measurement = self._reset_depth_measurement(p, q, mask, scene_state)
        else:
            shape = (batch, self.point_count)
            measurement = Measurement(
                jnp.zeros(shape),
                jnp.zeros(shape, bool),
                jnp.zeros((*shape, 3)),
                jnp.zeros(batch),
                jnp.zeros(batch),
            )
        low, high = self.latency_bounds
        delay = jax.random.uniform(
            jax.random.fold_in(physics.core.rng_key, 201), (batch,), minval=low, maxval=high
        )
        measurement = measurement._replace(available_time=measurement.acquisition_time + delay)
        acquisition_pose = jnp.concatenate([p, q], -1)
        fan_history = (
            None
            if self.paper_fan is None
            else empty_history(
                batch, frames=self.paper_fan.history_frames, sectors=self.paper_fan.sectors
            )
        )
        if self.paper_fan is not None and self.paper_fan.raw_history_frames > 1:
            shape = (batch, self.paper_fan.raw_history_frames, self.point_count)
            fan_history = {
                "ranges": fan_history,
                "points_world": jnp.zeros((*shape, 3)),
                "mask": jnp.zeros(shape, bool),
            }
        state = ObservationState(
            measurement=measurement,
            frame=jnp.zeros(batch, jnp.int32),
            previous_pose=pose,
            acquisition_pose=acquisition_pose,
            delay_s=delay,
            available=jnp.full(batch, self.config.profile == "d435i"),
            paper_fan_history=fan_history,
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

    def update(
        self, state: ObservationState, physics, active, *, scene_state=None
    ) -> ObservationState:
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
                selected_count=self.point_count,
                scene_state=scene_state,
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
        if self.paper_fan is not None:

            def capture_fan(_):
                # All snapshot hits share the recorded acquisition pose.
                quaternion = result.acquisition_pose[:, 3:]
                body_to_world = Rotation.from_quat(quaternion).as_matrix()
                if self.paper_fan.raw_history_frames > 1:
                    old = state.paper_fan_history
                    position = result.acquisition_pose[:, :3]
                    world_points = (
                        jnp.einsum("bij,bnj->bni", body_to_world, result.measurement.points_body)
                        + position[:, None]
                    )
                    points, masks = push_point_window(
                        old["points_world"], old["mask"], world_points, result.measurement.mask
                    )
                    relative = points.reshape(len(time), -1, 3) - position[:, None]
                    ring = encode_range_ring(
                        relative,
                        masks.reshape(len(time), -1),
                        sectors=self.paper_fan.sectors,
                        max_range_m=self.paper_fan.max_range_m,
                        half_height_m=self.paper_fan.half_height_m,
                    )
                    updated = {
                        "ranges": push_history(old["ranges"], ring),
                        "points_world": points,
                        "mask": masks,
                    }
                    return jax.tree.map(choose, old, updated)
                ring = encode_range_ring(
                    result.measurement.points_body,
                    result.measurement.mask,
                    rotation_body_to_level=body_to_world,
                    sectors=self.paper_fan.sectors,
                    max_range_m=self.paper_fan.max_range_m,
                    half_height_m=self.paper_fan.half_height_m,
                )
                next_history = push_history(state.paper_fan_history, ring)
                return jnp.where(due[:, None, None], next_history, state.paper_fan_history)

            result = result.replace(
                paper_fan_history=jax.lax.cond(
                    jnp.any(due), capture_fan, lambda _: state.paper_fan_history, None
                )
            )
        # Image/point acquisition is not part of the pathwise dynamics derivative.
        return jax.tree.map(jax.lax.stop_gradient, result)

    def encode(self, state: ObservationState) -> dict[str, jax.Array]:
        """Keep device calibration separate from the named policy preprocessing."""
        if self.config.profile == "d435i":
            if self.encoding.get("representation") == "metric":
                measurement = state.measurement
                return {
                    "depth": jnp.where(measurement.mask, measurement.values, 0.0)[..., None],
                    "depth_mask": measurement.mask,
                    "acquisition_time": measurement.acquisition_time,
                }
            depth, mask = preprocess_depth(state.measurement, **self.encoding)
            return {
                "depth": depth[..., None],
                "depth_mask": mask,
                "acquisition_time": state.measurement.acquisition_time,
            }
        measurement = state.measurement
        if self.paper_fan is not None:
            ranges = (
                state.paper_fan_history["ranges"]
                if self.paper_fan.raw_history_frames > 1
                else state.paper_fan_history
            )
            return {
                "fan_map": ranges[..., None],
                "acquisition_time": measurement.acquisition_time,
            }
        return {
            "points": measurement.points_body,
            "mask": measurement.mask,
            "point_times": jnp.broadcast_to(
                measurement.acquisition_time[..., None], measurement.values.shape
            ),
            "acquisition_time": measurement.acquisition_time,
        }
