"""Observation encoders are shared by training and frozen evaluation."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TrackingObservation:
    name: str = "state_reference"
    n_samples: int = 10
    interval: float = 0.1

    def specification(self):
        """Declare the ordered array fields consumed by the existing network."""
        fields = [
            dict(name=name, shape=[size], units=units, frame=frame)
            for name, size, units, frame in (
                ("position", 3, "m", "world"),
                ("quaternion_xyzw", 4, "unit quaternion", "world-from-body"),
                ("velocity", 3, "m/s", "world"),
                ("angular_velocity", 3, "rad/s", "body"),
            )
        ]
        if self.name == "state_reference":
            fields.append(
                dict(
                    name="reference_position_delta",
                    shape=[self.n_samples, 3],
                    units="m",
                    frame="world",
                    interval_s=self.interval,
                )
            )
        return dict(encoding="ordered_vector", fields=fields, shape=[self.size])

    @property
    def size(self):
        return 13 + (3 * self.n_samples if self.name == "state_reference" else 0)

    def __call__(self, state, references):
        parts = [
            state.pos[0, 0],
            state.quat[0, 0],
            state.vel[0, 0],
            state.ang_vel[0, 0],
        ]
        if self.name == "state_reference":
            parts.append((references - state.pos[0, 0]).reshape(-1))
        elif self.name != "state":
            raise ValueError(f"Unknown observation encoder: {self.name}")
        return jnp.concatenate(parts)

    def reference_goal(self, observation):
        """Recover the declared first world reference, using observed fields only."""
        if (
            self.name != "state_reference"
            or self.n_samples < 1
            or observation.shape[-1] != self.size
        ):
            raise ValueError("Physical goal decoding requires a state_reference observation")
        return observation[..., :3] + observation[..., 13:16]


@dataclass(frozen=True)
class NavigationObservation:
    """Policy input for the navigation task.

    The body state, the goal direction and the previous normalized action are
    enumerated by the observation protocol. A sensor block is appended by the
    perception variants, which reuse this class's state fields unchanged so the
    proprioceptive input is identical across D435 and MID360 units.
    """

    name: str = "navigation_state"
    include_goal: bool = True
    include_previous_action: bool = True
    action_size: int = 4

    def specification(self):
        fields = TrackingObservation(name="state").specification()["fields"]
        if self.include_goal:
            fields.append(dict(name="goal_position_delta", shape=[3], units="m", frame="world"))
        if self.include_previous_action:
            fields.append(
                dict(
                    name="previous_action",
                    shape=[self.action_size],
                    units="normalized",
                    frame="controller",
                )
            )
        return dict(encoding="ordered_vector", fields=fields, shape=[self.size])

    @property
    def size(self) -> int:
        return (
            13
            + (3 if self.include_goal else 0)
            + (self.action_size if self.include_previous_action else 0)
        )

    def __call__(self, states, goal, previous_action, extra=None):
        parts = [
            states.pos[0, 0],
            states.quat[0, 0],
            states.vel[0, 0],
            states.ang_vel[0, 0],
        ]
        if self.include_goal:
            parts.append(goal - states.pos[0, 0])
        if self.include_previous_action:
            parts.append(previous_action)
        if extra is not None:
            parts.append(extra)
        return jnp.concatenate(parts)


@dataclass(frozen=True)
class NavigationSensorObservation:
    """Policy input for a range-sensor navigation unit.

    The proprioceptive block is exactly ``NavigationObservation``'s, so the
    D435 and MID360 units share one body/goal/action interface and differ only
    in the sensor block. Both units are sized to the same number of policy
    inputs, which keeps the network shape identical across the comparison.

    The sensor block is ``history * points_per_frame * channels`` and each
    channel is normalized into a bounded range with an explicit validity
    channel, so a missing return is distinguishable from a distant surface.
    """

    name: str = "navigation_depth"
    include_goal: bool = True
    include_previous_action: bool = True
    action_size: int = 4
    history: int = 4
    points_per_frame: int = 300
    channels: int = 2
    near_m: float = 0.1
    far_m: float = 10.0

    def specification(self):
        """Use named blocks while preserving the frozen encoder's vector layout."""
        fields = NavigationObservation(
            include_goal=self.include_goal,
            include_previous_action=self.include_previous_action,
            action_size=self.action_size,
        ).specification()["fields"]
        fields.append(
            dict(
                name="sensor_history",
                shape=[self.history, self.points_per_frame, self.channels],
                channels=["inverse_depth", "valid"]
                if self.channels == 2
                else ["x", "y", "z", "range", "valid"],
                units="normalized",
                frame="sensor",
                near_m=self.near_m,
                far_m=self.far_m,
            )
        )
        return dict(encoding="ordered_vector", fields=fields, shape=[self.size])

    def __post_init__(self) -> None:
        """Validate and prepare the NavigationSensorObservation instance after initialization."""
        if self.name not in ("navigation_depth", "navigation_lidar"):
            raise ValueError(f"Unknown sensor observation: {self.name}")
        if self.channels not in (2, 5):
            raise ValueError("sensor channels must be 2 (depth) or 5 (point cloud)")

    def bind_sensor(self, sensor):
        """Resolve the sensor block from its one physical calibration source."""
        if sensor is None:
            raise ValueError("A perception observation requires an actual sensor")
        if self.name == "navigation_depth":
            near, far = sensor.near_m, sensor.far_m
        else:
            near, far = sensor.range_m[0], sensor.normalise_far_m
        return replace(
            self,
            history=sensor.history,
            points_per_frame=sensor.points_per_frame,
            channels=sensor.channels,
            near_m=near,
            far_m=far,
        )

    @property
    def proprioception_size(self) -> int:
        return (
            13
            + (3 if self.include_goal else 0)
            + (self.action_size if self.include_previous_action else 0)
        )

    @property
    def sensor_size(self) -> int:
        return self.history * self.points_per_frame * self.channels

    @property
    def size(self) -> int:
        return self.proprioception_size + self.sensor_size

    def encode_sensor(self, values: jax.Array) -> jax.Array:
        """Normalize one frame's channels into the policy input.

        Depth uses inverse depth so that distance is monotone and bounded;
        LiDAR uses sensor-frame point coordinates, a normalized range and the
        validity mask. Both keep validity as an explicit channel.
        """
        if self.channels == 2:
            depth = values[..., 0]
            valid = values[..., 1] > 0.5
            clipped = jnp.clip(jnp.where(valid, depth, self.far_m), self.near_m, self.far_m)
            signal = (1.0 / clipped - 1.0 / self.far_m) / (1.0 / self.near_m - 1.0 / self.far_m)
            return jnp.stack([signal, valid.astype(jnp.float32)], axis=-1)
        points = values[..., :3] / self.far_m
        distance = jnp.clip(values[..., 3], 0.0, self.far_m) / self.far_m
        valid = (values[..., 4] > 0.5).astype(jnp.float32)
        return jnp.concatenate([points, distance[..., None], valid[..., None]], axis=-1)

    def __call__(self, states, goal, previous_action, extra=None):
        parts = [
            states.pos[0, 0],
            states.quat[0, 0],
            states.vel[0, 0],
            states.ang_vel[0, 0],
        ]
        if self.include_goal:
            parts.append(goal - states.pos[0, 0])
        if self.include_previous_action:
            parts.append(previous_action)
        if extra is None:
            raise ValueError("a perception observation requires a sensor history")
        parts.append(self.encode_sensor(extra["values"]).reshape(-1))
        return jnp.concatenate(parts)


def numerically_valid_observation(observation: jax.Array) -> jax.Array:
    """Require representable finite second moments, not just finite scalar entries."""
    return jnp.all(jnp.isfinite(observation)) & jnp.isfinite(jnp.sum(observation**2))
