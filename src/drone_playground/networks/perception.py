"""Shared perception encoder and the actor/critic built on top of it.

Depth and LiDAR share an observation interface, an embedding width, an action
contract, a task and a reward. Their encoders differ: depth uses convolution,
while LiDAR uses point-wise pooling. For each sensor, PPO and D.VA use the same
encoder and actor definition. The environment assembles the flat observation;
``SensorLayout`` splits it into proprioception and a ``(history, points,
channels)`` sensor block. Equal flattened input size does not imply equal
sensor information or equal encoder parameter counts.

Information boundary (spec 8.1)
    The actor sees only: body pose/velocity, the goal direction, the previous
    normalised action, and range measurements. It never sees the scene manifest,
    obstacle identity, or future obstacle motion.
    The critic consumes only the proprioceptive subset already available to the
    actor.  It receives no additional privileged obstacle or scene field.  This
    preserves a differentiable terminal-state value for D.VA while keeping PPO
    and D.VA on the same information permission boundary.
    :func:`privileged_critic_fields` states that explicitly so a run record can
    be checked rather than trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as linen
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.agents.ppo import networks as ppo_networks

from drone_playground.environments.observations.state import NavigationSensorObservation

PROPRIOCEPTION_FIELDS = (
    "body position (world, m)",
    "body orientation (world, xyzw)",
    "body linear velocity (world, m/s)",
    "body angular velocity (body, rad/s)",
    "goal position minus body position (world, m)",
    "previous normalised action (4,)",
)
"""Actor-visible proprioception, in the observation's declared field order."""

SENSOR_FIELDS = {
    "navigation_depth": (
        "inverse depth normalised to [0, 1] over the declared range, per sampled pixel",
        "depth validity mask (1 when the pixel returned inside the range)",
    ),
    "navigation_lidar": (
        "hit point in the sensor frame divided by the normalisation range (x, y, z)",
        "range divided by the normalisation range",
        "return validity mask",
    ),
}
"""Actor-visible sensor channels, per observation name."""


def privileged_critic_fields(critic_uses_sensor: bool = False) -> dict:
    """Value-function visibility for the matched PPO/D.VA comparison.

    The actor sees proprioception plus the selected sensor.  The critic is kept
    deliberately smaller: it consumes the differentiable proprioceptive block
    only.  This mirrors D.VA's state-value bootstrap without leaking obstacle
    truth, and it lets the D.VA terminal value keep a state derivative even
    though the actor observation is detached from the physical state.
    """
    return {
        "privileged_fields": [],
        "actor_fields": list(PROPRIOCEPTION_FIELDS)
        + ["selected range sensor history"],
        "critic_fields": list(PROPRIOCEPTION_FIELDS)
        + (["selected range sensor history"] if critic_uses_sensor else []),
        "actor_and_critic_share_observation_container": True,
        "critic_uses_sensor": critic_uses_sensor,
        "scene_manifest_visible_to_policy": False,
        "future_obstacle_motion_visible_to_policy": False,
        "rejected_privileged_candidates": [
            "current obstacle centres and sizes",
            "current obstacle velocities",
            "scenario_id and difficulty label",
            "per-obstacle signed clearance",
            "distance to the nearest obstacle",
        ],
        "note": (
            "Range sensors and the body state are the only inputs. Adding any rejected "
            "candidate to the critic requires recording it here and in the unit's "
            "components.json before the run."
        ),
    }


@dataclass(frozen=True)
class SensorLayout:
    """Static split of one perception observation into its two blocks."""

    kind: str
    proprioception_size: int
    history: int
    points_per_frame: int
    channels: int
    grid: tuple[int, int] | None = None
    embedding_size: int = 128

    def __post_init__(self) -> None:
        if self.kind not in ("depth", "lidar"):
            raise ValueError(f"Unknown sensor kind: {self.kind}")
        if self.kind == "depth" and self.grid is None:
            raise ValueError("the depth layout needs its sampling grid")
        if self.history < 1 or self.points_per_frame < 1 or self.channels < 1:
            raise ValueError("sensor layout dimensions must be positive")

    @classmethod
    def from_observation(
        cls, observation: NavigationSensorObservation, grid=None
    ) -> SensorLayout:
        if observation.name not in SENSOR_FIELDS:
            raise ValueError(
                f"Unknown perception observation: {observation.name}"
            )
        return cls(
            kind="depth" if observation.name == "navigation_depth" else "lidar",
            proprioception_size=observation.proprioception_size,
            history=observation.history,
            points_per_frame=observation.points_per_frame,
            channels=observation.channels,
            grid=tuple(grid) if grid is not None else None,
        )

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "proprioception_size": self.proprioception_size,
            "history": self.history,
            "points_per_frame": self.points_per_frame,
            "channels": self.channels,
            "grid": list(self.grid) if self.grid else None,
            "embedding_size": self.embedding_size,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> SensorLayout:
        return cls(
            kind=payload["kind"],
            proprioception_size=int(payload["proprioception_size"]),
            history=int(payload["history"]),
            points_per_frame=int(payload["points_per_frame"]),
            channels=int(payload["channels"]),
            grid=tuple(payload["grid"]) if payload.get("grid") else None,
            embedding_size=int(payload.get("embedding_size", 128)),
        )

    @property
    def observation_name(self) -> str:
        return (
            "navigation_depth" if self.kind == "depth" else "navigation_lidar"
        )

    @property
    def sensor_size(self) -> int:
        return self.history * self.points_per_frame * self.channels

    @property
    def total_size(self) -> int:
        return self.proprioception_size + self.sensor_size

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (self.history, self.points_per_frame, self.channels)

    def split(self, observations: jax.Array):
        """``(..., total)`` into proprioception and ``(..., H, P, C)`` frames."""
        if observations.shape[-1] != self.total_size:
            raise ValueError(
                f"observation has {observations.shape[-1]} entries, layout expects "
                f"{self.total_size}"
            )
        proprio = observations[..., : self.proprioception_size]
        sensor = observations[..., self.proprioception_size :]
        return proprio, sensor.reshape(*sensor.shape[:-1], *self.frame_shape)

    def contract(self) -> dict:
        return {
            "layout": self.as_dict(),
            "actor_fields": list(PROPRIOCEPTION_FIELDS)
            + list(SENSOR_FIELDS[self.observation_name]),
            "encoder": "convolution for depth, point-wise with max/mean pooling for LiDAR",
            "shared_with": "PPO and D.V.A use this exact encoder and actor input",
            "privileged": privileged_critic_fields(),
        }


class DepthFrameEncoder(linen.Module):
    """Per-frame convolutional encoder over the sampled depth grid."""

    embedding_size: int = 128
    grid: tuple[int, int] = (20, 15)
    activation: object = linen.elu

    @linen.compact
    def __call__(self, frames: jax.Array) -> jax.Array:
        """``(..., H, P, 2)`` -> ``(..., H * embedding)``.

        Leading dimensions are preserved so an unbatched single observation and a
        batched rollout share one code path.
        """
        *leading, history, points, channels = frames.shape
        height, width = self.grid
        if height * width != points:
            raise ValueError(
                f"depth grid {self.grid} does not cover {points} sampled pixels"
            )
        grid = frames.reshape(
            *leading, history, height, width, channels
        ).reshape(-1, height, width, channels)
        hidden = self.activation(
            linen.Conv(features=32, kernel_size=(3, 3), padding="SAME")(grid)
        )
        hidden = self.activation(
            linen.Conv(features=64, kernel_size=(3, 3), padding="SAME")(hidden)
        )
        pooled = jnp.concatenate(
            [hidden.mean(axis=(1, 2)), hidden.max(axis=(1, 2))], axis=-1
        )
        embedded = self.activation(linen.Dense(self.embedding_size)(pooled))
        return embedded.reshape(*leading, history * self.embedding_size)


class PointFrameEncoder(linen.Module):
    """Per-frame point-wise encoder with max and mean aggregation."""

    embedding_size: int = 128
    activation: object = linen.elu

    @linen.compact
    def __call__(self, frames: jax.Array) -> jax.Array:
        """``(..., H, P, C)`` -> ``(..., H * embedding)``."""
        *leading, history, points, channels = frames.shape
        flat = frames.reshape(-1, points, channels)
        hidden = self.activation(linen.Dense(64)(flat))
        hidden = self.activation(linen.Dense(64)(hidden))
        pooled = jnp.concatenate(
            [hidden.max(axis=1), hidden.mean(axis=1)], axis=-1
        )
        embedded = self.activation(linen.Dense(self.embedding_size)(pooled))
        return embedded.reshape(*leading, history * self.embedding_size)


class SharedEncoder(linen.Module):
    """Sensor-specific encoder dispatch shared by PPO and D.VA."""

    layout: SensorLayout = None  # type: ignore[assignment]
    activation: object = linen.elu

    @linen.compact
    def __call__(self, frames: jax.Array) -> jax.Array:
        if self.layout.kind == "depth":
            encoder = DepthFrameEncoder(
                embedding_size=self.layout.embedding_size, grid=self.layout.grid
            )
        else:
            encoder = PointFrameEncoder(
                embedding_size=self.layout.embedding_size
            )
        return encoder(frames)

    def embedding_dim(self) -> int:
        return self.layout.history * self.layout.embedding_size


class PerceptionActor(linen.Module):
    """Shared encoder followed by the policy head."""

    layout: SensorLayout = None  # type: ignore[assignment]
    hidden_sizes: tuple[int, ...] = (128, 128)
    param_size: int = 4
    activation: object = linen.elu
    mean_scale: float = 0.01
    head_mode: str = "mean_std"
    """``mean_std`` returns ``(loc, scale)``; ``concat_log_std`` returns one vector.

    Brax's ``NormalDistribution`` consumes a ``(loc, scale)`` tuple while
    ``NormalTanhDistribution`` consumes a single concatenated parameter vector,
    so the head shape is chosen by the distribution rather than assumed.
    """
    init_noise_std: float = 0.367879
    """Initial scale parameter, interpreted by the selected Brax distribution.

    ``mean_std`` exposes this value directly. In ``concat_log_std`` the stored
    log(seed) is an unconstrained Brax scale parameter: NormalTanhDistribution
    uses softplus(log(seed)) + 0.001, about 0.31426135 for the P5 default.
    It is not the post-tanh action standard deviation.
    """
    proprio_scale: tuple[float, ...] | None = None
    mean_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    sensor_encoder: str = "pointnet"
    range_scale: float = 40.0

    @linen.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        proprio, frames = self.layout.split(observations)
        if self.proprio_scale is not None:
            proprio = proprio / jnp.asarray(self.proprio_scale)
        if self.sensor_encoder == "polar_range":
            from drone_playground.environments.observations.polar_range import polar_range_features

            polar = polar_range_features(frames, range_scale=self.range_scale)
            embedded = polar.reshape(*polar.shape[:-2], -1)
        else:
            embedded = SharedEncoder(layout=self.layout)(frames)
        hidden = jnp.concatenate([proprio, embedded], axis=-1)
        for index, size in enumerate(self.hidden_sizes):
            hidden = self.activation(
                linen.Dense(size, name=f"hidden_{index}")(hidden)
            )
        mean = linen.Dense(
            self.param_size,
            name="mean_head",
            kernel_init=jax.nn.initializers.orthogonal(self.mean_scale),
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.asarray(
                self.mean_bias, dtype
            ),
        )(hidden)
        if self.head_mode == "mean":
            return mean
        log_std = self.param(
            "log_std",
            lambda _: jnp.full(
                (self.param_size,), jnp.log(self.init_noise_std)
            ),
        )
        if self.head_mode == "mean_std":
            return mean, jnp.broadcast_to(jnp.exp(log_std), mean.shape)
        if self.head_mode == "concat_log_std":
            return jnp.concatenate(
                [mean, jnp.broadcast_to(log_std, mean.shape)], axis=-1
            )
        raise ValueError(f"Unknown actor head mode: {self.head_mode}")


class PerceptionCritic(linen.Module):
    """Proprioceptive value function used by both matched algorithms.

    It intentionally ignores the appended sensor block.  For PPO this avoids a
    second expensive perception encoder; for D.VA it preserves a useful
    derivative from the frozen target value to the terminal physical state.
    """

    layout: SensorLayout = None  # type: ignore[assignment]
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: object = linen.elu
    proprio_scale: tuple[float, ...] | None = None
    uses_sensor: bool = False
    sensor_encoder: str = "pointnet"
    range_scale: float = 40.0

    @linen.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        proprio, frames = self.layout.split(observations)
        if self.proprio_scale is not None:
            proprio = proprio / jnp.asarray(self.proprio_scale)
        hidden = proprio
        if self.uses_sensor:
            if self.sensor_encoder == "polar_range":
                from drone_playground.environments.observations.polar_range import polar_range_features

                polar = polar_range_features(
                    frames, range_scale=self.range_scale
                )
                embedded = polar.reshape(*polar.shape[:-2], -1)
            else:
                embedded = SharedEncoder(layout=self.layout)(frames)
            hidden = jnp.concatenate([hidden, embedded], axis=-1)
        for index, size in enumerate(self.hidden_sizes):
            hidden = self.activation(
                linen.Dense(size, name=f"hidden_{index}")(hidden)
            )
        return linen.Dense(1, name="value_head")(hidden)


def _feed_forward(
    module, preprocess, squeeze: bool = False
) -> networks.FeedForwardNetwork:
    """Brax-shaped network: ``apply(processor_params, policy_params, obs)``.

    Brax's own value network squeezes its trailing singleton axis, so the critic
    must do the same or the PPO loss broadcasts ``(T, B, 1)`` against ``(T, B)``.
    """

    def init(key):
        return module.init(key, jnp.zeros((1, module.layout.total_size)))

    def apply(processor_params, policy_params, observations):
        out = module.apply(
            policy_params, preprocess(observations, processor_params)
        )
        return jnp.squeeze(out, axis=-1) if squeeze else out

    return networks.FeedForwardNetwork(init=init, apply=apply)


def perception_network_factory(layout: SensorLayout, config: dict):
    """Return a Brax ``network_factory`` bound to one perception layout."""
    hidden = tuple(config.get("hidden_sizes", (128, 128)))
    distribution_type = config.get("distribution_type", "normal")
    mean_scale = config.get("mean_kernel_scale", 0.01)
    init_noise_std = config.get("init_noise_std", 0.367879)
    encoder = config.get("sensor_encoder", "pointnet")
    if encoder not in ("pointnet", "polar_range"):
        raise ValueError("Unknown navigation sensor encoder")
    if encoder == "polar_range" and layout.kind != "lidar":
        raise ValueError(
            "The polar range encoder requires LiDAR measurement channels"
        )
    range_scale = float(config.get("range_scale", 40.0))
    if not math.isfinite(range_scale) or range_scale <= 0:
        raise ValueError(
            "The physical normalization range must be positive and finite"
        )
    proprio_scale = config.get("proprio_scale")
    if proprio_scale is not None:
        proprio_scale = tuple(float(value) for value in proprio_scale)
        if len(proprio_scale) != layout.proprioception_size or not all(
            math.isfinite(value) and value > 0 for value in proprio_scale
        ):
            raise ValueError(
                "proprio_scale must contain one finite positive divisor per state field"
            )
    action_bias = tuple(
        float(value) for value in config.get("mean_action_bias", (0, 0, 0, 0))
    )
    if len(action_bias) != 4 or not all(
        math.isfinite(value) for value in action_bias
    ):
        raise ValueError(
            "mean_action_bias must contain four finite normalized actions"
        )
    if distribution_type == "tanh_normal":
        if any(abs(value) >= 1 for value in action_bias):
            raise ValueError(
                "tanh mean_action_bias must lie strictly within (-1, 1)"
            )
        mean_bias = tuple(math.atanh(value) for value in action_bias)
    else:
        mean_bias = action_bias

    def factory(observation_size, action_size, **kwargs):
        preprocess = kwargs.get(
            "preprocess_observations_fn",
            types.identity_observation_preprocessor,
        )
        # Brax hands in a shape tuple for array observations, not a scalar size.
        size = observation_size
        if isinstance(size, (tuple, list)):
            if len(size) != 1:
                raise ValueError(f"unexpected observation shape: {size}")
            size = size[0]
        size = int(size)
        if size != layout.total_size:
            raise ValueError(
                f"environment observation size {size} does not match the "
                f"perception layout {layout.total_size}"
            )
        if int(action_size) != 4:
            raise ValueError(
                "the navigation action contract is four-dimensional"
            )
        if distribution_type == "normal":
            action_distribution = distribution.NormalDistribution(
                event_size=action_size
            )
        elif distribution_type == "tanh_normal":
            action_distribution = distribution.NormalTanhDistribution(
                event_size=action_size
            )
        else:
            raise ValueError(
                f"Unsupported distribution type: {distribution_type}"
            )
        # The head mode follows the distribution Brax will actually consume:
        # normal wants a (loc, scale) tuple, tanh_normal one concatenated vector.
        if distribution_type == "normal":
            head_mode = "mean_std"
            expected = action_size
        else:
            head_mode = "concat_log_std"
            expected = 2 * action_size
        if action_distribution.param_size != expected:
            raise ValueError(
                f"action distribution param size {action_distribution.param_size} does "
                f"not match the {head_mode} head width {expected}"
            )
        actor = PerceptionActor(
            layout=layout,
            hidden_sizes=hidden,
            param_size=action_size,
            mean_scale=mean_scale,
            head_mode=head_mode,
            init_noise_std=init_noise_std,
            proprio_scale=proprio_scale,
            mean_bias=mean_bias,
            sensor_encoder=encoder,
            range_scale=range_scale,
        )
        critic = PerceptionCritic(
            layout=layout,
            hidden_sizes=hidden,
            proprio_scale=proprio_scale,
            uses_sensor=bool(config.get("critic_uses_sensor", False)),
            sensor_encoder=encoder,
            range_scale=range_scale,
        )
        return ppo_networks.PPONetworks(
            policy_network=_feed_forward(actor, preprocess),
            value_network=_feed_forward(critic, preprocess, squeeze=True),
            parametric_action_distribution=action_distribution,
        )

    return factory
