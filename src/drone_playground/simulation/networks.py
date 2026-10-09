"""Shared, algorithm-independent Flax actors and standalone value network.

Frozen inference depends only on Simulation. Observations are already preprocessed;
coordinates, action scaling, RNG, trainable distribution parameters, and memory reset
belong to the caller. ``params = actor.init(key, obs, memory)`` returns the full Flax
variables dictionary; pass it unchanged to ``actor.apply(params, obs, memory)``.

Architecture references: DiffAero 291ea14196ae, utils/nn.py; DiffPhysDrone
271936190b5c, model.py; Liu et al., Learning to Fly from Point Clouds, section III-C.
These are native Flax implementations, not upstream checkpoint converters.
"""

from collections.abc import Mapping
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn


class _StateMLP(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        for width in (256, 128):
            state = nn.Dense(width, kernel_init=nn.initializers.orthogonal(2**0.5))(state)
            state = nn.elu(nn.LayerNorm(epsilon=1e-5)(state))
        return nn.Dense(self.output_size, kernel_init=nn.initializers.orthogonal(0.01))(state)


class Actor(nn.Module):
    """Actor shared by PPO, APG/BPTT and SHAC; only ``kind`` selects architecture.

    Args:
        kind: ``state``: [256,128] LayerNorm/ELU MLP, four raw roll/pitch/yaw/thrust
            coordinates (caller applies tanh and physical scaling). ``depth``:
            CNN 32/64/128, additive 192-wide state fusion, GRU192, acceleration and
            velocity heads. ``lidar``: masked PointNet 64/128/1024, additive fusion,
            GRU192 and acceleration head. Perception actions are raw three-vectors.

    Call inputs: ``obs['state']`` [B,F]; depth [B,12,16,1], or points [B,N,3]
    and boolean mask [B,N]; explicit memory [B,192]. Returns (raw_action,
    new_memory, aux_velocity). Auxiliary velocity is [B,3], zero for state/lidar.
    State actors return memory unchanged. Reset selected rows externally to zero.
    One parameter tree is shared across the observation batch; use jax.vmap over
    init/apply for independently parameterized seeds.
    """

    kind: Literal["state", "depth", "lidar"] = "state"
    action_size: int = 4
    hidden_size: int = 192

    def initialize_memory(self, batch: int) -> jax.Array:
        """Return actor-owned recurrent state; the state MLP needs no memory."""
        return jnp.zeros((batch, 0 if self.kind == "state" else self.hidden_size))

    @property
    def specification(self) -> dict:
        """Describe network dimensions independently from the sensor and action units."""
        return {"kind": self.kind, "action_size": self.action_size, "hidden_size": self.hidden_size}

    @nn.compact
    def __call__(
        self, obs: Mapping[str, jax.Array], memory: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute raw actions, recurrent memory and auxiliary velocity."""
        if self.kind not in ("state", "depth", "lidar"):
            raise ValueError(f"Unknown actor kind: {self.kind!r}")
        state = obs["state"]
        if self.action_size < 1 or self.hidden_size < 1:
            raise ValueError("Network output and hidden sizes must be positive")
        expected_memory = (state.shape[0], 0 if self.kind == "state" else self.hidden_size)
        if state.ndim != 2 or memory.shape != expected_memory:
            raise ValueError(f"Expected state [B,F] and memory {expected_memory}")
        batch = state.shape[0]
        aux_velocity = jnp.zeros((batch, 3), dtype=state.dtype)
        if self.kind == "state":
            return _StateMLP(self.action_size, name="mlp")(state), memory, aux_velocity

        slope = 0.05 if self.kind == "depth" else 0.01
        if self.kind == "depth":
            features = obs["depth"]
            if features.shape != (batch, 12, 16, 1):
                raise ValueError("Expected preprocessed depth [B,12,16,1]")
            for index, (width, kernel, stride) in enumerate(((32, 2, 2), (64, 3, 1), (128, 3, 1))):
                features = nn.Conv(
                    width,
                    (kernel, kernel),
                    strides=(stride, stride),
                    padding="VALID",
                    use_bias=False,
                    name=f"conv_{index}",
                )(features)
                features = nn.leaky_relu(features, negative_slope=slope)
            features = features.reshape(batch, -1)
        else:
            points, mask = obs["points"], obs["mask"]
            if (
                points.ndim != 3
                or points.shape[0] != batch
                or points.shape[-1] != 3
                or mask.shape != points.shape[:-1]
                or mask.dtype != jnp.bool_
            ):
                raise ValueError("Expected points [B,N,3] and boolean mask [B,N]")
            # Mask before the MLP so NaN/Inf padding cannot poison parameter gradients.
            features = jnp.where(mask[..., None], points, 0)
            for index, width in enumerate((64, 128, 1024)):
                # Keep meter-scale returns from overwhelming state fusion at initialization.
                initializer = nn.initializers.variance_scaling(
                    0.01 if index == 0 else 1.0, "fan_in", "truncated_normal"
                )
                features = nn.Dense(width, kernel_init=initializer, name=f"point_{index}")(features)
                features = nn.leaky_relu(features, negative_slope=slope)
            features = jnp.max(
                jnp.where(mask[..., None], features, -jnp.inf), axis=1, initial=-jnp.inf
            )
            features = jnp.where(jnp.any(mask, axis=1, keepdims=True), features, 0)

        features = nn.Dense(self.hidden_size, use_bias=False, name="perception_projection")(
            features
        )
        state_features = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.variance_scaling(0.25, "fan_in", "truncated_normal"),
            name="state_projection",
        )(state)
        features = nn.leaky_relu(features + state_features, negative_slope=slope)
        new_memory, features = nn.GRUCell(self.hidden_size, name="gru")(memory, features)
        features = nn.leaky_relu(features, negative_slope=slope)
        raw_action = nn.Dense(
            self.action_size,
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(0.01),
            name="action_head",
        )(features)
        if self.kind == "depth":
            aux_velocity = nn.Dense(
                3,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(0.01),
                name="velocity_head",
            )(features)
        return raw_action, new_memory, aux_velocity


class Critic(nn.Module):
    """Independent value MLP with an optional, independently trained sensor encoder.

    By default, only state [B,F] is consumed, preserving existing checkpoints.
    Depth/LiDAR PPO may use its already available measurements for a more
    informative value estimate. Actor and critic never share parameters.
    Privileged mode consumes only the explicit training-only state vector.
    """

    kind: Literal["state", "depth", "lidar", "privileged"] = "state"

    @nn.compact
    def __call__(self, obs: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        """Estimate value using the explicitly configured observation source."""
        key = "privileged_state" if self.kind == "privileged" else "state"
        state = obs[key] if isinstance(obs, Mapping) else obs
        if state.ndim != 2:
            raise ValueError("Expected critic state [B,F]")
        if self.kind == "depth":
            if not isinstance(obs, Mapping):
                raise ValueError("Depth critic requires the depth observation")
            depth = obs["depth"]
            if depth.shape != (state.shape[0], 12, 16, 1):
                raise ValueError("Expected critic depth [B,12,16,1]")
            features = nn.relu(nn.Conv(16, (3, 3), strides=(2, 2), name="value_conv_0")(depth))
            features = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2), name="value_conv_1")(features))
            state = jnp.concatenate([state, jnp.mean(features, axis=(1, 2))], axis=-1)
        elif self.kind == "lidar":
            if not isinstance(obs, Mapping):
                raise ValueError("LiDAR critic requires the point observation")
            points, mask = obs["points"], obs["mask"]
            if points.ndim != 3 or points.shape[-1] != 3 or mask.shape != points.shape[:-1]:
                raise ValueError("Expected critic points [B,N,3] and mask [B,N]")
            features = jnp.where(mask[..., None], points, 0.0)
            features = nn.relu(
                nn.Dense(
                    32,
                    kernel_init=nn.initializers.variance_scaling(
                        0.01, "fan_in", "truncated_normal"
                    ),
                    name="value_point_0",
                )(features)
            )
            features = nn.relu(nn.Dense(64, name="value_point_1")(features))
            features = jnp.max(
                jnp.where(mask[..., None], features, -jnp.inf), axis=1, initial=-jnp.inf
            )
            features = jnp.where(jnp.any(mask, axis=1, keepdims=True), features, 0.0)
            state = jnp.concatenate([state, features], axis=-1)
        elif self.kind not in ("state", "privileged"):
            raise ValueError(f"Unknown critic kind: {self.kind}")
        return _StateMLP(1, name="mlp")(state)[..., 0]


def tanh_gaussian_log_prob(pre_tanh: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Log density of tanh(pre_tanh), summed over the final action dimension.

    ``mean``/``pre_tanh`` are [...,A]; externally trainable ``log_std`` broadcasts
    to them, normally [A]. Store pre_tanh in PPO rollouts: inversion of a saturated
    tanh loses information. The softplus Jacobian stays finite at saturation. This
    is the density in (-1,1), before any physical action scaling.
    """
    normal = -0.5 * ((pre_tanh - mean) * jnp.exp(-log_std)) ** 2
    normal = normal - log_std - 0.5 * jnp.log(2 * jnp.pi)
    log_jacobian = 2 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2 * pre_tanh))
    return jnp.sum(normal - log_jacobian, axis=-1)


def sample_tanh_gaussian(
    key: jax.Array, mean: jax.Array, log_std: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reparameterized sample: (normalized action, pre_tanh, log_probability).

    Initialize log_std separately, e.g. zeros([action_size]), and include it in the
    trainer's optimizer tree. Deterministic evaluation uses tanh(mean).
    """
    pre_tanh = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape, dtype=mean.dtype)
    return jnp.tanh(pre_tanh), pre_tanh, tanh_gaussian_log_prob(pre_tanh, mean, log_std)
