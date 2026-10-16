"""Versioned research actor heads for Fan (2025) and VisFly (2025).

Fan does not publish every ResNet layer: the residual-block widths below are
an explicit reconstruction, never a claimed released original checkpoint.
VisFly branches follow official 6/12/18 CNN + 128 feature, state/target
128/64 MLP, and 64/64 PPO heads.
"""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn


def _torch_initializers(fan_in):
    """Match torch Linear/Conv reset_parameters with a=sqrt(5), including bias."""
    bound = 1.0 / math.sqrt(fan_in)

    def initialize(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return {"kernel_init": initialize, "bias_init": initialize}


def _visfly_dense(x, width, initialization, name=None):
    kwargs = _torch_initializers(x.shape[-1]) if initialization == "pytorch" else {}
    return nn.Dense(width, name=name, **kwargs)(x)


class _FanResidual(nn.Module):
    """Small residual block with independently selectable channel count."""

    width: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        shortcut = x
        x = nn.relu(nn.Conv(self.width, (3, 3), strides=(self.stride, self.stride))(x))
        x = nn.Conv(self.width, (3, 3), padding="SAME")(x)
        if shortcut.shape != x.shape:
            shortcut = nn.Conv(self.width, (1, 1), strides=(self.stride, self.stride))(shortcut)
        return nn.relu(shortcut + x)


class FanFeatures(nn.Module):
    """36 historical LiDAR scans + 4D planar state + 2D relative target."""

    @nn.compact
    def __call__(self, obs):
        """Encode the named paper observation into its policy or value features."""
        image = obs["fan_map"]
        state = obs["fan_state"]
        command = obs["fan_command"]
        batch = image.shape[0]
        if image.shape != (batch, 36, 36, 1):
            raise ValueError("Fan paper LiDAR map must be [B,36,36,1]")
        if state.shape != (batch, 4) or command.shape != (batch, 2):
            raise ValueError("Fan paper state/command require 4 and 2 features")
        x = nn.relu(nn.Conv(16, (3, 3), strides=(2, 2))(image))
        for width, stride in ((16, 1), (32, 2), (64, 2)):
            x = _FanResidual(width, stride)(x)
        x = x.reshape(batch, -1)
        x = nn.relu(nn.Dense(128, name="obstacle_features")(x))
        kinematics = nn.relu(nn.Dense(64, name="kinematic_features")(state))
        target = nn.relu(nn.Dense(32, name="command_features")(command))
        return jnp.concatenate((x, kinematics, target), axis=-1)


class FanPolicy(nn.Module):
    """Planar acceleration policy reconstructed from Fan paper Fig. 2."""

    action_size: int

    @nn.compact
    def __call__(self, obs):
        """Encode the named paper observation into its policy or value features."""
        if self.action_size != 2:
            raise ValueError("Fan paper action is exactly 2D horizontal acceleration")
        x = FanFeatures()(obs)
        for width in (128, 64):
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(2, kernel_init=nn.initializers.orthogonal(0.01))(x)


class FanValue(nn.Module):
    """Reconstructed critic with the same observation access as the Fan actor."""

    @nn.compact
    def __call__(self, obs):
        """Include goal distance and obstacle history in the learned value."""
        x = FanFeatures()(obs)
        for width in (128, 64):
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(1)(x)[..., 0]


class VisFlyFeatures(nn.Module):
    """Official VisFly cluttered_flight StateTargetImageExtractor (Flax port)."""

    initialization: str = "legacy"

    @nn.compact
    def __call__(self, obs):
        """Encode the named paper observation into its policy or value features."""
        depth = obs["depth"]
        state = obs["visfly_state"]
        target = obs["visfly_target"]
        batch = depth.shape[0]
        if depth.shape != (batch, 64, 64, 1):
            raise ValueError("VisFly depth must be raw 64x64 metric depth [B,64,64,1]")
        if state.shape != (batch, 13) or target.shape != (batch, 3):
            raise ValueError("VisFly state=13 and absolute target=3")
        depth = jnp.where(jnp.isfinite(depth), depth, 0.0)
        for index, (width, kernel) in enumerate(((6, 5), (12, 3), (18, 3))):
            kwargs = (
                _torch_initializers(kernel * kernel * depth.shape[-1])
                if (self.initialization == "pytorch")
                else {}
            )
            depth = nn.relu(
                nn.Conv(width, (kernel, kernel), padding="VALID", name=f"conv_{index}", **kwargs)(
                    depth
                )
            )
        image = nn.relu(
            _visfly_dense(depth.reshape(batch, -1), 128, self.initialization, "depth_projection")
        )
        for prefix, branch in (("state", state), ("target", target)):
            for index, width in enumerate((128, 64)):
                branch = nn.relu(
                    _visfly_dense(branch, width, self.initialization, f"{prefix}_{index}")
                )
            if prefix == "state":
                state_features = branch
            else:
                target_features = branch
        return jnp.concatenate((image, state_features, target_features), axis=-1)


class VisFlyPolicy(nn.Module):
    """SB3-style feedforward policy network for 4D body-rate control."""

    action_size: int
    initialization: str = "legacy"

    @nn.compact
    def __call__(self, obs):
        """Encode the named paper observation into its policy or value features."""
        if self.action_size != 4:
            raise ValueError("VisFly cluttered_flight uses four body-rate controls")
        x = VisFlyFeatures(initialization=self.initialization)(obs)
        for width in (64, 64):
            x = nn.relu(_visfly_dense(x, width, self.initialization))
        return _visfly_dense(x, self.action_size, self.initialization)


class VisFlyValue(nn.Module):
    """Independent value branch with official-style sensor/state/target features."""

    initialization: str = "legacy"

    @nn.compact
    def __call__(self, obs):
        """Encode the named paper observation into its policy or value features."""
        x = VisFlyFeatures(initialization=self.initialization)(obs)
        for width in (64, 64):
            x = nn.relu(_visfly_dense(x, width, self.initialization))
        return _visfly_dense(x, 1, self.initialization)[..., 0]
