"""The point-wise encoder, additive state fusion and GRU in paper Sec. III-C."""

import math

import jax.numpy as jnp
from flax import linen as nn

from drone_playground.actions.decoders import decode_acceleration_velocity


class PointNetGruPolicy(nn.Module):
    hidden_size: int = 192
    point_channels: tuple[int, ...] = (64, 128, 1024)
    negative_slope: float = 0.01
    point_input_scale: float = 1.0
    output_init_scale: float = 1.0

    def setup(self):
        if not math.isfinite(self.point_input_scale) or self.point_input_scale <= 0:
            raise ValueError("Point coordinate scale must be finite and positive")
        if not math.isfinite(self.output_init_scale) or self.output_init_scale <= 0:
            raise ValueError("Output initialization scale must be finite and positive")
        self.point_layers = tuple(
            nn.Dense(width, name=f"point_{index}")
            for index, width in enumerate(self.point_channels)
        )
        self.point_projection = nn.Dense(self.hidden_size, name="point_projection")
        self.state_projection = nn.Dense(self.hidden_size, name="state_projection")
        self.memory = nn.GRUCell(features=self.hidden_size, name="memory")
        self.action_head = nn.Dense(
            3, name="acceleration",
            kernel_init=nn.initializers.variance_scaling(
                self.output_init_scale, "fan_in", "truncated_normal"
            ),
        )

    def encode(self, points, valid):
        """Encode (..., points, 3), independently of point order and point count."""
        if points.shape[-1] != 3 or valid.shape != points.shape[:-1]:
            raise ValueError("Point XYZ and validity shapes disagree")
        if points.shape[-2] < 1:
            raise ValueError("Use a masked placeholder for an empty scan")
        hidden = points * self.point_input_scale
        for layer in self.point_layers:
            hidden = nn.leaky_relu(layer(hidden), negative_slope=self.negative_slope)
        valid = valid.astype(bool)
        pooled = jnp.max(jnp.where(valid[..., None], hidden, -jnp.inf), axis=-2)
        present = jnp.any(valid, axis=-1, keepdims=True)
        pooled = jnp.where(present, pooled, 0.0)
        embedding = self.point_projection(pooled)
        return jnp.where(present, embedding, 0.0)

    def act(self, embedding, proprioception, hidden):
        """Advance recurrent memory once per policy tick; embedding may be cached."""
        if proprioception.shape[-1] != 10:
            raise ValueError("Paper proprioception has velocity/target/attitude/radius: 10 fields")
        fused = embedding + self.state_projection(proprioception)
        hidden, _ = self.memory(hidden, fused)
        return self.action_head(hidden), hidden

    def __call__(self, points, valid, proprioception, hidden):
        return self.act(self.encode(points, valid), proprioception, hidden)


def inverse_depth_image(depth, valid, far_m=10.0):
    """Metric 48x64 depth -> closest-return inverse depth at 12x16."""
    if depth.shape[-2:] != (48, 64) or valid.shape != depth.shape:
        raise ValueError("Depth-flight input must be [...,48,64] metric depth plus validity")
    if far_m <= 0.3:
        raise ValueError("Inverse-depth far range must exceed the 0.3m preprocessing floor")
    signal = 3.0 / jnp.clip(jnp.where(valid, depth, far_m), 0.3, far_m) - 0.6
    return nn.max_pool(signal[..., None], (4, 4), (4, 4), padding="VALID")


class DepthCnnGruPolicy(nn.Module):
    hidden_size: int = 192
    input_shape: tuple[int, int] = (48, 64)
    far_m: float = 10.0

    def setup(self):
        self.convolutions = (
            nn.Conv(32, (2, 2), (2, 2), padding="VALID", use_bias=False),
            nn.Conv(64, (3, 3), padding="VALID", use_bias=False),
            nn.Conv(128, (3, 3), padding="VALID", use_bias=False),
        )
        self.image_projection = nn.Dense(self.hidden_size, use_bias=False)
        self.state_projection = nn.Dense(self.hidden_size)
        self.memory = nn.GRUCell(self.hidden_size)
        self.output_projection = nn.Dense(
            6,
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(0.0001, "fan_in", "uniform"),
        )

    def predict(self, depth, valid, proprioception, hidden):
        if proprioception.shape[-1] != 10:
            raise ValueError("Depth flight requires velocity, target velocity, attitude and margin")
        image = inverse_depth_image(depth, valid, self.far_m)
        for convolution in self.convolutions:
            image = nn.leaky_relu(convolution(image), negative_slope=0.05)
        image = jnp.swapaxes(jnp.swapaxes(image, -1, -3), -1, -2)
        image = image.reshape((*image.shape[:-3], -1))
        fused = nn.leaky_relu(
            self.image_projection(image) + self.state_projection(proprioception),
            negative_slope=0.05,
        )
        memory, _ = self.memory(hidden, fused)
        predictions = self.output_projection(nn.leaky_relu(memory, negative_slope=0.05))
        return predictions, memory

    def __call__(self, depth, valid, proprioception, hidden):
        predictions, memory = self.predict(depth, valid, proprioception, hidden)
        # Acceleration feedforward minus predicted velocity with 1/s feedback gain.
        return decode_acceleration_velocity(predictions), memory
