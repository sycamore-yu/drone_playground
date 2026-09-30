"""The point-wise encoder, additive state fusion and GRU in paper Sec. III-C."""

import math

import jax.numpy as jnp
from flax import linen as nn


class PointCloudPolicy(nn.Module):
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
