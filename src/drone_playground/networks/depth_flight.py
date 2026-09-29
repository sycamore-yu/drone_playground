"""JAX depth/GRU actor for the differentiable-flight navigation adaptation.

Architecture reference: Learning vision-based agile flight via differentiable
physics (Nature Machine Intelligence, 2025). This is an independent Flax
implementation; it does not load or redistribute the upstream PyTorch code.
"""

import jax.numpy as jnp
from flax import linen as nn


def inverse_depth_image(depth, valid, far_m=10.0):
    """Metric 48x64 depth -> closest-return inverse depth at 12x16."""
    if depth.shape[-2:] != (48, 64) or valid.shape != depth.shape:
        raise ValueError("Depth-flight input must be [...,48,64] metric depth plus validity")
    if far_m <= 0.3:
        raise ValueError("Inverse-depth far range must exceed the 0.3m preprocessing floor")
    signal = 3.0 / jnp.clip(jnp.where(valid, depth, far_m), 0.3, far_m) - 0.6
    return nn.max_pool(signal[..., None], (4, 4), (4, 4), padding="VALID")


class DepthFlightPolicy(nn.Module):
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
        return predictions[..., :3] - predictions[..., 3:], memory
