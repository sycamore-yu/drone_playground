"""Batched command history on the existing physical clock."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


def delay_range(seconds):
    """Validate a fixed delay or a per-episode uniform interval, in seconds."""
    bounds = np.asarray([seconds, seconds] if np.ndim(seconds) == 0 else seconds, float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] <= bounds[1]:
        raise ValueError("Delay must be nonnegative seconds or an ordered [low, high] interval")
    return tuple(float(x) for x in bounds)


@struct.dataclass
class CommandState:
    """History and last applied command for each independent world."""

    values: jax.Array
    inputs: jax.Array
    due: jax.Array
    issued: jax.Array
    delay_ticks: jax.Array
    requested_delay: jax.Array
    applied: jax.Array
    applied_input: jax.Array

    @classmethod
    def create(cls, key, native, inputs, bounds, physics_hz, method_hz):
        """Initialize with an explicitly supplied neutral command."""
        batch = native.shape[0]
        capacity = math.ceil(bounds[1] * method_hz) + 2
        requested = jax.random.uniform(key, (batch,), minval=bounds[0], maxval=bounds[1])
        ticks = jnp.ceil(requested * physics_hz - 1e-6).astype(jnp.int32)
        return cls(
            jnp.broadcast_to(native[:, None], (batch, capacity, native.shape[-1])),
            jnp.broadcast_to(inputs[:, None], (batch, capacity, inputs.shape[-1])),
            jnp.full((batch, capacity), -1, jnp.int32),
            jnp.zeros((batch, capacity), jnp.int32),
            ticks,
            requested,
            native,
            inputs,
        )

    def push(self, native, inputs, tick, active):
        """Stage one decision without changing finished worlds."""

        def append(old, new):
            updated = jnp.concatenate([old[:, 1:], new[:, None]], axis=1)
            mask = active.reshape((len(active),) + (1,) * (updated.ndim - 1))
            return jnp.where(mask, updated, old)

        return self.replace(
            values=append(self.values, native),
            inputs=append(self.inputs, inputs),
            due=append(self.due, tick + self.delay_ticks),
            issued=append(self.issued, tick),
        )

    def at(self, tick):
        """Return the newest delivered command and its issue tick."""
        indices = jnp.arange(self.due.shape[1])[None]
        latest = jnp.max(jnp.where(self.due <= tick[:, None], indices, 0), axis=1)
        row = jnp.arange(len(tick))
        return self.values[row, latest], self.inputs[row, latest], self.issued[row, latest]
