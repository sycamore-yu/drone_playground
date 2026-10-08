"""Explicit directional nearest-range encoding of measured LiDAR points only.

The representation retains frame order and validity. It introduces no obstacle
state, target route, safety filter or planner. Angular bin assignments are
discrete measurement processing; valid range values retain finite derivatives.
"""

import math

import jax
import jax.numpy as jnp


def polar_range_features(frames, azimuth_bins=24, elevation_bins=3, range_scale=40.0):
    """Bin valid ray returns into azimuth and elevation range features."""
    if min(azimuth_bins, elevation_bins) < 1 or not math.isfinite(range_scale) or range_scale <= 0:
        raise ValueError("Range encoding requires positive bin counts and physical range scale")
    if frames.shape[-1] != 5:
        raise ValueError("Range encoding consumes XYZ, range and validity")
    leading, points = frames.shape[:-2], frames.shape[-2]
    flat = frames.reshape((-1, points, 5))
    xyz = jax.lax.stop_gradient(flat[..., :3])
    valid = jax.lax.stop_gradient(flat[..., 4] > 0.5)
    # A fixed direction for invalid zero points avoids the atan2(0,0)
    # singularity. Bin indices carry no spatial derivative by definition.
    xyz = jnp.where(valid[..., None], xyz, jnp.array([1.0, 0.0, 0.0]))
    azimuth = jnp.arctan2(xyz[..., 1], xyz[..., 0])
    elevation = jnp.arctan2(
        xyz[..., 2],
        jnp.sqrt(jnp.maximum(jnp.sum(xyz[..., :2] ** 2, axis=-1), 1e-12)),
    )
    ai = (
        jnp.floor((azimuth + jnp.pi) / (2 * jnp.pi) * azimuth_bins).astype(jnp.int32) % azimuth_bins
    )
    ei = jnp.clip(
        jnp.floor((elevation + jnp.pi / 2) / jnp.pi * elevation_bins),
        0,
        elevation_bins - 1,
    ).astype(jnp.int32)
    bins = ei * azimuth_bins + ai
    distance = jnp.maximum(flat[..., 3] * range_scale, 0.0)
    inverse = jnp.where(valid, 1.0 / (1.0 + distance), 0.0)
    count = azimuth_bins * elevation_bins

    def pool(indices, signal, mask):
        closest = jnp.zeros(count).at[indices].max(signal)
        observed = jnp.zeros(count).at[indices].max(mask.astype(jnp.float32))
        return jnp.stack([closest, observed], axis=-1).reshape(count * 2)

    return jax.vmap(pool)(bins, inverse, valid).reshape(*leading, count * 2)
