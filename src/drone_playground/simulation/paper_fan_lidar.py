"""Fan et al. (RA-L 2025) LiDAR history encoding, independent of PPO.

A level-frame 3D hit cloud is reduced to 36 azimuth range bins; the newest
36 frames form a 36 x 36 obstacle map.  Invalid returns mean maximum range.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FanLidarConfig:
    """Paper-reported spatial and temporal compression settings."""

    sectors: int = 36
    history_frames: int = 36
    max_range_m: float = 10.0
    half_height_m: float = 1.0
    raw_history_frames: int = 1

    def __post_init__(self):
        """Reject nonphysical and undefined range-map settings."""
        if self.sectors < 2 or self.history_frames < 1:
            raise ValueError("Sector and history counts must be positive")
        if self.max_range_m <= 0 or self.half_height_m <= 0:
            raise ValueError("Range and altitude bounds must be positive")
        if type(self.raw_history_frames) is not int or self.raw_history_frames < 1:
            raise ValueError("Raw point history length must be a positive integer")


def encode_range_ring(
    points_body: jax.Array,
    valid_mask: jax.Array,
    *,
    rotation_body_to_level: jax.Array | None = None,
    sectors: int = 36,
    max_range_m: float = 10.0,
    half_height_m: float = 1.0,
) -> jax.Array:
    """Return [B,sectors] with each direction's nearest valid normalized range."""
    points = jnp.asarray(points_body)
    mask = jnp.asarray(valid_mask)
    if points.ndim != 3 or points.shape[-1] != 3 or mask.shape != points.shape[:2]:
        raise ValueError("Expected points [B,N,3] and mask [B,N]")
    if mask.dtype != jnp.bool_:
        raise ValueError("Expected boolean point mask")
    if sectors < 2 or max_range_m <= 0 or half_height_m <= 0:
        raise ValueError("Invalid range ring settings")
    finite = jnp.all(jnp.isfinite(points), axis=-1)
    points = jnp.where((mask & finite)[..., None], points, 0.0)
    if rotation_body_to_level is not None:
        rotation = jnp.asarray(rotation_body_to_level)
        if rotation.shape != (points.shape[0], 3, 3):
            raise ValueError("Expected body-to-level rotation [B,3,3]")
        points = jnp.einsum("bij,bnj->bni", rotation, points)
    distance = jnp.linalg.norm(points, axis=-1)
    valid = mask & finite & (jnp.abs(points[..., 2]) <= half_height_m) & (distance < max_range_m)
    angle = jnp.arctan2(points[..., 1], points[..., 0])
    sectors_index = jnp.mod(
        jnp.floor((angle + jnp.pi) * (sectors / (2 * jnp.pi))).astype(jnp.int32),
        sectors,
    )
    candidate = jnp.where(valid, distance, max_range_m)
    bins = jnp.full((points.shape[0], sectors), max_range_m, dtype=points.dtype)
    batch = jnp.broadcast_to(jnp.arange(points.shape[0])[:, None], sectors_index.shape)
    nearest = bins.at[batch, sectors_index].min(candidate)
    return jnp.clip(nearest / max_range_m, 0.0, 1.0)


def empty_history(batch: int, *, frames: int = 36, sectors: int = 36) -> jax.Array:
    """Initialize empty scans as maximum range, not an obstacle at zero."""
    return jnp.ones((batch, frames, sectors), dtype=jnp.float32)


def push_point_window(history, history_mask, world_points, valid):
    """Keep raw hits in world coordinates, before spatial range compression."""
    if history.shape[0] != world_points.shape[0] or history.shape[2:] != world_points.shape[1:]:
        raise ValueError("Raw point window and current cloud dimensions disagree")
    if history_mask.shape != history.shape[:-1] or valid.shape != world_points.shape[:-1]:
        raise ValueError("Raw point masks must match their point arrays")
    finite = valid & jnp.all(jnp.isfinite(world_points), axis=-1)
    return (
        jnp.concatenate(
            (jnp.where(finite[..., None], world_points, 0.0)[:, None], history[:, :-1]), axis=1
        ),
        jnp.concatenate((finite[:, None], history_mask[:, :-1]), axis=1),
    )


def push_history(history: jax.Array, scan: jax.Array) -> jax.Array:
    """Insert the newest scan at index zero."""
    if history.ndim != 3 or scan.shape != (history.shape[0], history.shape[2]):
        raise ValueError("History and range ring shapes disagree")
    return jnp.concatenate((scan[:, None], history[:, :-1]), axis=1)


def clear_done_histories(history: jax.Array, done: jax.Array) -> jax.Array:
    """Prevent a terminated episode from leaking its LiDAR history to the next."""
    if done.shape != (history.shape[0],):
        raise ValueError("Expected per-world done mask")
    return jnp.where(done[:, None, None], jnp.ones_like(history), history)
