"""Shared JAX numerical operations with explicit derivative conventions."""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def euclidean_norm(value):
    """Exact last-axis norm with the zero subgradient at the origin."""
    return jnp.linalg.norm(value, axis=-1)


@euclidean_norm.defjvp
def _euclidean_norm_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    norm = euclidean_norm(value)
    derivative = jnp.sum(value * tangent, axis=-1) / jnp.where(norm > 0, norm, 1.0)
    return norm, derivative


def quat_to_matrix_xyzw(quat):
    """Convert a unit xyzw quaternion to body-to-world rotation, without normalizing."""
    x, y, z, w = quat
    return jnp.stack(
        [
            jnp.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ]
            ),
            jnp.stack(
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ]
            ),
            jnp.stack(
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ]
            ),
        ]
    )


def pseudo_huber(value):
    """Unit pseudo-Huber: finite derivative at zero and linear growth at large errors."""
    return jnp.sqrt(1.0 + value * value) - 1.0
