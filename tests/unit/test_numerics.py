"""Shared mathematical kernels retain the original values and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np


def _original_matrix(q):
    x, y, z, w = q
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_quaternion_values_and_jvp_do_not_add_normalization():
    from drone_playground.numerics import quat_to_matrix_xyzw

    q = jnp.array([0.3, -0.4, 0.1, 0.8])
    tangent = jnp.array([0.1, 0.2, -0.3, 0.4])
    expected = jax.jvp(_original_matrix, (q,), (tangent,))
    actual = jax.jit(lambda q, v: jax.jvp(quat_to_matrix_xyzw, (q,), (v,)))(q, tangent)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(quat_to_matrix_xyzw(jnp.array([0.0, 0.0, 0.0, 1.0])), np.eye(3))


def test_zero_norm_has_finite_zero_derivative():
    from drone_playground.numerics import euclidean_norm

    value, derivative = jax.jvp(euclidean_norm, (jnp.zeros(3),), (jnp.ones(3),))
    assert value == 0
    assert derivative == 0
