"""Declared point-coordinate conditioning preserves the full paper encoder."""

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.networks.pointcloud import PointCloudPolicy


def test_point_conditioning_is_exact_explicit_input_rescaling():
    reference = PointCloudPolicy(hidden_size=8, point_channels=(8, 16))
    conditioned = PointCloudPolicy(hidden_size=8, point_channels=(8, 16), point_input_scale=0.02)
    points = jax.random.normal(jax.random.PRNGKey(3), (2, 12, 3)) * 50
    valid = jnp.ones((2, 12), bool).at[1].set(False)
    proprio, hidden = jnp.ones((2, 10)), jnp.zeros((2, 8))
    params = reference.init(jax.random.PRNGKey(4), points, valid, proprio, hidden)
    expected = reference.apply(params, points * 0.02, valid, proprio, hidden)
    actual = conditioned.apply(params, points, valid, proprio, hidden)
    for one, two in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(one, two)
    unchanged = PointCloudPolicy(hidden_size=8, point_channels=(8, 16), point_input_scale=1.0)
    for one, two in zip(
        reference.apply(params, points, valid, proprio, hidden),
        unchanged.apply(params, points, valid, proprio, hidden),
        strict=True,
    ):
        np.testing.assert_array_equal(one, two)
