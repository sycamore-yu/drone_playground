"""Configure initialization without changing the point-cloud policy architecture."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.networks.pointcloud import PointCloudPolicy


def inputs():
    return (jnp.ones((2, 5, 3)), jnp.ones((2, 5), bool),
            jnp.ones((2, 10)), jnp.zeros((2, 192)))


def test_small_head_changes_only_initial_acceleration_weights():
    key = jax.random.PRNGKey(18)
    default = PointCloudPolicy(point_input_scale=.1)
    small = PointCloudPolicy(point_input_scale=.1, output_init_scale=.0001)
    original = default.init(key, *inputs())
    changed = small.init(key, *inputs())
    for name in original["params"]:
        if name == "acceleration":
            np.testing.assert_allclose(changed["params"][name]["kernel"],
                                       original["params"][name]["kernel"] * .01,
                                       rtol=1e-5, atol=1e-8)
            np.testing.assert_array_equal(changed["params"][name]["bias"],
                                          original["params"][name]["bias"])
        else:
            for a, b in zip(jax.tree.leaves(original["params"][name]),
                            jax.tree.leaves(changed["params"][name]), strict=True):
                np.testing.assert_array_equal(a, b)
    first, _ = default.apply(original, *inputs())
    second, _ = small.apply(changed, *inputs())
    np.testing.assert_allclose(second, first * .01, rtol=1e-5, atol=1e-7)
    assert first.shape == second.shape == (2, 3)
    # Initializer configuration never changes execution of frozen parameters.
    reloaded, _ = small.apply(original, *inputs())
    np.testing.assert_array_equal(reloaded, first)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_initialization_scale_fails_before_training(value):
    policy = PointCloudPolicy(output_init_scale=value)
    with pytest.raises(ValueError, match="Output initialization scale"):
        policy.init(jax.random.PRNGKey(0), *inputs())
