"""Range-sensor derivatives stay finite across masked misses and padded primitives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.environments.scenes.navigation import (
    KIND_BOX,
    KIND_CAPSULE,
    KIND_CYLINDER,
    KIND_SPHERE,
)
from drone_playground.environments.sensors.rays import cast_rays


@pytest.mark.parametrize("kind", [KIND_BOX, KIND_CYLINDER, KIND_SPHERE, KIND_CAPSULE])
def test_masked_ray_misses_have_finite_position_derivatives(kind):
    directions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    def measured(position):
        distances = cast_rays(
            jnp.array([kind, 0]),
            jnp.array([[0.5, 1.0, 0.5], [0.0, 0.0, 0.0]]),
            jnp.array([[3.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
            jnp.array([True, False]),
            jnp.broadcast_to(position, (3, 3)),
            directions,
            jnp.array([-10.0, -10.0, 0.0]),
            jnp.array([10.0, 10.0, 10.0]),
            True,
        )
        return jnp.where(jnp.isfinite(distances), distances, 0.0)

    position = jnp.array([0.0, 0.0, 2.0])
    actual = jax.jit(jax.jacrev(measured))(position)
    assert np.isfinite(np.asarray(actual)).all(), actual
    eps = 0.001
    differences = jnp.stack(
        [
            (measured(position + eps * jnp.eye(3)[i]) - measured(position - eps * jnp.eye(3)[i]))
            / (2 * eps)
            for i in range(3)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(actual, differences, atol=0.001)
