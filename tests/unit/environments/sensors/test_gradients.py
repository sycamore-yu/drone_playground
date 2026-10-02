"""Range-sensor derivatives stay finite across masked misses and padded primitives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.environments.scenes.geometry import (
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


def test_detached_lidar_has_identical_measurements_and_explicit_zero_state_derivative():
    from drone_playground.environments.sensors.lidar import Mid360Lidar, cast_lidar
    from tests.helpers.scenes import synthetic_bank

    bank = synthetic_bank([dict(kind=KIND_BOX, size=(0.5, 2.0, 2.0), origin=(4.0, 0.0, 2.0))])
    direct = Mid360Lidar()
    detached = Mid360Lidar(state_gradient="detached")
    position, quat = jnp.array([1.0, 0.0, 2.0]), jnp.array([0.0, 0.0, 0.0, 1.0])

    def measure(sensor, p):
        return cast_lidar(sensor, bank, jnp.int32(0), p, quat, jnp.float32(0), 0)

    a, b = measure(direct, position), measure(detached, position)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)
    grad = jax.jacrev(lambda p: measure(detached, p).points_sensor)(position)
    np.testing.assert_array_equal(grad, 0.0)
    full = jax.jacrev(lambda p: measure(direct, p).points_sensor)(position)
    assert np.isfinite(full).all() and np.linalg.norm(full) > 0
    assert detached.calibration()["state_gradient"] == "detached"


def test_lidar_derivative_contract_rejects_unknown_rules():
    from drone_playground.environments.sensors.lidar import Mid360Lidar

    with pytest.raises(ValueError):
        Mid360Lidar(state_gradient="implicit")
