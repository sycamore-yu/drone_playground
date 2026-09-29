"""Parallel obstacle intersections preserve the exact primitive geometry and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.environments.sensors.rays import cast_rays


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("rotated", [False, True])
def test_chunked_intersections_match_serial_hits_and_derivatives(block_size, rotated):
    kind = jnp.array([1, 2, 3, 4, 2])
    size = jnp.array(
        [[0.5, 4.0, 0.0], [0.5, 0.6, 0.9], [0.8, 0.8, 0.8], [0.4, 1.0, 0.0], [1.0, 1.0, 1.0]]
    )
    centres = jnp.array(
        [[4.0, 0.0, 2.0], [6.0, 2.0, 2.0], [3.0, -3.0, 2.0], [7.0, -2.0, 2.0], [1.0, 0.0, 2.0]]
    )
    active = jnp.array([True, True, True, True, False])
    angles = jnp.arange(33) * (2 * jnp.pi / 33)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles), -0.03 * jnp.ones_like(angles)], -1)
    rotations = jnp.tile(jnp.eye(3)[None], (5, 1, 1)) if rotated else None
    if rotated:
        a = 0.3
        rotations = rotations.at[1].set(
            jnp.array([[jnp.cos(a), -jnp.sin(a), 0], [jnp.sin(a), jnp.cos(a), 0], [0, 0, 1]])
        )

    def measured(position, chunk):
        d = cast_rays(
            kind,
            size,
            centres,
            active,
            jnp.broadcast_to(position, directions.shape),
            directions,
            jnp.array([-10.0, -10.0, 0.0]),
            jnp.array([20.0, 10.0, 6.0]),
            rotations=rotations,
            obstacle_batch_size=chunk,
        )
        return jnp.where(jnp.isfinite(d), d, 0.0)

    p = jnp.array([0.5, 0.1, 2.2])
    a, b = measured(p, 1), measured(p, block_size)
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
    ga, gb = (
        jax.jacrev(lambda p: measured(p, 1))(p),
        jax.jacrev(lambda p: measured(p, block_size))(p),
    )
    assert np.isfinite(ga).all() and np.isfinite(gb).all()
    np.testing.assert_allclose(ga, gb, atol=1e-4, rtol=1e-4)


def test_zero_obstacle_bank_still_intersects_ground_when_batching():
    result = cast_rays(
        jnp.zeros(0, int),
        jnp.zeros((0, 3)),
        jnp.zeros((0, 3)),
        jnp.zeros(0, bool),
        jnp.array([[0.0, 0.0, 2.0]]),
        jnp.array([[0.0, 0.0, -1.0]]),
        jnp.array([-5.0, -5.0, 0.0]),
        jnp.array([5.0, 5.0, 6.0]),
        obstacle_batch_size=16,
    )
    np.testing.assert_array_equal(result, [2.0])
