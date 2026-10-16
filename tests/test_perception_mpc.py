"""JAX-native, known-landmark predictive sampling perception objective."""

import jax.numpy as jnp
import numpy as np

from drone_playground.simulation.perception_mpc import (
    landmark_image_coordinates,
    perception_cost,
)


def test_view_geometry_uses_actual_quaternion_and_forward_body_x():
    """A point directly ahead is centered; one behind is invisible."""
    center, depth = landmark_image_coordinates(
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([0.0, 0.0, 0.0, 1.0]),
        jnp.array([3.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(center, [0.0, 0.0])
    assert float(depth) == 3.0
    off, back = landmark_image_coordinates(
        jnp.zeros(3), jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.array([-3.0, 0.0, 0.0])
    )
    visible = perception_cost(center, depth, center, horizontal_tangent=0.95, vertical_tangent=0.5)
    invisible = perception_cost(off, back, center, horizontal_tangent=0.95, vertical_tangent=0.5)
    assert float(visible[0]) == 0.0
    assert float(invisible[0]) > 4.0


def test_image_velocity_penalty_is_zero_for_stationary_projection():
    """Moving imagery is measured independently from geometric field-of-view."""
    projection = jnp.array([0.1, -0.2])
    static = perception_cost(
        projection, 2.0, projection, horizontal_tangent=0.9, vertical_tangent=0.7
    )
    moving = perception_cost(
        projection, 2.0, projection + 0.2, horizontal_tangent=0.9, vertical_tangent=0.7
    )
    assert float(static[1]) == 0.0
    assert float(moving[1]) > 0.01
