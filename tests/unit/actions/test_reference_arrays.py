"""Reference sampling is shared by native adapters and differentiable policies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_trajectory_sampling_preserves_time_derivatives():
    from drone_playground.references import Trajectory

    coefficients = np.zeros((1, 4, 4))
    coefficients[0, 0] = [1, 2, 3, 4]
    curve = Trajectory(2.0, [1.0], coefficients)
    sample = curve.sample(2.5)
    assert sample["position"][0] == pytest.approx(3.25)
    assert sample["velocity"][0] == pytest.approx(8.0)
    assert sample["acceleration"][0] == pytest.approx(18.0)
    with pytest.raises(ValueError, match="interval"):
        curve.sample(3.1)


def test_trajectory_can_carry_policy_gradients_without_host_conversion():
    from drone_playground.references import Trajectory

    def sample(coefficient):
        values = jnp.zeros((1, 4, 3)).at[0, 0, 2].set(coefficient)
        trajectory = Trajectory(jnp.float32(0), jnp.array([1.0]), values)
        return trajectory.sample(jnp.float32(0.5))["position"][0]

    assert float(jax.jit(jax.grad(sample))(2.0)) == pytest.approx(0.25)


def test_waypoint_is_a_pytree_with_preserved_validity():
    from drone_playground.references import Waypoint

    def make(position):
        return Waypoint(position[None], 0.5, generated_at=2.0, valid_until=3.0)

    output = jax.jit(jax.vmap(make))(jnp.ones((2, 3)))
    assert output.positions.shape == (2, 1, 3)
    np.testing.assert_allclose(output.generated_at, 2.0)
