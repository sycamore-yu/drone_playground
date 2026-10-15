"""Check the standalone policy-cloning diagnostic without importing its CLI."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

KL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/ppo_rehearsal_diagnostic.py"))[
    "gaussian_kl"
]


def test_teacher_policy_has_zero_kl_and_zero_gradient():
    """The anchor is stationary at its source policy, including exploration scale."""
    mean, log_std = jnp.array([[0.2, -0.4, 0.8]]), jnp.array([-2.5, -1.0, 0.2])
    value, gradients = jax.value_and_grad(
        lambda current_mean, current_std: KL(mean, log_std, current_mean, current_std).mean(),
        argnums=(0, 1),
    )(mean, log_std)
    np.testing.assert_array_equal(value, 0.0)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


def test_policy_shift_has_known_kl_and_a_restoring_mean_gradient():
    """Same unit covariance gives half squared mean distance, with correct KL direction."""
    teacher, current = jnp.zeros((1, 3)), jnp.array([[1.0, -2.0, 0.5]])
    value, gradient = jax.value_and_grad(
        lambda mean: KL(teacher, jnp.zeros(3), mean, jnp.zeros(3)).mean()
    )(current)
    np.testing.assert_allclose(value, 2.625)
    np.testing.assert_allclose(gradient, current)
    assert float(KL(teacher, jnp.zeros(3), teacher, jnp.full(3, jnp.log(2.0))).mean()) > 0
