"""SHAC objectives and actual Crazyflow updates at the training interface."""

import importlib
import importlib.util
import unittest

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


class SHACTests(unittest.TestCase):
    def module(self):
        name = "drone_playground.learning.algorithms.shac"
        self.assertIsNotNone(importlib.util.find_spec(name), "SHAC implementation must exist")
        return importlib.import_module(name)

    def test_lambda_returns_terminate_and_timeout(self):
        m = self.module()
        rewards = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        values = jnp.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        done = jnp.array([[False, False], [True, True], [False, False]])
        terminal = jnp.array([[False, False], [True, False], [False, False]])
        actual = m.lambda_returns(rewards, values, done, terminal, 0.9, 0.5)
        # At a timeout, bootstrap from its pre-reset state, never the next episode.
        expected = [
            [1 + 0.9 * (0.5 * 10 + 0.5 * 2), 1 + 0.9 * (0.5 * 10 + 0.5 * 20)],
            [2, 20],
            [30, 30],
        ]
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_bootstrap_freezes_weights_but_preserves_state_derivative(self):
        m = self.module()

        def value(weights, state):
            return jnp.sum(weights * state, axis=-1)

        def f(w, x):
            return m.bootstrap_value(value, w, x, jnp.array(False))

        w, x = jnp.array([2.0, 3.0]), jnp.array([4.0, 5.0])
        np.testing.assert_array_equal(jax.grad(f, 0)(w, x), jnp.zeros(2))
        np.testing.assert_allclose(jax.grad(f, 1)(w, x), w)
        np.testing.assert_array_equal(
            jax.grad(lambda v: m.bootstrap_value(value, w, v, jnp.array(True)))(x), jnp.zeros(2)
        )

    def test_discounted_segments_close_on_termination_and_window_end(self):
        m = self.module()
        r = jnp.array([[1.0], [2.0], [4.0]])
        v = jnp.array([[10.0], [0.0], [3.0]])
        d = jnp.array([[False], [True], [False]])
        # First segment 1 + .9*2, second 4 + .9*3; mean over 3 transitions.
        self.assertAlmostEqual(
            float(m.segment_objective(r, v, d, 0.9)), -(1 + 1.8 + 4 + 2.7) / 3, places=6
        )
