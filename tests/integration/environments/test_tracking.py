"""Behavioral checks at the same task interface used by training and evaluation."""

import unittest

import crazyflow  # noqa: F401 — establish SciPy's Array API mode before imports.
import jax
import jax.numpy as jnp
import numpy as np


class TrackingContractTests(unittest.TestCase):
    def make_env(self, **kwargs):
        from tests.helpers.environments import tracking_environment

        env = tracking_environment(device="cpu", **kwargs)
        self.addCleanup(env.close)
        return env

    def test_matches_upstream_figure_eight_before_terminal(self):
        env = self.make_env()
        s = env.reset(jax.random.PRNGKey(7))
        original = env.simulation
        original.sim.data = s.pipeline_state.sim_data
        original._marked_for_reset = jnp.zeros(1, dtype=bool)
        step = jax.jit(env.step)
        for i in range(20):
            action = jnp.asarray([0.01 * np.sin(i / 5), 0.01, 0, 0.2], dtype=jnp.float32)
            s = step(s, action)
            obs, reward, terminal, _, _ = original.step(env.physical_action(action)[None])
            flat = np.concatenate(
                [np.asarray(obs[k][0]) for k in ("pos", "quat", "vel", "ang_vel", "local_samples")]
            )
            np.testing.assert_allclose(s.obs, flat, atol=3e-5)
            self.assertAlmostEqual(float(s.reward), float(reward[0]), places=6)
            self.assertEqual(bool(s.done), bool(terminal[0]))

    def test_reset_is_reproducible_and_seed_sensitive(self):
        env = self.make_env()
        a = env.reset(jax.random.PRNGKey(3))
        b = env.reset(jax.random.PRNGKey(3))
        c = env.reset(jax.random.PRNGKey(4))
        np.testing.assert_array_equal(a.obs, b.obs)
        self.assertFalse(np.array_equal(a.obs, c.obs))

    def test_timeout_keeps_terminal_observation_and_draws_fresh_reset(self):
        from drone_playground.learning.wrappers import wrap_for_training

        env = self.make_env()
        wrapper = wrap_for_training(env, episode_length=2, action_repeat=1)
        s = wrapper.reset(jax.random.split(jax.random.PRNGKey(8), 2))
        initial = np.asarray(s.obs).copy()
        step = jax.jit(wrapper.step)
        action = jnp.tile(env.hover_action, (2, 1))
        s = step(step(s, action), action)
        self.assertTrue(np.all(np.asarray(s.done)))
        self.assertTrue(np.all(np.asarray(s.info["truncation"])))
        self.assertTrue(np.all(np.asarray(s.info["steps"]) == 2))
        self.assertFalse(np.array_equal(initial, np.asarray(s.obs)))
        self.assertFalse(np.array_equal(s.info["terminal_observation"], s.obs))
        s = step(s, action)
        self.assertTrue(np.all(np.asarray(s.info["steps"]) == 1))

    def test_reset_reserves_an_unconsumed_key_for_later_episodes(self):
        from drone_playground.learning.wrappers import wrap_for_training

        env = self.make_env()
        keys = jax.random.split(jax.random.PRNGKey(18), 2)
        wrapper = wrap_for_training(env, episode_length=3)
        state = wrapper.reset(keys)
        children = jax.vmap(jax.random.split)(keys)
        expected = jax.vmap(env.reset)(children[:, 1])
        np.testing.assert_array_equal(state.info["reset_key"], children[:, 0])
        np.testing.assert_array_equal(state.obs, expected.obs)

    def test_local_dynamics_gradient_matches_finite_difference(self):
        env = self.make_env()
        s0 = env.reset(jax.random.PRNGKey(4))

        def loss(thrust):
            action = jnp.array([0.01, -0.01, 0, thrust])
            end, _ = jax.lax.scan(lambda s, _: (env.step(s, action), None), s0, None, length=12)
            return (end.pipeline_state.sim_data.states.pos[0, 0, 2] - 0.9) ** 2

        loss = jax.jit(loss)
        grad = float(jax.grad(loss)(jnp.float32(0.15)))
        fd = (float(loss(jnp.float32(0.153))) - float(loss(jnp.float32(0.147)))) / 0.006
        self.assertGreater(abs(grad), 1e-7)
        self.assertLess(abs(grad - fd) / max(abs(fd), 1e-6), 0.04)

    def test_random_reference_matches_lsy_scipy_construction(self):
        from scipy.interpolate import CubicSpline

        from drone_playground.references import random_trajectory

        seed, duration, freq = 21, 15.0, 50
        takeoff = np.array([-1.5, 1.0, 0.07])
        knots = np.random.RandomState(seed).uniform(-1, 1, (10, 3))
        knots = knots * [1.2, 1.2, 0.5] + 0.3 * takeoff + [0, 0, 0.7]
        knots[:3] = [[-1.5, 1.0, 0.07], [-1.0, 0.55, 0.4], [0.3, 0.35, 0.7]]
        expected = CubicSpline(
            np.linspace(0, duration, 10), knots, bc_type=((1, np.array([0, 0, 0.4])), "not-a-knot")
        )
        actual = random_trajectory(seed, duration=duration, freq=freq)
        np.testing.assert_allclose(actual, expected(np.linspace(0, duration, 750)), atol=1e-6)

    def test_random_task_reset_and_batch_shapes(self):
        env = self.make_env(task="random", reference_seed=100, reference_count=8)
        keys = jax.random.split(jax.random.PRNGKey(19), 4)
        states = jax.jit(jax.vmap(env.reset))(keys)
        self.assertEqual(states.obs.shape, (4, 43))
        np.testing.assert_allclose(
            states.pipeline_state.sim_data.states.pos[:, 0, 0],
            np.tile([-1.5, 1.0, 0.07], (4, 1)),
            atol=1e-6,
        )
        states = jax.jit(jax.vmap(env.step))(states, jnp.tile(env.hover_action, (4, 1)))
        self.assertTrue(np.isfinite(np.asarray(states.obs)).all())


if __name__ == "__main__":
    unittest.main()
