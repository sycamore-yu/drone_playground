"""SHAC objectives and actual Crazyflow updates at the training interface."""

import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


class SHACTests(unittest.TestCase):
    def module(self):
        name = "drone_playground.learning.shac"
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

    def test_real_actor_critic_updates_and_snapshot_budget(self):
        m = self.module()
        from drone_playground.tasks.tracking import TrackingEnv

        env = TrackingEnv(device="cpu")
        self.addCleanup(env.close)
        snapshots = []
        make_policy, params, result = m.train(
            env,
            dict(
                policy_updates=2,
                num_envs=2,
                horizon_length=4,
                num_evals=3,
                hidden_sizes=[16, 16],
                learning_rate=0.001,
                critic_learning_rate=0.001,
                critic_updates=2,
                normalize_observations=False,
                seed=5,
            ),
            policy_params_fn=lambda step, maker, p: snapshots.append(step),
        )
        self.assertEqual(result["actual_steps"], 16)
        self.assertGreater(result["critic_parameter_delta_l2"], 0)
        self.assertGreater(result["actor_parameter_delta_l2"], 0)
        self.assertEqual(snapshots, [0, 8, 16])
        action, _ = make_policy(params, deterministic=True)(
            env.reset(jax.random.PRNGKey(4)).obs, jax.random.PRNGKey(0)
        )
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.isfinite(action).all())

    def test_public_checkpoint_uses_same_brax_inference_contract(self):
        self.module()
        from brax.training.acme import running_statistics, specs

        from drone_playground.learning.train import load_policy, network_factory, save_policy

        config = dict(algorithm="shac", hidden_sizes=[16, 16], normalize_observations=False)
        net = network_factory(config)(43, 4)
        p = (
            running_statistics.init_state(specs.Array((43,), jnp.float32)),
            net.policy_network.init(jax.random.PRNGKey(1)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_policy(Path(directory), p, config, 16)
            maker, loaded, meta = load_policy(path)
            a, _ = maker(loaded, deterministic=True)(jnp.zeros(43), jax.random.PRNGKey(0))
            self.assertTrue(np.isfinite(a).all())
            self.assertEqual(meta["config"]["algorithm"], "shac")

    def test_full_state_resume_matches_continuous_updates(self):
        m = self.module()
        from drone_playground.tasks.tracking import TrackingEnv

        env = TrackingEnv(device="cpu")
        self.addCleanup(env.close)
        config = dict(
            task="figure8",
            dynamics="so_rpy",
            drone="cf2x_L250",
            policy_updates=2,
            num_envs=2,
            horizon_length=4,
            num_evals=3,
            hidden_sizes=[16, 16],
            learning_rate=0.001,
            critic_learning_rate=0.001,
            critic_updates=2,
            normalize_observations=False,
            seed=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, continuous, _ = m.train(env, config, state_directory=directory)
            saved = directory / "update-0000001.pkl"
            loaded, meta = m.load_training_state(saved)
            self.assertEqual(int(loaded.updates), 1)
            self.assertTrue(meta["typed_key_paths"])
            _, resumed, result = m.train(env, config, restore_state=saved)
            self.assertEqual(result["actual_steps"], 16)
            for a, b in zip(jax.tree.leaves(continuous), jax.tree.leaves(resumed)):
                np.testing.assert_array_equal(a, b)
