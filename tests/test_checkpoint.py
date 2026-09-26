"""A saved policy must reproduce its actions in a fresh loader."""

import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs


class CheckpointTests(unittest.TestCase):
    def test_normal_policy_uses_configured_state_independent_initial_std(self):
        from drone_playground.learning.networks import network_factory

        net = network_factory(
            {
                "algorithm": "ppo",
                "distribution_type": "normal",
                "init_noise_std": 0.367879,
                "hidden_sizes": [32, 32],
            }
        )(43, 4)
        params = net.policy_network.init(jax.random.PRNGKey(5))
        logits = net.policy_network.apply(None, params, jnp.zeros((2, 43)))
        dist = net.parametric_action_distribution.create_dist(logits)
        np.testing.assert_allclose(dist.scale, 0.367879, atol=1e-6)
        np.testing.assert_allclose(dist.loc, 0.0, atol=1e-6)

    def test_saved_brax_policy_reproduces_actions_and_normalization(self):
        from drone_playground.learning.networks import network_factory
        from drone_playground.runs.checkpoints import load_policy, save_policy

        cfg = {
            "algorithm": "ppo",
            "task": "figure8",
            "dynamics": "so_rpy",
            "normalize_observations": False,
            "hidden_sizes": [32, 32],
            "init_noise_std": 0.37,
        }
        factory = network_factory(cfg)
        net = factory(43, 4)
        params = (
            running_statistics.init_state(specs.Array((43,), jnp.float32)),
            net.policy_network.init(jax.random.PRNGKey(1)),
            net.value_network.init(jax.random.PRNGKey(2)),
        )
        from brax.training.agents.ppo import networks

        obs = jnp.ones((3, 43))
        expected = networks.make_inference_fn(net)(params, deterministic=True)(
            obs, jax.random.PRNGKey(0)
        )[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = save_policy(Path(tmp), params, cfg, step=123)
            make_policy, loaded, metadata = load_policy(path)
            actual = make_policy(loaded, deterministic=True)(obs, jax.random.PRNGKey(0))[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(metadata["step"], 123)
        self.assertEqual(metadata["checkpoint_kind"], "inference-parameters")


if __name__ == "__main__":
    unittest.main()
