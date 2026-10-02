"""A saved policy must reproduce its actions in a fresh loader."""

import unittest

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


class CheckpointTests(unittest.TestCase):
    def test_normal_policy_uses_configured_state_independent_initial_std(self):
        from drone_playground.networks.factory import network_factory

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

if __name__ == "__main__":
    unittest.main()
