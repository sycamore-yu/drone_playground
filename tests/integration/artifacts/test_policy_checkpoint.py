"""Saved-policy reconstruction through the composed training contract."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from drone_playground.configuration import compose_experiment
from drone_playground.learning.brax_configuration import native_training_config
from drone_playground.learning.checkpointing import save_policy
from drone_playground.learning.inference import load_policy
from drone_playground.networks.factory import network_factory


def test_saved_brax_policy_reproduces_actions_and_normalization():
    composed = compose_experiment("control/ppo", "tracking")
    composed["network"].update(
        normalize_observations=False,
        hidden_sizes=[32, 32],
        distribution_type="tanh_normal",
        init_noise_std=0.37,
    )
    config = native_training_config(composed)
    networks = network_factory(config)(43, 4)
    params = (
        running_statistics.init_state(specs.Array((43,), jnp.float32)),
        networks.policy_network.init(jax.random.PRNGKey(1)),
        networks.value_network.init(jax.random.PRNGKey(2)),
    )
    obs = jnp.ones((3, 43))
    expected = ppo_networks.make_inference_fn(networks)(params, deterministic=True)(
        obs, jax.random.PRNGKey(0)
    )[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = save_policy(Path(tmp), params, config, step=123)
        make_policy, loaded, metadata = load_policy(path)
        actual = make_policy(loaded, deterministic=True)(obs, jax.random.PRNGKey(0))[0]
    np.testing.assert_array_equal(actual, expected)
    assert metadata["step"] == 123
    assert metadata["checkpoint_kind"] == "inference-parameters"
