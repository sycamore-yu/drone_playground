"""SHAC checkpoint reconstruction through the public composition contract."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs

from drone_playground.artifacts.checkpoints import load_policy, save_policy
from drone_playground.composition import compose_experiment
from drone_playground.learning.brax_configuration import native_training_config
from drone_playground.networks.factory import network_factory


def test_public_checkpoint_uses_same_brax_inference_contract():
    resolved = compose_experiment("control/shac", "tracking")
    resolved["network"].update(hidden_sizes=[16, 16], normalize_observations=False)
    config = native_training_config(resolved)
    networks = network_factory(config)(43, 4)
    params = (
        running_statistics.init_state(specs.Array((43,), jnp.float32)),
        networks.policy_network.init(jax.random.PRNGKey(1)),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = save_policy(Path(directory), params, config, 16)
        maker, loaded, metadata = load_policy(path)
        action, _ = maker(loaded, deterministic=True)(jnp.zeros(43), jax.random.PRNGKey(0))
    assert np.isfinite(action).all()
    assert metadata["config"]["algorithm"]["name"] == "shac"
