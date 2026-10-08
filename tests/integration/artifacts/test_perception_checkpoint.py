"""Perception checkpoint round-trip through current composition."""

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
from drone_playground.networks.perception import SensorLayout


def test_perception_checkpoint_rebuilds_the_same_network():
    resolved = compose_experiment("navigation/ppo")
    config = native_training_config(resolved)
    layout = SensorLayout.from_dict(config["sensor_layout"])
    networks = network_factory(config)((layout.total_size,), 4)
    normalizer = running_statistics.init_state(specs.Array((layout.total_size,), jnp.float32))
    params = (
        normalizer,
        networks.policy_network.init(jax.random.PRNGKey(1)),
        networks.value_network.init(jax.random.PRNGKey(2)),
    )
    obs = jax.random.uniform(jax.random.PRNGKey(3), (2, layout.total_size))
    expected = ppo_networks.make_inference_fn(networks)(params, deterministic=True)(
        obs, jax.random.PRNGKey(4)
    )[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = save_policy(Path(tmp), params, config, 17)
        maker, loaded, metadata = load_policy(path)
        actual = maker(loaded, deterministic=True)(obs, jax.random.PRNGKey(4))[0]
    np.testing.assert_array_equal(actual, expected)
    assert metadata["step"] == 17
    assert metadata["config"]["env"]["task"]["observation"]["name"] == "navigation_lidar"
