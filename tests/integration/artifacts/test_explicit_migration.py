"""Historical inference weights migrate only with reviewed configuration and unchanged inputs."""

import copy
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from drone_playground.artifacts.checkpoints import load_policy, save_policy
from drone_playground.artifacts.migration import migrate_checkpoint
from drone_playground.composition import compose_experiment
from drone_playground.learning.brax_configuration import native_training_config
from drone_playground.networks.factory import network_factory


def test_explicit_migration_preserves_source_bytes_and_frozen_action(tmp_path):
    config = compose_experiment("control/bptt", overrides=["network.hidden_sizes=[8,8]"])
    native = native_training_config(config)
    net = network_factory(native)(43, 4)
    parameters = (
        running_statistics.init_state(specs.Array((43,), jnp.float32)),
        net.policy_network.init(jax.random.key(1)),
    )
    source = save_policy(tmp_path / "source", parameters, native, 7)
    maker, params, _ = load_policy(source)
    observation = jnp.ones(43)
    expected = maker(params, deterministic=True)(observation, jax.random.key(2))[0]
    metadata = json.loads(source.with_suffix(".json").read_text())
    old = copy.deepcopy(config)
    old["config_version"] = 3
    old["env"]["observation"] = old["env"]["task"].pop("observation")
    old["objective"] = old["env"]["task"].pop("reward")
    old["env"]["execution"] = dict(
        dynamics=old["env"].pop("dynamics"),
        controller=old["env"].pop("controller"),
        command="attitude_thrust",
    )
    old["method"]["output"] = "attitude_thrust"
    metadata.update(config=old, config_version=3)
    source.with_suffix(".json").write_text(json.dumps(metadata))
    original = [path.read_bytes() for path in (source, source.with_suffix(".json"))]
    with pytest.raises(ValueError, match="migrate"):
        load_policy(source)
    destination = tmp_path / "converted/policy.pkl"
    evidence = migrate_checkpoint(source, destination, config)
    assert evidence["source_preserved"]
    assert source.read_bytes() == destination.read_bytes() == original[0]
    assert source.with_suffix(".json").read_bytes() == original[1]
    maker, restored, result = load_policy(destination)
    np.testing.assert_array_equal(
        maker(restored, deterministic=True)(observation, jax.random.key(2))[0], expected
    )
    assert result["migration"]["source_sha256"] == hashlib.sha256(original[0]).hexdigest()
    with pytest.raises(FileExistsError):
        migrate_checkpoint(source, destination, config)
    changed = copy.deepcopy(config)
    changed["env"]["task"]["observation"]["interval"] = 0.2
    with pytest.raises(ValueError, match="observation"):
        migrate_checkpoint(source, tmp_path / "invalid.pkl", changed)
