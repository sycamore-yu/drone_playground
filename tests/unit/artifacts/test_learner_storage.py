"""Shared checkpoint storage preserves typed RNG keys and rejects corrupt bytes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct


@struct.dataclass
class Snapshot:
    params: object
    key: object
    updates: object


def test_learner_snapshot_restores_typed_rng_and_metadata(tmp_path):
    from drone_playground.artifacts.training_state import load_learner_state, save_learner_state

    state = Snapshot({"w": jnp.array([1.0, 2.0])}, jax.random.key(13), jnp.int32(7))
    path = tmp_path / "state.pkl"
    save_learner_state(path, state, {"seed": 13}, kind="fixture-training-state")
    actual, metadata = load_learner_state(path, kind="fixture-training-state")
    np.testing.assert_array_equal(actual.params["w"], state.params["w"])
    np.testing.assert_array_equal(jax.random.key_data(actual.key), jax.random.key_data(state.key))
    assert actual.key.dtype == state.key.dtype
    assert int(actual.updates) == 7
    assert metadata["config"] == {"seed": 13}
    with pytest.raises(ValueError, match=r"kind|Kind|checkpoint"):
        load_learner_state(path, kind="other-training-state")
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match=r"digest|Digest"):
        load_learner_state(path, kind="fixture-training-state")
