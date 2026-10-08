"""Learner state persistence with shared typed-RNG and integrity handling."""

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.artifacts.checkpoints import load_checkpoint, save_checkpoint
from drone_playground.artifacts.reporting import tree_digest


def _host_state(state):
    typed = []

    def to_host(path, value):
        if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            typed.append(jax.tree_util.keystr(path))
            return np.asarray(jax.random.key_data(value))
        return np.asarray(value)

    return jax.tree_util.tree_map_with_path(to_host, state), typed


def _device_state(state, metadata):
    paths = set(metadata.get("typed_key_paths", ()))
    return jax.tree_util.tree_map_with_path(
        lambda path, value: (
            jax.random.wrap_key_data(jnp.asarray(value))
            if jax.tree_util.keystr(path) in paths
            else jnp.asarray(value)
        ),
        state,
    )


def save_learner_state(path, state, config, *, kind):
    """Save optimizer, model, environment and RNG leaves without interpreting them."""
    host, typed = _host_state(state)
    return save_checkpoint(
        path,
        host,
        dict(
            kind=kind,
            config=config,
            typed_key_paths=typed,
            updates=int(state.updates),
        ),
    )


def load_learner_state(path, *, kind):
    """Restore the exact learner tree; the algorithm checks continuation settings."""
    state, metadata = load_checkpoint(path, kind=kind)
    return _device_state(state, metadata), metadata


def save_training_state(path, state, config, selection=None, *, selection_report=None):
    """Save a recurrent learner and its checkpoint selection evidence."""
    host, typed = _host_state(state)
    return save_checkpoint(
        path,
        host,
        dict(
            family="recurrent_policy",
            kind="complete-training-state",
            config=config,
            updates=int(state.updates),
            typed_key_paths=typed,
            parameter_sha256=tree_digest(state.params),
            selection=selection,
            selection_report=selection_report,
        ),
    )


def load_training_state(path):
    """Read a recurrent snapshot without creating an environment or network."""
    from drone_playground.artifacts.schema import require_current

    state, metadata = load_checkpoint(path)
    metadata["config"] = require_current(metadata["config"])
    if metadata.get("family") != "recurrent_policy":
        raise ValueError("Wrong checkpoint family")
    state = _device_state(state, metadata)
    if tree_digest(state.params) != metadata["parameter_sha256"]:
        raise ValueError("Checkpoint parameters differ from the recorded digest")
    return state, metadata
