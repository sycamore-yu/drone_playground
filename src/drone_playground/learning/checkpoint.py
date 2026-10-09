"""Atomic, versioned Flax checkpoints and standalone inference archives.

Archives contain JSON identity/provenance and MessagePack arrays, never executable
pickles. Restoring training requires a freshly initialized matching state template;
its static Simulation metadata remains owned by the resolved environment config.
"""

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from drone_playground.simulation.policy import CHECKPOINT_VERSION, read_checkpoint


def _save(path, payload, *, purpose, config, provenance, kind=None, actor_spec=None):
    path = Path(path)
    metadata = {
        "format_version": CHECKPOINT_VERSION,
        "purpose": purpose,
        "config": config,
        "provenance": provenance,
        "kind": kind,
        "actor": actor_spec,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("metadata.json", metadata_json)
                archive.writestr("variables.msgpack", payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _is_typed_key(value):
    return isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)


def _serializable_keys(tree):
    return jax.tree.map(
        lambda value: jax.random.key_data(value) if _is_typed_key(value) else value, tree
    )


def save_state(path, state, *, config, provenance):
    """Save every TrainingState field and the resolved run config atomically."""
    _save(
        path,
        serialization.to_bytes(jax.device_get(_serializable_keys(state))),
        purpose="training",
        config=config,
        provenance=provenance,
    )


def load_state(path, template):
    """Return (restored TrainingState, metadata), checking leaf shapes and dtypes."""
    payload, metadata = read_checkpoint(path, "training")
    restored = serialization.from_bytes(_serializable_keys(template), payload)

    def restore_leaf(expected, actual):
        if _is_typed_key(expected):
            if np.shape(actual) != jax.random.key_data(expected).shape or np.asarray(
                actual
            ).dtype != np.dtype("uint32"):
                raise ValueError("Checkpoint RNG shape or dtype differs from the training template")
            return jax.random.wrap_key_data(actual, impl=jax.random.key_impl(expected))
        if (
            np.shape(expected) != np.shape(actual)
            or np.asarray(expected).dtype != np.asarray(actual).dtype
        ):
            raise ValueError("Checkpoint array shape or dtype differs from the training template")
        return jax.device_put(actual) if isinstance(expected, jax.Array) else actual

    restored = jax.tree.map(restore_leaf, template, restored)
    return restored, metadata


def save_inference(path, params, *, kind, config, provenance, actor_spec=None):
    """Save shared Actor full variables; inference needs only Simulation's Actor."""
    if kind not in ("state", "depth", "lidar"):
        raise ValueError("Unknown actor kind")
    if not isinstance(actor_spec, dict) or actor_spec.get("kind") != kind:
        raise ValueError("Actor specification must match the frozen inference kind")
    _save(
        path,
        serialization.to_bytes(jax.device_get(params)),
        purpose="inference",
        config=config,
        provenance=provenance,
        kind=kind,
        actor_spec=actor_spec,
    )


def load_inference(path):
    """Return params/kind/config/provenance and format metadata for frozen inference."""
    payload, metadata = read_checkpoint(path, "inference")
    return {
        **metadata,
        "params": jax.tree.map(jax.device_put, serialization.msgpack_restore(payload)),
    }
