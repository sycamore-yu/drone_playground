"""Atomic, versioned Flax checkpoints and standalone inference archives.

Archives contain JSON identity/provenance and MessagePack arrays, never executable
pickles. Restoring training requires a freshly initialized matching state template;
its static Simulation metadata remains owned by the resolved environment config.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
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
    """Save a configured Actor without a trainer or executable checkpoint payload."""
    if kind not in ("state", "depth", "lidar") and actor_spec is None:
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


def save_model(path, state, *, config, provenance, kind, actor_spec, export_policy=False):
    """Write a version-three Orbax checkpoint under a model-NNNNNN folder.

    The full training state is saved once per update; the inference actor
    is exported only when selected. Metadata stays human-readable.
    """
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # State actors deliberately have no recurrent memory (shape [B,0]).
    # Orbax cannot serialize zero-size arrays; use an internal padded surrogate
    # only for persistence, never change the Actor or live TrainingState shape.
    recurrent = getattr(state, "recurrent_memory", None)
    empty_recurrent = recurrent is not None and recurrent.ndim == 2 and recurrent.shape[1] == 0
    serialized_state = (
        state.replace(recurrent_memory=jnp.zeros((recurrent.shape[0], 1), recurrent.dtype))
        if empty_recurrent
        else state
    )
    state_path = directory / "training"
    if state_path.exists() and not (state_path / "commit_success.txt").is_file():
        raise ValueError(f"Incomplete Orbax training checkpoint: {state_path}")
    if not state_path.exists():
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(state_path, serialized_state)
            checkpointer.wait_until_finished()
    if export_policy and not (directory / "actor").exists():
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(directory / "actor", state.params)
            checkpointer.wait_until_finished()
    metadata = {
        "format_version": 3,
        "purpose": "model",
        "config": config,
        "provenance": provenance,
        "kind": kind,
        "actor": actor_spec,
        "has_inference_actor": (directory / "actor").is_dir(),
        "padded_empty_recurrent_memory": bool(empty_recurrent),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=".metadata.", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(metadata, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "metadata.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    # Keep the selected actor and a small rolling window of resumable states.
    # This avoids accumulating a full copy of the large LiDAR sensor history
    # at every scheduled checkpoint.
    retention = config.get("experiment", {}).get("results", {}).get("keep_checkpoints", 3)
    if not isinstance(retention, int) or isinstance(retention, bool) or retention < 1:
        raise ValueError("results.keep_checkpoints must be a positive integer")
    candidates = sorted(
        (
            candidate
            for candidate in directory.parent.iterdir()
            if candidate.is_dir()
            and re.fullmatch(r"model_\d{6,}", candidate.name)
            and (candidate / "training/commit_success.txt").is_file()
        ),
        key=lambda candidate: int(candidate.name.split("_")[1]),
    )
    keep = set(candidates[-retention:])
    keep.update(candidate for candidate in candidates if (candidate / "actor").is_dir())
    for candidate in candidates:
        if candidate not in keep:
            shutil.rmtree(candidate)
    return directory


def load_model_state(path, template):
    """Restore the complete training PyTree into a matching Flax template."""
    directory = Path(path).resolve()
    _, metadata = read_checkpoint(directory, "training")
    empty_recurrent = metadata.get("padded_empty_recurrent_memory", False)
    if empty_recurrent:
        recurrent = getattr(template, "recurrent_memory", None)
        if recurrent is None or recurrent.ndim != 2 or recurrent.shape[1] != 0:
            raise ValueError("Orbax padded State actor requires an empty-memory template")
        restored_template = template.replace(
            recurrent_memory=jnp.zeros((recurrent.shape[0], 1), recurrent.dtype)
        )
    else:
        restored_template = template
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(directory / "training", restored_template)
    if empty_recurrent:
        restored = restored.replace(recurrent_memory=template.recurrent_memory)
    if jax.tree.structure(restored) != jax.tree.structure(template):
        raise ValueError("Orbax training checkpoint structure differs from template")
    for expected, actual in zip(jax.tree.leaves(template), jax.tree.leaves(restored), strict=True):
        if np.shape(expected) != np.shape(actual) or getattr(expected, "dtype", None) != getattr(
            actual, "dtype", None
        ):
            raise ValueError("Orbax training checkpoint leaf shape or dtype differs")
    return restored, metadata
