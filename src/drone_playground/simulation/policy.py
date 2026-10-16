"""Load and execute a frozen Actor without importing a trainer or optimizer."""

import hashlib
import json
import zipfile
from pathlib import Path

import jax
from flax import serialization

from drone_playground.simulation.networks import Actor

CHECKPOINT_VERSION = 2


def read_checkpoint(path: str | Path, purpose: str) -> tuple[bytes, dict]:
    """Validate the versioned array archive shared by training and frozen inference."""
    path = Path(path)
    if path.is_dir():
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("format_version") != 3 or metadata.get("purpose") != "model":
            raise ValueError("Unsupported Orbax model format")
        required = path / ("training" if purpose == "training" else "actor")
        if purpose not in ("training", "inference") or not required.is_dir():
            raise ValueError(f"Orbax model has no {purpose} payload")
        if purpose == "inference" and not isinstance(metadata.get("actor"), dict):
            raise ValueError("Orbax model requires an explicit Actor specification")
        return b"", metadata
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        payload = archive.read("variables.msgpack")
    if metadata.get("format_version") != CHECKPOINT_VERSION or metadata.get("purpose") != purpose:
        raise ValueError("Unsupported checkpoint version or purpose")
    if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
        raise ValueError("Checkpoint payload checksum mismatch")
    if purpose == "inference" and not isinstance(metadata.get("actor"), dict):
        raise ValueError("Inference archive requires an explicit Actor specification")
    return payload, metadata


def load_policy(path: str | Path, *, actor=None) -> tuple[Actor, dict, dict]:
    """Restore an Actor selected by trusted configuration, never checkpoint executable names."""
    path = Path(path)
    payload, metadata = read_checkpoint(path, "inference")
    if actor is None and metadata.get("kind") not in {
        "state",
        "depth",
        "lidar",
        "fan_lidar",
        "visfly_depth",
    }:
        raise ValueError("Checkpoint does not declare a supported actor kind")
    if path.is_dir():
        import orbax.checkpoint as ocp

        with ocp.StandardCheckpointer() as checkpointer:
            parameters = checkpointer.restore(path.resolve() / "actor")
    else:
        parameters = jax.tree.map(jax.device_put, serialization.msgpack_restore(payload))
    specification = metadata["actor"]
    if specification.get("kind") != metadata["kind"]:
        raise ValueError("Checkpoint Actor kind and specification disagree")
    if actor is not None:
        if actor.specification != specification:
            raise ValueError("Checkpoint Actor specification differs from selected method")
        return actor, parameters, metadata
    return Actor(**specification), parameters, metadata
