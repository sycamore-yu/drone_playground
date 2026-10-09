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


def load_policy(path: str | Path) -> tuple[Actor, dict, dict]:
    """Return the shared Actor, its explicit variables and its complete experiment identity."""
    payload, metadata = read_checkpoint(path, "inference")
    if metadata.get("kind") not in {"state", "depth", "lidar"}:
        raise ValueError("Checkpoint does not declare a supported actor kind")
    parameters = jax.tree.map(jax.device_put, serialization.msgpack_restore(payload))
    specification = metadata["actor"]
    if specification.get("kind") != metadata["kind"]:
        raise ValueError("Checkpoint Actor kind and specification disagree")
    return Actor(**specification), parameters, metadata
