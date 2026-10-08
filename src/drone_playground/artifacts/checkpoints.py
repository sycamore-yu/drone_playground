"""Integrity-checked checkpoint bytes and metadata, without model reconstruction."""

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from brax.io import model


def save_checkpoint(path, tree, metadata):
    """Atomically write a trusted local array tree and its digest-bearing sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    model.save_params(str(temporary), jax.tree.map(np.asarray, tree))
    temporary.replace(path)
    record = dict(metadata, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    sidecar = path.with_suffix(".json")
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(sidecar)
    return path


def load_checkpoint(path, *, kind=None):
    """Verify a trusted local checkpoint before deserializing its array tree."""
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix(".json").read_text())
    if kind is not None and metadata.get("kind") != kind:
        raise ValueError(f"Wrong checkpoint kind: expected {kind}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Checkpoint digest mismatch")
    return model.load_params(str(path)), metadata
