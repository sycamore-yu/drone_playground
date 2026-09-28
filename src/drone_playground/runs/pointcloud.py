"""Authenticated paper-policy and optimizer snapshots using the existing Brax store."""

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from brax.io import model

from drone_playground.evaluation.tracking import save_report, tree_digest


def save_training_state(path, state, config, selection=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    model.save_params(str(temporary), jax.tree.map(np.asarray, state))
    temporary.replace(path)
    save_report(
        path.with_suffix(".json"),
        dict(
            family="paper_pointcloud_gru",
            kind="complete-training-state",
            config=config,
            updates=int(state.updates),
            parameter_sha256=tree_digest(state.params),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            selection=selection,
        ),
    )


def load_training_state(path):
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix(".json").read_text())
    from drone_playground.runs.migration import require_current

    metadata["config"] = require_current(metadata["config"])
    if metadata.get("family") != "paper_pointcloud_gru":
        raise ValueError("Wrong checkpoint family")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Checkpoint bytes differ from the recorded digest")
    state = jax.tree.map(jax.numpy.asarray, model.load_params(str(path)))
    if tree_digest(state.params) != metadata["parameter_sha256"]:
        raise ValueError("Checkpoint parameters differ from the recorded digest")
    return state, metadata
