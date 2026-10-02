"""Authenticated paper-policy and optimizer snapshots using the existing Brax store."""

import hashlib
import json
import pickle
from pathlib import Path

import jax
import numpy as np
from brax.io import model

from drone_playground.artifacts.layout import resolve_artifact
from drone_playground.artifacts.reporting import save_report, tree_digest


class _TrainingStateUnpickler(pickle.Unpickler):
    """Resolve the one moved state class in authenticated existing Brax pickles."""

    def find_class(self, module, name):
        if (module, name) == (
            "drone_playground.learning.algorithms.pointcloud_bptt", "TrainingState"
        ):
            from drone_playground.learning.algorithms.recurrent_bptt import TrainingState

            return TrainingState
        return super().find_class(module, name)


def save_training_state(path, state, config, selection=None, *, selection_report=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    model.save_params(str(temporary), jax.tree.map(np.asarray, state))
    temporary.replace(path)
    save_report(
        path.with_suffix(".json"),
        dict(
            family="pointcloud_gru",
            kind="complete-training-state",
            config=config,
            updates=int(state.updates),
            parameter_sha256=tree_digest(state.params),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            selection=selection,
            selection_report=selection_report,
        ),
    )


def load_training_state(path):
    path = resolve_artifact(path).resolve()
    metadata = json.loads(path.with_suffix(".json").read_text())
    from drone_playground.artifacts.schema import require_current

    metadata["config"] = require_current(metadata["config"])
    # Frozen artifact provenance tag stored in existing checkpoint metadata; not a public name.
    if metadata.get("family") not in ("pointcloud_gru", "paper_pointcloud_gru"):
        raise ValueError("Wrong checkpoint family")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Checkpoint bytes differ from the recorded digest")
    with path.open("rb") as stream:
        state = jax.tree.map(jax.numpy.asarray, _TrainingStateUnpickler(stream).load())
    if tree_digest(state.params) != metadata["parameter_sha256"]:
        raise ValueError(
            "Checkpoint parameters differ from the recorded digest"
        )
    return state, metadata
