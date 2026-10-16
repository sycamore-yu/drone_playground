"""Orbax checkpoints containing resumable training state and a frozen Actor."""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from drone_playground.simulation.policy import read_checkpoint


def _replace_link(path, target):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target, target_is_directory=True)
    try:
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _publish(directory, selection):
    # Readers follow one atomic pointer after both Orbax payloads and metadata commit.
    best = selection["best_checkpoint"]
    if best:
        read_checkpoint(best, "inference")
        _replace_link(directory / "best", os.path.relpath(best, directory))
    _replace_link(directory.parent / "latest", directory.name)
    if best:
        _replace_link(directory.parent.parent / "best_model", "checkpoints/latest/best")


def save_model(path, state, *, config, provenance, kind, actor_spec):
    """Write a version-three Orbax checkpoint under a model-NNNNNN folder.

    Each update stores both resumable state and a frozen actor.
    Metadata stays human-readable.
    """
    directory = Path(path).resolve()
    settings = config.get("experiment", {}).get("results", {})
    retention = settings.get("keep_checkpoints", 3)
    if not isinstance(retention, int) or isinstance(retention, bool) or retention < 1:
        raise ValueError("results.keep_checkpoints must be a positive integer")
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
    if not (directory / "actor").exists():
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(directory / "actor", state.params)
            checkpointer.wait_until_finished()
    if not (directory / "actor/commit_success.txt").is_file():
        raise ValueError(f"Incomplete Orbax inference checkpoint: {directory}")
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
    if "selection" in provenance:
        _publish(directory, provenance["selection"])
    # Keep actor snapshots and a rolling window of large resumable states.
    # This avoids accumulating a full copy of the large LiDAR sensor history
    # at every scheduled checkpoint.
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
    for candidate in candidates:
        if candidate not in keep:
            # Evaluation history must survive the rolling learner-state window.
            shutil.rmtree(candidate / "training")
    return directory


def load_model_state(path, template):
    """Restore the complete training PyTree into a matching Flax template."""
    directory = Path(path).resolve()
    metadata = read_checkpoint(directory, "training")
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
