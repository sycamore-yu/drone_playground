"""Restore and select recurrent checkpoints without owning a task's rollout or loss."""

import copy
from pathlib import Path

import numpy as np

from drone_playground.artifacts.reporting import save_report, tree_digest
from drone_playground.artifacts.training_state import load_training_state, save_training_state


def continuation_contract(config):
    """Keep training semantics while excluding session/output placement settings."""
    value = copy.deepcopy(config)
    for key in ("run_id", "checkpoint", "evaluation", "replay", "hydra", "visualization"):
        value.pop(key, None)
    for key in ("resume", "max_wall_seconds", "stop_after_updates", "device"):
        value["training"].pop(key, None)
    value["runtime"].pop("device", None)
    return value


def restore_recurrent_state(initial, config, contract=continuation_contract):
    """Restore the optimizer, RNG and selection only for an identical training contract."""
    resume = config["training"].get("resume")
    if not resume:
        return initial, {}
    state, metadata = load_training_state(resume)
    if contract(metadata["config"]) != contract(config):
        raise ValueError("Exact recurrent continuation contract differs")
    return state, metadata


def save_recurrent_snapshot(
    path, state, config, report, score, selected, selected_report, *, report_path, fields=None
):
    """Persist a snapshot and its best-so-far record using the caller's existing score."""
    path = Path(path)
    score = tuple(float(value) for value in score)
    if not score or not np.isfinite(score).all():
        raise FloatingPointError("Checkpoint selection score must be finite")
    previous = None
    if selected is not None:
        previous = selected.get("score")
        if previous is None:
            raise ValueError("Recorded recurrent selection has no comparison score")
    if previous is None or score > tuple(previous):
        selected = dict(
            checkpoint=str(path.resolve()),
            updates=int(state.updates),
            score=list(score),
            parameter_sha256=tree_digest(state.params),
            selection_role="eval",
            **(fields or {}),
        )
        selected_report = copy.deepcopy(report)
    save_training_state(path, state, config, selected, selection_report=selected_report)
    save_report(report_path, report)
    save_report(path.parent.parent / "best.json", selected)
    return selected, selected_report
