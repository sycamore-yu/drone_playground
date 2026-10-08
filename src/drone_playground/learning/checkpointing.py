"""Restore and select recurrent checkpoints without owning a task's rollout or loss."""

from __future__ import annotations

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


def run_recurrent_updates(
    state,
    update,
    snapshot,
    config,
    recorder,
    *,
    steps_per_update,
    milestones,
    validate=None,
    metric_prefix="",
):
    """Run a recurrent training stage without choosing its update or task score.

    The caller owns rollout, loss, optimizer and evaluation. This loop owns update
    counting, finite metrics, reporting, snapshot scheduling and the wall limit.
    """
    import time

    settings = config["training"]
    target = int(settings["policy_updates"])
    stop = min(target, int(settings.get("stop_after_updates") or target))
    first = int(state.updates)
    if first >= stop:
        raise ValueError("Continuation must perform at least one additional update")
    started = time.monotonic()
    durations, metrics = [], {}
    last_snapshot = snapshot(state)
    last_saved = first
    for iteration in range(first + 1, stop + 1):
        recorder.phase("training", int(state.updates) * steps_per_update, target_updates=target)
        tick = time.monotonic()
        state, values = update(state)
        metrics = {name: float(value) for name, value in values.items()}
        elapsed = time.monotonic() - tick
        durations.append(elapsed)
        if not all(np.isfinite(value) for value in metrics.values()):
            save_training_state(recorder.path / "training-state/nonfinite.pkl", state, config)
            raise FloatingPointError(f"Non-finite recurrent update {iteration}: {metrics}")
        if validate is not None:
            validate(metrics)
        if iteration == first + 1 or iteration % 10 == 0 or iteration in milestones:
            recorder.log(
                iteration * steps_per_update,
                {
                    **{metric_prefix + key: value for key, value in metrics.items()},
                    metric_prefix + "updates": iteration,
                    metric_prefix + "update_seconds": elapsed,
                },
            )
        if iteration in milestones or iteration == stop:
            last_snapshot = snapshot(state)
            last_saved = iteration
        if time.monotonic() - started >= settings["max_wall_seconds"]:
            break
    if last_saved != int(state.updates):
        last_snapshot = snapshot(state)
    return (
        state,
        metrics,
        dict(
            started=started,
            durations=durations,
            last_snapshot=last_snapshot,
        ),
    )


def require_matching_physical_decoder(metadata, config):
    """Parameter transfer cannot silently reinterpret a geometric head's units."""
    from drone_playground.artifacts.schema import require_current
    from drone_playground.control.decoders import PhysicalActionDecoder

    current = require_current(config)
    expected = current["method"].get("physical_decoder")
    recorded = metadata.get("physical_decoder")
    if expected is None and recorded is None:
        return
    if (
        expected is None
        or recorded is None
        or PhysicalActionDecoder(**expected) != PhysicalActionDecoder(**recorded)
        or current["method"].get("goal_source", "task_goal")
        != metadata["config"]["method"].get("goal_source", "task_goal")
    ):
        raise ValueError(
            "Geometric warm start changes the physical decoder or goal source; "
            "explicit migration required"
        )


def save_policy(directory: Path, params, config: dict, step: int, *, physical_decoder=None) -> Path:
    """Record a policy with its reconstructed observation and physical output contract."""
    from dataclasses import asdict
    from pathlib import Path

    from drone_playground.artifacts.checkpoints import save_checkpoint
    from drone_playground.artifacts.reporting import tree_digest
    from drone_playground.artifacts.schema import require_current

    directory = Path(directory)
    current = require_current(config)
    component_only = (
        physical_decoder is not None and current["method"].get("physical_decoder") is None
    )
    if physical_decoder is None:
        physical_decoder = current["method"].get("physical_decoder")
    from drone_playground.environments.factory import (
        build_controller,
        build_observer,
        build_sensor,
    )

    controller = build_controller(current["env"]["controller"])
    observer = build_observer(current, build_sensor(current))
    action_size = len(controller.input_fields)
    if physical_decoder is not None:
        from drone_playground.control.decoders import PhysicalActionDecoder

        decoder = PhysicalActionDecoder(**physical_decoder)
        if decoder.kind != current["method"]["output"]:
            raise ValueError("Checkpoint method output differs from its physical decoder")
        action_size = decoder.action_size
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{int(step):010d}.pkl"
    metadata = dict(
        config_version=4,
        step=int(step),
        config=current,
        observation_size=observer.size,
        observation_spec=observer.specification(),
        action_size=action_size,
        policy_family="brax",
        checkpoint_kind="component-inference-parameters"
        if component_only
        else "inference-parameters",
        parameter_sha256=tree_digest(params),
        continuation={
            "ppo": "warm start only; optimizer/RNG reinitialized",
            "apg": "inference only; native APG has no restore hook",
            "shac": "full continuation is stored in training-state/",
            "bptt": "full continuation is stored in training-state/",
            "dva": "full continuation is stored in training-state/",
        }[current["algorithm"]["name"]],
    )
    if physical_decoder is not None:
        # Persist every physical default so future decoder defaults cannot
        # silently change a frozen policy's units, anchors or finite horizon.
        metadata["physical_decoder"] = asdict(decoder)
        if component_only:
            metadata["continuation"] = (
                "frozen physical component; optimizer continuation is not encoded"
            )
    return save_checkpoint(path, params, metadata)
