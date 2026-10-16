"""Load and execute a frozen Actor without importing a trainer or optimizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drone_playground.simulation.networks import Actor

CHECKPOINT_VERSION = 3


def resolve_checkpoint(selector, *, root, task, sensor, algorithm, obstacle_motion=None):
    """Resolve latest or recorded reward-best within the newest compatible run."""
    if selector not in {"best", "latest"}:
        path = Path(selector).resolve()
        read_checkpoint(path, "inference")
        return path
    candidates = []
    for run in Path(root).rglob("run.json"):
        identity = json.loads(run.read_text())
        if identity.get("layout_version") != 3:
            continue
        published = run.parent / "checkpoints/latest"
        metadata_paths = (
            [published.resolve(strict=True) / "metadata.json"]
            if published.is_symlink()
            else (run.parent / "checkpoints").glob("model_*/metadata.json")
        )
        for metadata_path in metadata_paths:
            metadata = json.loads(metadata_path.read_text())
            experiment = metadata["config"]["experiment"]
            if (
                experiment["task"]["name"] == task
                and experiment["sensor"]["name"] == sensor
                and experiment["learning"]["algorithm"] == algorithm
                and (
                    obstacle_motion is None
                    or experiment.get("simulation", {}).get("procedural", {}).get("obstacle_motion")
                    == obstacle_motion
                )
                and (metadata_path.parent / "actor/commit_success.txt").is_file()
            ):
                candidates.append(
                    (
                        identity["started_unix"],
                        metadata["provenance"]["updates"],
                        str(metadata_path.parent.resolve()),
                    )
                )
    if not candidates:
        raise FileNotFoundError(f"No compatible Orbax checkpoint in {root}")
    chosen = max(candidates)
    if selector == "best":
        best = checkpoint_summary(chosen[2])["best_checkpoint"]
        if best:
            path = Path(best).resolve()
            read_checkpoint(path, "inference")
            return path
    return Path(chosen[2])


def checkpoint_summary(path):
    """Read selection facts from the committed model, never a mutable run summary."""
    path = Path(path).resolve()
    metadata = read_checkpoint(path, "inference")
    selection = metadata["provenance"].get("selection", {})
    return {
        "latest_checkpoint": str(path),
        "best_checkpoint": selection.get("best_checkpoint"),
        "best_mean_reward": selection.get("best_mean_reward"),
        "best_source": metadata["config"]
        .get("experiment", {})
        .get("learning", {})
        .get("best", {})
        .get("source"),
    }


def read_checkpoint(path: str | Path, purpose: str) -> dict:
    """Validate the Orbax model metadata and required committed payload."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Checkpoint must be an Orbax model directory")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("format_version") != CHECKPOINT_VERSION or metadata.get("purpose") != "model":
        raise ValueError("Unsupported Orbax model format")
    required = path / ("training" if purpose == "training" else "actor")
    if purpose not in ("training", "inference") or not required.is_dir():
        raise ValueError(f"Orbax model has no {purpose} payload")
    if not (required / "commit_success.txt").is_file():
        raise ValueError(f"Incomplete Orbax {purpose} checkpoint: {required}")
    if purpose == "inference" and not isinstance(metadata.get("actor"), dict):
        raise ValueError("Orbax model requires an explicit Actor specification")
    return metadata


def load_policy(path: str | Path, *, actor=None) -> tuple[Actor, dict, dict]:
    """Restore an Actor selected by trusted configuration, never checkpoint executable names."""
    from drone_playground.simulation.networks import Actor

    path = Path(path)
    metadata = read_checkpoint(path, "inference")
    if actor is None and metadata.get("kind") not in {
        "state",
        "depth",
        "lidar",
        "fan_lidar",
        "visfly_depth",
    }:
        raise ValueError("Checkpoint does not declare a supported actor kind")
    import orbax.checkpoint as ocp

    with ocp.StandardCheckpointer() as checkpointer:
        parameters = checkpointer.restore(path.resolve() / "actor")
    specification = metadata["actor"]
    if specification.get("kind") != metadata["kind"]:
        raise ValueError("Checkpoint Actor kind and specification disagree")
    if actor is not None:
        if actor.specification != specification:
            raise ValueError("Checkpoint Actor specification differs from selected method")
        return actor, parameters, metadata
    return Actor(**specification), parameters, metadata
