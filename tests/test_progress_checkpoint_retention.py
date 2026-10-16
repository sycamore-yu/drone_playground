"""Preserve an improving 0/48 pilot separately from its rolling checkpoints."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import struct

from drone_playground.learning.checkpoint import save_model


@struct.dataclass
class _SmallState:
    params: dict
    recurrent_memory: object = None


def _screen(directory, successes=0):
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "complete": True,
        "protocol": "diagnostic",
        "reports": {
            scene: {"task": "navigation", "episodes": 8, "successes": successes}
            for scene in ("S01", "S02", "S03", "D01", "D02", "D03")
        },
    }
    (directory / "report.json").write_text(json.dumps(report))


def test_retention_honours_safe_progress_peak_even_when_all_screens_are_zero(tmp_path, monkeypatch):
    """A 0/48 score tie may not erase the earlier genuine progress evidence."""

    class MockCheckpointer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def save(self, destination, tree):
            destination = Path(destination)
            destination.mkdir(parents=True)
            (destination / "commit_success.txt").write_text("committed")

        def wait_until_finished(self):
            return None

    monkeypatch.setattr(ocp, "StandardCheckpointer", MockCheckpointer)
    root = tmp_path / "checkpoints"
    for update in (100, 200, 300, 400, 500):
        d = root / f"model_{update:06d}"
        (d / "training").mkdir(parents=True)
        (d / "training" / "commit_success.txt").write_text("committed")
        _screen(d / "screening")

    save_model(
        root / "model_000600",
        SimpleNamespace(params={"x": jnp.zeros(1)}, recurrent_memory=None),
        config={
            "experiment": {
                "results": {"keep_checkpoints": 2, "keep_best_screening": True},
                "screening": {
                    "progress_aware": {
                        "min_progress_delta": 0.03,
                        "max_unsafe_increase": 0.0,
                    }
                },
            }
        },
        provenance={
            "screening_progress": {"progress_guard": {"best_progress_checkpoint_update": 100}}
        },
        kind="state",
        actor_spec={"kind": "state"},
    )
    retained = sorted(path.parent.name for path in root.glob("model_*/training"))
    assert retained == ["model_000100", "model_000500", "model_000600"]


def test_original_retention_does_not_keep_progress_only_checkpoint(tmp_path, monkeypatch):
    """Feature-flagged policy cannot retroactively alter historical checkpoint pruning."""

    class MockCheckpointer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def save(self, destination, tree):
            destination = Path(destination)
            destination.mkdir(parents=True)
            (destination / "commit_success.txt").write_text("committed")

        def wait_until_finished(self):
            return None

    monkeypatch.setattr(ocp, "StandardCheckpointer", MockCheckpointer)
    root = tmp_path / "checkpoints"
    for update in (100, 200, 300, 400, 500):
        d = root / f"model_{update:06d}"
        (d / "training").mkdir(parents=True)
        (d / "training" / "commit_success.txt").write_text("committed")
        _screen(d / "screening")
    save_model(
        root / "model_000600",
        SimpleNamespace(params={"x": jnp.zeros(1)}, recurrent_memory=None),
        config={"experiment": {"results": {"keep_checkpoints": 2, "keep_best_screening": True}}},
        provenance={"screening_progress": {"best_progress_checkpoint_update": 100}},
        kind="state",
        actor_spec={"kind": "state"},
    )
    retained = sorted(path.parent.name for path in root.glob("model_*/training"))
    assert retained == ["model_000500", "model_000600"]


def test_retention_keeps_best_training_completion_separately_from_fixed_scene_score(tmp_path):
    """Zero long-distance success must not erase a learned short-distance arrival policy."""
    root = tmp_path / "checkpoints"
    for update, successes in ((100, 6), (200, 1), (300, 2)):
        directory = root / f"model_{update:06d}"
        (directory / "training").mkdir(parents=True)
        (directory / "training/commit_success.txt").write_text("committed")
        _screen(directory / "screening")
        probe = directory / "training_distribution"
        probe.mkdir()
        (probe / "report.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "reports": {
                        "train/procedural": {
                            "task": "navigation",
                            "episodes": 8,
                            "successes": successes,
                        }
                    },
                }
            )
        )
    save_model(
        root / "model_000400",
        _SmallState(params={"x": jnp.zeros(1)}),
        config={
            "experiment": {
                "results": {"keep_checkpoints": 1, "keep_best_screening": True},
                "screening": {"training_distribution_probe": True},
            }
        },
        provenance={},
        kind="state",
        actor_spec={"kind": "state"},
    )
    assert sorted(path.parent.name for path in root.glob("model_*/training")) == [
        "model_000100",  # Best procedural arrival rate: 6/8.
        "model_000300",  # Latest completed fixed-scene score tie.
        "model_000400",  # Latest recovery state.
    ]
