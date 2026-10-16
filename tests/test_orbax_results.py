"""Orbax payload integrity, bounded state retention, and checkpoint selection."""

import json
import struct as binary

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf
from tensorboardX.proto.event_pb2 import Event

from drone_playground.learning.checkpoint import load_model_state, save_model
from drone_playground.simulation.networks import Actor
from drone_playground.simulation.policy import load_policy, read_checkpoint, resolve_checkpoint
from drone_playground.simulation.records import RunRecord


@struct.dataclass
class ExampleTrainingState:
    """Structured optimizer state with a typed RNG key."""

    params: dict
    optimizer_state: dict
    rng: jax.Array
    updates: int


def save_example(path, update=25, retention=3, selection=None):
    """Write a real small state with a complete experiment identity."""
    actor = Actor(kind="state", action_size=4)
    state = ExampleTrainingState(
        params={"params": {"linear": {"w": jnp.arange(4, dtype=jnp.float32)}}},
        optimizer_state={"actor": {"m": jnp.ones(4)}},
        rng=jax.random.key(82),
        updates=update,
    )
    config = {
        "experiment": {
            "results": {"keep_checkpoints": retention},
            "task": {"name": "tracking"},
            "sensor": {"name": "state"},
            "learning": {"algorithm": "bptt"},
        }
    }
    save_model(
        path,
        state,
        config=config,
        provenance={"updates": update, **({"selection": selection} if selection else {})},
        kind="state",
        actor_spec=actor.specification,
    )
    return actor, state


def test_orbax_restores_full_training_state_and_actor(tmp_path):
    """Every saved model can be resumed and played independently."""
    path = tmp_path / "checkpoints/model_000025"
    actor, state = save_example(path)
    restored, metadata = load_model_state(path, state)
    assert restored.updates == 25 and metadata["format_version"] == 3
    np.testing.assert_array_equal(jax.random.key_data(restored.rng), jax.random.key_data(state.rng))
    restored_actor, params, _ = load_policy(path, actor=actor)
    assert restored_actor is actor
    np.testing.assert_array_equal(params["params"]["linear"]["w"], jnp.arange(4))
    assert read_checkpoint(path, "inference")["kind"] == "state"
    with pytest.raises(ValueError, match="specification differs"):
        load_policy(path, actor=Actor(kind="state", action_size=3))
    (path / "actor/commit_success.txt").unlink()
    with pytest.raises(ValueError, match="Incomplete Orbax"):
        load_policy(path)


def test_retention_preserves_all_actors_and_evidence(tmp_path):
    """Only old optimizer/environment payloads leave the rolling retention window."""
    for update in (25, 50, 75, 100, 125):
        path = tmp_path / f"model_{update:06d}"
        _, state = save_example(path, update)
        (path / "episodes.csv").write_text(f"episode,event\n{update},COLLISION\n")
    assert sorted(p.parent.name for p in tmp_path.glob("model_*/training")) == [
        "model_000075",
        "model_000100",
        "model_000125",
    ]
    for update in (25, 50, 75, 100, 125):
        path = tmp_path / f"model_{update:06d}"
        read_checkpoint(path, "inference")
        assert (path / "episodes.csv").read_text() == f"episode,event\n{update},COLLISION\n"
    with pytest.raises(ValueError, match="no training payload"):
        load_model_state(tmp_path / "model_000025", state)


def test_selectors_use_newest_compatible_committed_run(tmp_path):
    """Best transparently falls back to latest without claiming a quality ranking."""
    for name, started, update in (("older", 100, 50), ("newer", 200, 25), ("unfinished", 300, 75)):
        root = tmp_path / name
        save_example(root / "checkpoints" / f"model_{update:06d}", update)
        (root / "run.json").write_text(json.dumps({"layout_version": 3, "started_unix": started}))
    (tmp_path / "unfinished/checkpoints/model_000075/actor/commit_success.txt").unlink()
    args = dict(root=tmp_path, task="tracking", sensor="state", algorithm="bptt")
    expected = tmp_path / "newer/checkpoints/model_000025"
    assert resolve_checkpoint("latest", **args) == expected
    assert resolve_checkpoint("best", **args) == expected
    assert resolve_checkpoint(str(expected), **args) == expected
    with pytest.raises(FileNotFoundError, match="No compatible"):
        resolve_checkpoint("latest", **{**args, "algorithm": "ppo"})


def test_selectors_keep_static_and_dynamic_training_runs_independent(tmp_path):
    """A newer dynamic policy cannot become the default for a static experiment."""
    for started, motion in enumerate(("static", "dynamic"), 1):
        root = tmp_path / motion
        path = root / "checkpoints/model_000025"
        save_example(path)
        metadata = json.loads((path / "metadata.json").read_text())
        metadata["config"]["experiment"]["simulation"] = {"procedural": {"obstacle_motion": motion}}
        (path / "metadata.json").write_text(json.dumps(metadata))
        (root / "run.json").write_text(json.dumps({"layout_version": 3, "started_unix": started}))
    for motion in ("static", "dynamic"):
        selected = resolve_checkpoint(
            "latest",
            root=tmp_path,
            task="tracking",
            sensor="state",
            algorithm="bptt",
            obstacle_motion=motion,
        )
        assert selected == tmp_path / motion / "checkpoints/model_000025"


def test_publication_failure_keeps_latest_and_best_committed(tmp_path, monkeypatch):
    """An unreferenced committed payload cannot silently become the latest model."""
    from drone_playground.learning import checkpoint

    root = tmp_path / "run"
    first = root / "checkpoints/model_000001"
    selection = {"best_checkpoint": str(first), "best_mean_reward": -3.0}
    save_example(first, 1, retention=1, selection=selection)
    (root / "run.json").write_text(json.dumps({"layout_version": 3, "started_unix": 1}))
    second = root / "checkpoints/model_000002"
    save_example(second, 2, retention=1, selection=selection)
    assert not (first / "training").exists()
    assert (root / "best_model").resolve() == first
    third = root / "checkpoints/model_000003"
    replacement = {"best_checkpoint": str(third), "best_mean_reward": -1.0}
    original = checkpoint.os.replace

    def fail_pointer(source, destination):
        if destination.name == "latest":
            raise OSError("injected pointer failure")
        return original(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", fail_pointer)
    with pytest.raises(OSError, match="pointer failure"):
        save_example(third, 3, retention=1, selection=replacement)
    args = dict(root=tmp_path, task="tracking", sensor="state", algorithm="bptt")
    assert resolve_checkpoint("latest", **args) == second
    assert resolve_checkpoint("best", **args) == first
    assert (root / "best_model").resolve() == first
    monkeypatch.setattr(checkpoint.os, "replace", original)
    save_example(third, 3, retention=1, selection=replacement)
    assert resolve_checkpoint("latest", **args) == third
    assert resolve_checkpoint("best", **args) == third
    assert (root / "best_model").resolve() == third


def test_tensorboard_and_run_index_record_training(tmp_path):
    """TensorBoard and the machine-readable event log share the same measurements."""
    record = RunRecord(tmp_path / "run", OmegaConf.create({"seed": 0}))
    record.event("update", update=2, interactions=16, reward=-0.3)
    record.finish("interrupted", updates=2)
    assert record.layout_version == 3
    rows = [
        json.loads(line) for line in (record.directory / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["reward"] == -0.3 and rows[-1]["interactions"] == 16
    events = {}
    for path in record.directory.glob("events.out.tfevents.*"):
        with path.open("rb") as stream:
            while header := stream.read(12):
                length = binary.unpack("<Q", header[:8])[0]
                event = Event.FromString(stream.read(length))
                assert len(stream.read(4)) == 4
                events.update({value.tag: value for value in event.summary.value})
    np.testing.assert_allclose(events["train/reward"].simple_value, -0.3)
    identity = json.loads((record.directory / "run.json").read_text())
    assert identity["status"] == "interrupted" and identity["updates"] == 2


def test_legacy_run_cannot_resume_into_current_layout(tmp_path):
    """Reject legacy archives without changing their records."""
    path = tmp_path / "run.json"
    contents = json.dumps({"layout_version": 2})
    path.write_text(contents)
    with pytest.raises(ValueError, match="Orbax"):
        RunRecord(tmp_path, OmegaConf.create({"seed": 0}), resume=True)
    assert path.read_text() == contents


def test_resumed_checkpoint_can_start_a_new_run_directory(tmp_path):
    """An external resumed attempt gets a fresh config and run identity."""
    record = RunRecord(tmp_path / "new-attempt", OmegaConf.create({"seed": 3}), resume=True)
    record.finish("interrupted")
    assert OmegaConf.load(record.directory / "config.yaml").seed == 3
    assert json.loads((record.directory / "run.json").read_text())["layout_version"] == 3
