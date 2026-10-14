"""Test final JAX results layout without altering legacy training runs."""

import json
import struct as binary

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from omegaconf import OmegaConf
from tensorboardX.proto.event_pb2 import Event

from drone_playground.learning.checkpoint import load_model_state, save_model
from drone_playground.simulation.networks import Actor
from drone_playground.simulation.policy import load_policy, read_checkpoint
from drone_playground.simulation.records import RunRecord


@struct.dataclass
class ExampleTrainingState:
    """Represent an explicitly structured Flax state with a typed RNG key."""

    params: dict
    optimizer_state: dict
    rng: jax.Array
    updates: int


def test_orbax_model_directory_restores_training_and_selected_actor(tmp_path):
    """Save complete optimizer/RNG state while exporting the selected Actor separately."""
    actor = Actor(kind="state", action_size=4)
    state = ExampleTrainingState(
        params={"params": {"linear": {"w": jnp.arange(4, dtype=jnp.float32)}}},
        optimizer_state={"actor": {"m": jnp.ones(4)}},
        rng=jax.random.key(82),
        updates=25,
    )
    path = tmp_path / "checkpoints/model_000025"
    cfg = {"experiment": {"seed": 7}, "learning": {"actor": actor.specification}}
    save_model(
        path,
        state,
        config=cfg,
        provenance={"updates": 25},
        kind="state",
        actor_spec=actor.specification,
    )
    restored, metadata = load_model_state(path, state)
    assert restored.updates == 25
    assert metadata["format_version"] == 3
    np.testing.assert_array_equal(restored.params["params"]["linear"]["w"], jnp.arange(4))
    np.testing.assert_array_equal(jax.random.key_data(restored.rng), jax.random.key_data(state.rng))
    assert not (path / "actor").exists()

    save_model(
        path,
        state,
        config=cfg,
        provenance={"updates": 25, "selected": True},
        kind="state",
        actor_spec=actor.specification,
        export_policy=True,
    )
    restored_actor, parameters, metadata = load_policy(path, actor=actor)
    assert restored_actor is actor
    assert metadata["provenance"]["selected"] is True
    np.testing.assert_array_equal(parameters["params"]["linear"]["w"], jnp.arange(4))
    _, meta = read_checkpoint(path, "inference")
    assert meta["kind"] == "state"


def test_tensorboard_records_do_not_duplicate_jsonl_metrics(tmp_path):
    """Use one TensorBoard event stream plus a small machine-readable run index."""
    config = OmegaConf.create({"seed": 0, "results": {"format": "orbax"}})
    record = RunRecord(tmp_path / "run", config)
    record.event("update", update=2, interactions=16, reward=-0.3, actor_grad_norm=0.12)
    scenes = {"S01": {"episodes": 8, "successes": 3}}
    record.event(
        "screening_evaluation",
        update=2,
        passed=False,
        episodes=8,
        per_scene=scenes,
        score=0.3,
        best_score=0.4,
        stagnant_screens=2,
    )
    record.event("checkpoint_evaluation", update=2, passed=False)
    record.finish("interrupted", updates=2)
    assert record.layout_version == 3
    assert (record.directory / "config.yaml").is_file()
    assert list(record.directory.glob("events.out.tfevents.*"))
    assert not (record.directory / "metrics.jsonl").exists()
    assert OmegaConf.load(record.directory / "config.yaml").seed == 0
    events = {}
    for path in record.directory.glob("events.out.tfevents.*"):
        with path.open("rb") as stream:
            while header := stream.read(12):
                length = binary.unpack("<Q", header[:8])[0]
                event = Event.FromString(stream.read(length))
                assert len(stream.read(4)) == 4
                events.update({value.tag: value for value in event.summary.value})
    screening = json.loads(events["event/screening_evaluation/text_summary"].tensor.string_val[0])
    assert screening["per_scene"] == scenes
    assert screening["stagnant_screens"] == 2
    np.testing.assert_allclose(events["evaluation/score"].simple_value, 0.3)
    np.testing.assert_allclose(events["evaluation/best_score"].simple_value, 0.4)
    assert events["evaluation/stagnant_screens"].simple_value == 2
    assert (
        json.loads((record.directory / "run.json").read_text())["last_evaluation"]["event"]
        == "checkpoint_evaluation"
    )


def test_orbax_retains_last_three_plus_selected_actor(tmp_path):
    """Bound sensor/optimizer snapshot growth without deleting a selected model."""
    config = {
        "experiment": {"results": {"keep_checkpoints": 3}},
        "learning": {},
    }
    actor = Actor(kind="state", action_size=4)
    base = ExampleTrainingState(
        params={"params": {"layer": {"kernel": jnp.ones((2, 2))}}},
        optimizer_state={"actor": jnp.zeros(1)},
        rng=jax.random.key(5),
        updates=0,
    )
    for update in (25, 50, 75, 100, 125):
        save_model(
            tmp_path / f"model_{update:06d}",
            base.replace(updates=update),
            config=config,
            provenance={"updates": update},
            kind="state",
            actor_spec=actor.specification,
            export_policy=update == 25,
        )
    assert sorted(path.name for path in tmp_path.glob("model_*")) == [
        "model_000025",
        "model_000075",
        "model_000100",
        "model_000125",
    ]


def test_orbax_optionally_retains_best_incomplete_screening_checkpoint(tmp_path):
    """Preserve measured PPO progress even when no checkpoint passes all scenes."""
    config = {"experiment": {"results": {"keep_checkpoints": 2, "keep_best_screening": True}}}
    actor = Actor(kind="state", action_size=4)
    state = ExampleTrainingState(
        params={"params": {"weight": jnp.ones(2)}},
        optimizer_state={"actor": jnp.zeros(1)},
        rng=jax.random.key(3),
        updates=0,
    )
    for update, successes in ((10, 1), (20, 7), (30, 2), (40, 0), (50, 0)):
        destination = tmp_path / f"model_{update:06d}"
        save_model(
            destination,
            state.replace(updates=update),
            config=config,
            provenance={"updates": update},
            kind="state",
            actor_spec=actor.specification,
        )
        screening = destination / "screening"
        screening.mkdir()
        (screening / "report.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "reports": {
                        "S01": {"task": "navigation", "episodes": 8, "successes": successes}
                    },
                }
            )
        )
    assert sorted(path.name for path in tmp_path.glob("model_*")) == [
        "model_000020",
        "model_000040",
        "model_000050",
    ]


def test_legacy_run_keeps_old_log_contract(tmp_path):
    """Existing archive runs retain their JSONL log format when resumed."""
    config = OmegaConf.create({"seed": 0, "results": {"format": "legacy"}})
    record = RunRecord(tmp_path / "legacy", config)
    record.event("update", update=1, reward=0.5)
    record.finish("interrupted", updates=1)
    assert record.layout_version == 2
    assert (record.directory / "metrics.jsonl").is_file()
    assert not list(record.directory.glob("events.out.tfevents.*"))
