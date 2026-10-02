"""Recurrent task trainers must retain best selection across exact continuation."""

import copy
import json

import jax
import jax.numpy as jnp
import pytest

from drone_playground.learning.algorithms.recurrent_bptt import TrainingState


def test_selection_survives_resume_without_reselecting_a_worse_model(tmp_path):
    from drone_playground.learning.checkpointing import (
        continuation_contract,
        restore_recurrent_state,
        save_recurrent_snapshot,
    )

    config = {"config_version": 3, "training": {"resume": None}, "runtime": {"device": "cpu"}}
    state = TrainingState({"weight": jnp.ones(2)}, {}, jax.random.PRNGKey(2), jnp.int32(1))
    first = tmp_path / "training-state/update-0000001.pkl"
    best, best_report = save_recurrent_snapshot(
        first, state, config, {"quality_passed": True}, (0.9, -1.0), None, None,
        report_path=tmp_path / "eval/update-0000001.json",
    )
    second = first.with_name("update-0000002.pkl")
    state = state.replace(updates=jnp.int32(2), params={"weight": jnp.zeros(2)})
    best, best_report = save_recurrent_snapshot(
        second, state, config, {"quality_passed": False}, (0.1, -2.0), best, best_report,
        report_path=tmp_path / "eval/update-0000002.json",
    )
    requested = copy.deepcopy(config)
    requested["training"]["resume"] = str(second)
    restored, metadata = restore_recurrent_state(state, requested, continuation_contract)
    assert int(restored.updates) == 2
    assert metadata["selection"]["checkpoint"] == str(first.resolve())
    assert metadata["selection_report"]["quality_passed"] is True
    assert json.loads((tmp_path / "best.json").read_text()) == best


def test_resume_rejects_changed_training_contract(tmp_path):
    from drone_playground.learning.checkpointing import continuation_contract, restore_recurrent_state
    from drone_playground.artifacts.training_state import save_training_state

    config = {"config_version": 3, "training": {"resume": None, "num_envs": 2}, "runtime": {"device": "cpu"}}
    state = TrainingState({}, {}, jax.random.PRNGKey(1), jnp.int32(2))
    path = tmp_path / "state.pkl"
    save_training_state(path, state, config)
    config["training"].update(resume=str(path), num_envs=3)
    with pytest.raises(ValueError, match="continuation contract"):
        restore_recurrent_state(state, config, continuation_contract)
