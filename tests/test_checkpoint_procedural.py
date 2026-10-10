"""Orbax roundtrips preserve independently generated scenario state."""

import jax
import numpy as np

from drone_playground.learning.checkpoint import load_model_state, save_model
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import make_environment


def test_procedural_empty_state_actor_memory_roundtrip(tmp_path):
    """Do not change Actor's zero-length memory contract to accommodate Orbax."""
    env = make_environment(
        task="navigation",
        scene="train/procedural",
        num_envs=2,
        device="cpu",
        sensor="state",
        procedural={"max_obstacles": 6},
    )
    trainer = Trainer(env, kind="state", algorithm="ppo", config={"horizon": 1, "ppo_epochs": 1})
    state = trainer.initialize()
    assert state.recurrent_memory.shape == (2, 0)
    directory = tmp_path / "model_000001"
    save_model(
        directory,
        state,
        config={"experiment": {"results": {"keep_checkpoints": 3}}},
        provenance={"updates": 0},
        kind="state",
        actor_spec=trainer.actor.specification,
    )
    restored, metadata = load_model_state(directory, trainer.initialize())
    assert metadata["padded_empty_recurrent_memory"]
    assert restored.recurrent_memory.shape == (2, 0)
    np.testing.assert_allclose(
        restored.env_state.scene_state.centers, state.env_state.scene_state.centers
    )
    np.testing.assert_array_equal(
        restored.env_state.scene_state.episode, state.env_state.scene_state.episode
    )
    assert jax.tree.structure(restored) == jax.tree.structure(state)
