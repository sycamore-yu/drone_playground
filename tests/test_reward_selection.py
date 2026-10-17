"""Episode accounting, fixed evaluation, and reward-based checkpoint identity."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground.learning.runner import train
from drone_playground.learning.selection import RewardSelection, fixed_evaluator
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.factory import create_environment
from drone_playground.simulation.policy import resolve_checkpoint
from drone_playground.simulation.records import RunRecord


@struct.dataclass
class EpisodeState:
    """The episode accumulator fields used by the shared accounting function."""

    episode_returns: jax.Array
    episode_lengths: jax.Array


def test_completed_returns_cross_updates_and_exclude_partial_episodes():
    """The mean uses completed episode sums, including their earlier updates."""
    state = EpisodeState(jnp.zeros(2), jnp.zeros(2, jnp.int32))
    state, metrics = Trainer.episode_statistics(
        None, state, jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[False, False], [True, False]])
    )
    assert float(metrics["episode_return_sum"]) == 4
    assert int(metrics["episodes_completed"]) == 1
    state, metrics = Trainer.episode_statistics(
        None, state, jnp.array([[5.0, 6.0]]), jnp.array([[False, True]])
    )
    assert float(metrics["episode_return_sum"]) == 12
    assert int(metrics["episode_length_sum"]) == 3
    np.testing.assert_array_equal(state.episode_returns, [5, 0])


def recipe(tmp_path):
    """Use a tiny real physical rollout; this is not a quality evaluation."""
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "src/drone_playground/configs")
    ):
        return compose(
            config_name="config",
            overrides=[
                "experiment=tracking",
                "learning=bptt",
                "simulation.device=cpu",
                "simulation.num_envs=2",
                "task.duration=0.04",
                "learning.options.horizon=2",
                "learning.options.minibatches=1",
                "learning.max_iterations=3",
                "learning.best.source=evaluation",
                "learning.best.interval=4",
                "learning.best.episodes=2",
                f"output={tmp_path / 'run'}",
            ],
        )


def test_fixed_evaluation_is_repeatable_and_preserves_training_state(tmp_path):
    """Evaluation consumes neither training RNG nor live environments or memory."""
    config = recipe(tmp_path)
    trainer = Trainer(
        create_environment(config),
        algorithm="bptt",
        config=OmegaConf.to_container(config.learning.options, resolve=True),
    )
    state = trainer.initialize()

    def array(value):
        if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        return np.array(value)

    before = [array(value) for value in jax.tree.leaves(state)]
    evaluate = fixed_evaluator(config, trainer)
    first, second = evaluate(state.params), evaluate(state.params)
    assert first == second and len(first) == 2
    assert all(row["length"] > 0 and np.isfinite(row["return"]) for row in first)
    for previous, current in zip(before, jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(previous, array(current))


def test_best_survives_a_worse_final_update(tmp_path, monkeypatch):
    """Run real updates while prescribing nonmonotone scores at the evaluator seam."""
    config = recipe(tmp_path)
    rewards = iter([-3.0, -1.0, -2.0])
    monkeypatch.setattr(
        "drone_playground.learning.selection.fixed_evaluator",
        lambda *_: lambda _: [{"return": next(rewards)}],
    )
    record = RunRecord(config.output, config)
    train(config, record)
    info = json.loads((record.directory / "run.json").read_text())
    args = dict(root=tmp_path, task="tracking", sensor="state", algorithm="bptt")
    assert resolve_checkpoint("latest", **args).name == "model_000003"
    assert resolve_checkpoint("best", **args).name == "model_000002"
    assert (record.directory / "best_model").resolve().name == "model_000002"
    assert info["best_mean_reward"] == -1 and info["interactions"] == 12
    metadata = json.loads(
        (record.directory / info["latest_checkpoint"] / "metadata.json").read_text()
    )
    restored = RewardSelection(
        config, SimpleNamespace(distance_curriculum=False), metadata["provenance"]["selection"]
    )
    assert restored.state["best_mean_reward"] == -1


def test_training_selection_strict_improvement_and_no_incomplete_score(tmp_path):
    """No finished episodes means no score; ties do not replace the best Actor."""
    config = recipe(tmp_path)
    config.learning.best.source = "training"
    selector = RewardSelection(config, SimpleNamespace(distance_curriculum=False))
    assert selector.measure(SimpleNamespace(interactions=4)) is None
    metrics = dict(
        episodes_completed=2, episode_return_sum=-6.0, episode_length_sum=10, episode_successes=0
    )
    selector.observe(metrics)
    assert selector.measure(SimpleNamespace(interactions=4))["mean_reward"] == -3
    selector.state["best_mean_reward"] = -3.0
    selector.observe(metrics)
    assert not selector.measure(SimpleNamespace(interactions=8))["improved"]
