"""Verify recorded evaluation facts retain the caller's chosen task criterion."""

import csv
import json

import numpy as np
import pytest

from drone_playground.simulation.environment import Environment
from drone_playground.simulation.runner import rollout, save_evaluation


def test_score_only_rollout_preserves_all_episode_facts(tmp_path, monkeypatch):
    """Keep every outcome while omitting unused trace computation and storage."""
    from drone_playground.simulation import runner
    from drone_playground.simulation.records import acceptance

    env = Environment(task="navigation", scene="S01", sensor="depth", num_envs=3, duration=0.06)
    complete, recorded = rollout(env, seed=11, chunk_steps=2)
    assert recorded["position"].shape[1] == 3

    def forbidden_sample(*args):
        raise AssertionError("Score-only rollout sampled replay data")

    monkeypatch.setattr(runner, "_sample", forbidden_sample)
    outcomes, traces = rollout(env, seed=11, chunk_steps=2, record=False)
    assert traces == {}
    for expected, actual in zip(complete, outcomes, strict=True):
        assert {k: v for k, v in expected.items() if k != "evaluation_wall_seconds"} == {
            k: v for k, v in actual.items() if k != "evaluation_wall_seconds"
        }
    report = save_evaluation(
        tmp_path,
        env,
        outcomes,
        traces,
        report=acceptance("navigation", outcomes),
        replay_episodes=0,
    )
    assert report["episodes"] == 3
    assert len(list(csv.DictReader((tmp_path / "episodes.csv").open()))) == 3
    assert not (tmp_path / "trajectories.npz").exists()
    assert json.loads((tmp_path / "report.json").read_text())["replays"] == []


def test_empty_traces_do_not_create_an_empty_archive(tmp_path):
    """Do not allocate an archive when a completed evaluation has no requested replay."""
    env = Environment(duration=0.02)
    episodes, _ = rollout(env, seed=7)
    save_evaluation(
        tmp_path,
        env,
        episodes,
        {},
        report={"criterion": "diagnostic", "passed": None},
        replay_episodes=0,
    )
    assert not (tmp_path / "trajectories.npz").exists()


@pytest.mark.parametrize("criterion, passed", [("diagnostic", None), ("C4", True)])
def test_recording_preserves_selected_criterion_and_all_outcomes(tmp_path, criterion, passed):
    """Persist the selected criterion with successes and collisions in the same CSV."""
    environment = Environment(task="navigation", scene="S01", sensor="state")
    episodes = [
        {
            "scene": "S01",
            "method": "ego" if criterion == "diagnostic" else "policy",
            "event": "SUCCESS" if index < 23 else "COLLISION",
            "world_index": index,
            "initialization_seed": 2_000_000,
        }
        for index in range(25)
    ]
    traces = {"time": np.broadcast_to(np.array([[0.0], [1.0]]), (2, 25))}
    selected_report = {
        "criterion": criterion,
        "passed": passed,
        "episodes": 25,
        "successes": 23,
        "success_rate": 23 / 25,
        "successful_position_rmse": None,
    }
    report = save_evaluation(
        tmp_path,
        environment,
        episodes,
        traces,
        report=selected_report,
        replay_episodes=0,
    )
    saved = json.loads((tmp_path / "report.json").read_text())
    assert report["criterion"] == criterion
    assert report["passed"] is passed
    assert saved["criterion"] == criterion
    assert saved["passed"] is passed
    with (tmp_path / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 25
    assert sum(row["event"] == "SUCCESS" for row in rows) == 23
    assert sum(row["event"] == "COLLISION" for row in rows) == 2
    with np.load(tmp_path / "trajectories.npz") as saved_traces:
        np.testing.assert_array_equal(saved_traces["time"], traces["time"])
