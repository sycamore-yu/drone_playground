"""Verify recorded evaluation facts retain the caller's chosen task criterion."""

import csv
import json

import numpy as np
import pytest

from drone_playground.simulation.environment import Environment
from drone_playground.simulation.runner import save_evaluation


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
