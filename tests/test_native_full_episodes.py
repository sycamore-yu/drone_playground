"""Full-episode acceptance must not truncate Command Builder tasks."""

import argparse
import shlex
from collections import Counter

import numpy as np

from drone_playground.simulation.scene import Scene
from drone_playground.simulation.tasks import Task
from tools.validate_full_episodes import (
    TASK_DURATION,
    build_command,
    builder_catalog,
    case_slug,
    physical_duration,
)


def test_full_episode_catalog_uses_complete_task_horizons(tmp_path):
    """All native method commands retain real Task durations and replay capture."""
    args = argparse.Namespace(ego_port=50151, super_port=50152)
    cases = builder_catalog()
    assert len(cases) == 172
    assert TASK_DURATION == {"tracking": 20.0, "racing": 60.0, "navigation": 300.0}
    assert len({case_slug(i, case) for i, case in enumerate(cases)}) == len(cases)
    assert Counter(case["mode"] + ":" + case["method"] for case in cases) == {
        "play:controller": 8,
        "play:planner": 120,
        "benchmark:controller": 8,
        "benchmark:planner": 36,
    }
    for index, case in enumerate(cases):
        arguments = build_command(case, tmp_path / str(index), args)
        assert arguments[1:3] == ["-m", "drone_playground.cli"]
        assert "benchmark.episodes=1" in arguments
        assert "benchmark.record_replay=true" in arguments
        assert "benchmark.replay_episodes=1" in arguments
        assert not any(arg.startswith("task.duration=") for arg in arguments)
        assert physical_duration(case) in TASK_DURATION.values()
        published = shlex.split(case["command"])[3:]
        for item in published:
            assert item in arguments


def test_only_real_task_duration_defines_timeout(tmp_path):
    """A task remains 20, 60 or 300 seconds, independent of the chosen episode count."""
    args = argparse.Namespace(ego_port=50151, super_port=50152)
    samples = builder_catalog()
    for task, expected in [("tracking", 20.0), ("racing", 60.0), ("static_oa", 300.0)]:
        case = next(x for x in samples if x["task"] == task)
        command = build_command(case, tmp_path / task, args)
        assert physical_duration(case) == expected
        assert "task.duration=0.14" not in command
        assert not any("duration=" in token for token in command)


def test_non_native_commands_are_not_accidentally_in_full_runner():
    """No policy checkpoint or learning command is inferred from the native catalog."""
    for case in builder_catalog():
        assert case["method"] in {"controller", "planner"}
        assert case["mode"] in {"play", "benchmark"}
        assert "checkpoint_eval" not in case["command"]


def test_short_tracking_budget_changes_reference_dynamics_not_just_episode_length():
    """The historical 0.14-second test must never be used as tracking-quality evidence."""
    scene = Scene("empty")
    full = Task("tracking", scene, reference="figure_eight", duration=20.0)
    truncated = Task("tracking", scene, reference="figure_eight", duration=0.14)
    _, full_velocity, _ = full.target(np.asarray([0.0]))
    _, short_velocity, _ = truncated.target(np.asarray([0.0]))
    assert np.isclose(np.asarray(full_velocity)[0, 0], 2 * np.pi / 20.0)
    assert np.isclose(np.asarray(short_velocity)[0, 0] / np.asarray(full_velocity)[0, 0], 20 / 0.14)
