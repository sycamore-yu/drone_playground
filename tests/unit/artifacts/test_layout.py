from datetime import UTC
from pathlib import Path

import pytest

from drone_playground.artifacts.layout import (
    experiment_directory,
    find_experiment,
    iter_experiments,
)


def scoped(task="tracking", method="ppo"):
    return {"env": {"task": {"name": task}}, "method": {"name": method}}


def test_new_runs_are_grouped_by_task_and_method(tmp_path):
    target = experiment_directory(tmp_path, "trial-a", config=scoped())
    assert target == tmp_path / "results/runs/tracking/ppo/trial-a"
    target.mkdir(parents=True)
    assert experiment_directory(tmp_path, "trial-a", config=scoped("racing", "shac")) == target
    assert find_experiment(tmp_path, "trial-a") == target


def test_retired_flat_directories_are_not_current_runs(tmp_path):
    old = tmp_path / "results/legacy-run"
    old.mkdir(parents=True)
    assert find_experiment(tmp_path, old.name) is None
    assert experiment_directory(tmp_path, old.name, config=scoped()) != old


def test_new_run_creation_requires_task_method_config(tmp_path):
    with pytest.raises(ValueError, match=r"Task/Method"):
        experiment_directory(tmp_path, "unscoped-run")


def test_ambiguous_run_identity_is_rejected(tmp_path):
    for task, method in [("tracking", "ppo"), ("racing", "shac")]:
        (tmp_path / "results/runs" / task / method / "duplicate").mkdir(parents=True)
    with pytest.raises(ValueError, match=r"Ambiguous"):
        find_experiment(tmp_path, "duplicate")


def test_artifact_reader_uses_the_exact_supplied_path(tmp_path):
    moved = tmp_path / "results/runs/navigation/ppo/run-a/training-state/model.pkl"
    moved.parent.mkdir(parents=True)
    moved.write_bytes(b"original checkpoint")
    old = tmp_path / "results/run-a/training-state/model.pkl"
    assert Path(old) == old
    assert Path(moved) == moved
    unrelated = tmp_path / "some-other-data/missing.pkl"
    assert Path(unrelated) == unrelated


def test_listing_excludes_selected_and_scratch(tmp_path):
    for relative in [
        "results/runs/tracking/ppo/first",
        "results/runs/navigation/super/second",
        "results/selected",
        "results/scratch/previews/demo",
    ]:
        (tmp_path / relative).mkdir(parents=True)
    assert {p.name for p in iter_experiments(tmp_path)} == {"first", "second"}


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "/absolute", ".", "..", "runs", "selected", "scratch", "tmp", "main_result"],
)
def test_run_identifiers_cannot_escape_or_collide_with_layout_names(tmp_path, run_id):
    with pytest.raises(ValueError):
        experiment_directory(tmp_path, run_id, config=scoped())


def test_cli_status_reads_scoped_runs(tmp_path, monkeypatch, capsys):
    import json
    from datetime import datetime

    from drone_playground import cli

    for name, task, method in [
        ("first", "tracking", "ppo"),
        ("second", "navigation", "super"),
    ]:
        run = experiment_directory(tmp_path, name, config=scoped(task, method))
        run.mkdir(parents=True)
        (run / "state.json").write_text(
            json.dumps(
                dict(
                    status="completed",
                    updated_at=datetime.now(UTC).isoformat(),
                )
            )
        )
    monkeypatch.setattr(cli, "ROOT", Path(tmp_path))
    cli.main(["status"])
    rows = json.loads(capsys.readouterr().out)
    assert {row["run_id"] for row in rows} == {"first", "second"}


def test_cli_replay_uses_explicit_scratch_rollout(tmp_path, monkeypatch, capsys):
    from drone_playground import cli
    from drone_playground.visualization import rscope_publish

    replay = tmp_path / "results/scratch/replays/replay-run"
    replay.mkdir(parents=True)
    observed = []
    monkeypatch.setattr(rscope_publish, "publish_run", lambda path: observed.append(path) or path)
    cli.main(["replay", "--directory", str(replay)])
    assert observed == [replay]


def test_play_resolves_explicit_replay_before_dispatch(tmp_path, monkeypatch):
    from drone_playground.app import run_experiment
    from drone_playground.configuration import compose_experiment
    from drone_playground.visualization import viewer

    replay = tmp_path / "results/scratch/replays/replay-run"
    replay.mkdir(parents=True)
    original = str(replay)
    config = compose_experiment(overrides=["mode=play", "runtime.device=cpu"])
    config["replay"]["directory"] = original
    monkeypatch.setattr(viewer, "replay", lambda path, **kwargs: {"directory": str(path)})
    result = run_experiment(config, tmp_path, "view")
    assert result["directory"] == str(replay)
    assert config["replay"]["directory"] == original
