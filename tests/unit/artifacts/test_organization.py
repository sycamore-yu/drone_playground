import hashlib
import importlib.util
import json
import shutil
import sys

import pytest

from tests.helpers.paths import REPO_ROOT


@pytest.fixture(scope="module")
def organizer():
    path = REPO_ROOT / "scripts/tools/organize_experiments.py"
    spec = importlib.util.spec_from_file_location("experiment_organizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_result_date_requires_recorded_start_time(tmp_path, organizer):
    run = tmp_path / "run-20260929"
    run.mkdir()
    organizer.write(
        run / "manifest.json", dict(started_at="2026-09-30T01:00:00+00:00")
    )
    assert organizer.run_date(run) == "260930"
    (run / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        organizer.run_date(run)


def test_selection_manifest_references_runs_without_copying_artifacts(tmp_path, organizer):
    run = tmp_path / "results/runs/navigation/pointcloud/cloud-trial"
    weights = run / "training-state/update-0000125.pkl"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"trusted original checkpoint bytes")
    organizer.write(
        weights.with_suffix(".json"),
        dict(
            sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            parameter_sha256="params",
        ),
    )
    organizer.write(
        run / "resolved-config.json",
        dict(mode="train", method=dict(trainable=True), training=dict(seed=0)),
    )
    organizer.write(
        run / "manifest.json",
        dict(started_at="2026-09-30T00:00:00+00:00", code=dict(commit="frozen")),
    )
    organizer.write(run / "result.json", dict(selected=dict(updates=125)))
    report = dict(
        quality_passed=False,
        checkpoint=str(weights),
        parameter_sha256="params",
    )
    organizer.write(run / "eval/update-0000125.json", report)
    progress = dict(
        cells=[
            dict(
                method="pointcloud",
                task="static",
                status="development_not_confirmed",
                passed=False,
            )
        ],
        passed_cells=0,
        user_accepted_cells=0,
        release_requirement_satisfied_cells=0,
        required_cells=1,
        remaining_quality_cells=1,
    )

    index = organizer.build_selection(tmp_path, progress, "test-goal", run.name)
    selected = tmp_path / "results/selected/test-goal.json"
    assert selected.is_file()
    document = json.loads(selected.read_text())
    assert document["goal"] == "test-goal"
    assert index[0]["quality_passed"] is False
    chosen = index[0]["runs"][0]
    assert chosen["run_id"] == run.name
    assert chosen["source_run"] == "results/runs/navigation/pointcloud/cloud-trial"
    assert chosen["report"] == "results/runs/navigation/pointcloud/cloud-trial/eval/update-0000125.json"
    assert chosen["checkpoint"]["path"] == str(weights.relative_to(tmp_path))
    assert chosen["checkpoint"]["sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert not (tmp_path / "results/selected/test-goal").exists()

    next_run = run.with_name("cloud-next-trial")
    shutil.copytree(run, next_run)
    organizer.write(next_run / "eval/update-0000125.json", {**report, "arrived": 6})
    next_index = organizer.build_selection(tmp_path, progress, "test-goal", next_run.name)
    assert next_index[0]["runs"][0]["run_id"] == next_run.name
    assert len(list((tmp_path / "results/selected").iterdir())) == 2


def test_migration_separates_real_runs_from_legacy_scratch(tmp_path, organizer):
    real = tmp_path / "results/tmp/261001/real-run"
    real.mkdir(parents=True)
    organizer.write(
        real / "resolved-config.json",
        {
            "env": {"task": {"name": "tracking"}},
            "method": {"name": "ppo"},
        },
    )
    legacy = tmp_path / "results/tmp/261001/diagnostic-bundle"
    legacy.mkdir(parents=True)

    plan = {
        str(source.relative_to(tmp_path)): str(destination.relative_to(tmp_path))
        for source, destination in organizer.migration_plan(tmp_path)
    }
    assert plan["results/tmp/261001/real-run"] == (
        "results/runs/tracking/ppo/real-run"
    )
    assert plan["results/tmp/261001/diagnostic-bundle"] == (
        "results/scratch/legacy-runs/261001/diagnostic-bundle"
    )


def test_selection_cli_uses_current_progress_or_explicit_archive(tmp_path, organizer, monkeypatch, capsys):
    organizer.write(tmp_path / "artifacts/verification/release-progress.json", {"cells": []})
    legacy = tmp_path / "results/scratch/legacy/main_result/test-goal"
    package = legacy / "01-ppo-tracking/seed0"
    cell = {
        "cell": "01-ppo-tracking", "method": "ppo", "task": "tracking",
        "status": "passed", "quality_passed": True, "release_requirement_satisfied": True,
        "runs": [{"run_id": "frozen-trial", "training_seed": 0}],
    }
    organizer.write(legacy / "index.json", {"cells": [cell]})
    organizer.write(package / "selection.json", {"run_id": "frozen-trial"})
    frozen = tmp_path / "frozen/frozen-trial"
    organizer.write(frozen / "eval/report.json", {"num_trials": 1, "completed": 1})
    (package / "original-run").symlink_to(frozen, target_is_directory=True)
    arguments = ["organize_experiments.py", "--root", str(tmp_path), "--goal", "test-goal"]
    monkeypatch.setattr(sys, "argv", arguments)
    organizer.main()
    assert json.loads(capsys.readouterr().out)["source"] == "artifacts/verification/release-progress.json"
    monkeypatch.setattr(sys, "argv", [*arguments, "--import-legacy", "--apply"])
    organizer.main()
    selected = organizer.read(tmp_path / "results/selected/test-goal.json")["cells"][0]
    assert selected["quality_passed"] is True
    chosen = selected["runs"][0]
    assert (tmp_path / chosen["source_run"]).samefile(frozen)
    assert (tmp_path / chosen["report"]).samefile(frozen / "eval/report.json")
