"""One-time result migration preserves archives, cases and the original run."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_migrator():
    """Load the standalone maintenance command without importing simulation."""
    path = Path(__file__).parents[1] / "tools/migrate_results.py"
    spec = importlib.util.spec_from_file_location("migrate_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_run(path):
    """Create a small legacy run with real numeric evidence and opaque weights."""
    path.mkdir()
    (path / "config.yaml").write_text("seed: 1\n")
    (path / "events").mkdir()
    (path / "events/metrics.jsonl").write_text('{"event":"update","update":25}\n')
    (path / "rollouts").mkdir()
    (path / "checkpoints").mkdir()
    (path / "checkpoints/update-00000025.policy.zip").write_bytes(b"immutable policy archive")
    (path / "checkpoints/latest.training.zip").write_bytes(b"immutable training archive")
    selection = {
        "checkpoint": f"results/{path.name}/checkpoints/update-00000025.policy.zip",
        "update": 25,
    }
    (path / "selection.json").write_text(json.dumps(selection))
    (path / "run.json").write_text(
        json.dumps(
            {
                "status": "accepted",
                "selected_checkpoint": selection["checkpoint"],
                "benchmark_directory": f"results/{path.name}/benchmark",
            }
        )
    )
    for directory, mode in [
        ("checkpoint_eval/update-00000025", "checkpoint_eval"),
        ("benchmark", "benchmark"),
    ]:
        reports = {}
        for scene, outcome in [("S01", "SUCCESS"), ("D01", "COLLISION")]:
            p = path / directory / scene
            p.mkdir(parents=True)
            (p / "episodes.csv").write_text(
                f"scene,event,world_index,initialization_seed\n{scene},{outcome},0,100\n"
            )
            reports[scene] = {"episodes": 1, "successes": int(outcome == "SUCCESS"), "replays": []}
            (p / "report.json").write_text(json.dumps(reports[scene]))
            np.savez_compressed(p / "trajectories.npz", position=np.array([[1.0, 2.0, 3.0]]))
        (path / directory / "report.json").write_text(
            json.dumps({"mode": mode, "reports": reports, "passed": False})
        )
    return path


def test_migration_is_copy_only_and_preserves_case_denominators(tmp_path):
    """Flatten tables without dropping failures, changing weights or rewriting the source."""
    migrate = load_migrator()
    source = old_run(tmp_path / "old")
    before = migrate.fingerprints(source)
    destination = tmp_path / "new"
    migrate.migrate_run(source, destination, apply=False)
    assert not destination.exists()
    migrate.migrate_run(source, destination, apply=True)
    assert migrate.fingerprints(source) == before
    assert (
        destination / "checkpoints/step-000025/policy.zip"
    ).read_bytes() == b"immutable policy archive"
    assert (
        destination / "checkpoints/latest.training.zip"
    ).read_bytes() == b"immutable training archive"
    assert not (destination / "selection.json").exists()
    assert not (destination / "events").exists() and not (destination / "rollouts").exists()
    for directory in ("checkpoints/step-000025", "eval/001"):
        with (destination / directory / "episodes.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2 and sum(row["event"] == "COLLISION" for row in rows) == 1
        with np.load(destination / directory / "trajectories.npz") as arrays:
            assert len(arrays.files) == 2
            for array in arrays.values():
                np.testing.assert_array_equal(array, [[1.0, 2.0, 3.0]])
    identity = json.loads((destination / "run.json").read_text())
    assert identity["selection"]["checkpoint"] == "checkpoints/step-000025/policy.zip"
    assert identity["benchmark_directory"] == "eval/001"
    with pytest.raises(FileExistsError):
        migrate.migrate_run(source, destination, apply=True)


def test_migration_refuses_running_or_nested_destination(tmp_path):
    """Do not copy a changing run or place its destination inside the source."""
    migrate = load_migrator()
    source = old_run(tmp_path / "old")
    (source / "run.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="running"):
        migrate.migrate_run(source, tmp_path / "new", apply=True)
    (source / "run.json").write_text('{"status":"accepted"}')
    with pytest.raises(ValueError, match="inside"):
        migrate.migrate_run(source, source / "nested", apply=True)


@pytest.mark.parametrize("entry", ["benchmark/S01", "run.json"])
def test_migration_rejects_links_that_would_modify_external_evidence(tmp_path, entry):
    """A copied symlink must not turn migration into a write to the source evidence."""
    migrate = load_migrator()
    source = old_run(tmp_path / "old")
    path = source / entry
    target = tmp_path / "external"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    before = migrate.fingerprints(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        migrate.migrate_run(source, tmp_path / "new", apply=True)
    assert migrate.fingerprints(tmp_path) == before
