"""External runner translation without importing either simulator stack."""

import csv
import json
from types import SimpleNamespace

import pytest


def test_candidate_cannot_change_fixed_evaluation_or_resume(tmp_path):
    """Only training options cross the mutable candidate boundary."""
    from tools.autoresearch import DronePlaygroundBackend

    backend = DronePlaygroundBackend()
    bundle = SimpleNamespace(
        target_name="local",
        train_seconds=None,
        train_episodes=0,
        eval_episodes=2,
        max_steps=3000,
        execution_backend={"train_env_steps": 8192},
        benchmark=SimpleNamespace(
            env_kwargs={
                "experiment": "navigation_procedural_depth_ppo_validation",
                "algorithm": "ppo",
                "eval_scenes": ["S01", "D01"],
                "eval_mode": "checkpoint_eval",
            },
            train_seed=7,
            eval_seed_start=1_600_000,
            device="external",
        ),
        candidate_metadata={"recipe": {"learning.options.lr": 0.0001}},
        external_dir=tmp_path,
    )
    backend.build_bundle(bundle)
    train = backend.overrides(bundle, "train")
    evaluate = backend.overrides(bundle, "eval", tmp_path / "model")
    assert "learning.max_interactions=8192" in train
    assert "seed=7" in train and "initial_checkpoint=null" in train
    assert "seed=0" in evaluate and "benchmark.seed_base=1600000" in evaluate
    assert "benchmark.episodes=1" in evaluate
    assert 'benchmark.scenes=["S01", "D01"]' in evaluate
    for key in ("benchmark.scenes", "seed", "resume", "task.duration"):
        bundle.candidate_metadata["recipe"] = {key: 1}
        with pytest.raises(ValueError, match="candidate"):
            backend.build_bundle(bundle)


def test_eval_translation_keeps_failures_and_missing_return(tmp_path):
    """Measure success over all requested scenes without fabricating reward."""
    from tools.autoresearch import DronePlaygroundBackend

    (tmp_path / "request.json").write_text(
        json.dumps({"eval_scenes": ["S01", "D01"], "eval_episodes": 2})
    )
    destination = tmp_path / "eval"
    destination.mkdir()
    (destination / "config.yaml").write_text("simulation:\n  method_hz: 10\n")
    (destination / "report.json").write_text(
        json.dumps({"complete": True, "passed": None, "reports": {"S01": {}, "D01": {}}})
    )
    with (destination / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene", "event", "flight_seconds"])
        writer.writeheader()
        writer.writerows(
            [
                {"scene": "S01", "event": "SUCCESS", "flight_seconds": 30},
                {"scene": "D01", "event": "COLLISION", "flight_seconds": 10},
            ]
        )
    result = DronePlaygroundBackend().normalize_eval(SimpleNamespace(root=tmp_path))
    assert result["success_rate"] == result["collision_rate"] == 0.5
    assert result["worst_scene_success_rate"] == 0
    assert result["avg_return"] is None
    assert len(result["episode_records"]) == 2
    (destination / "report.json").write_text('{"complete": false}')
    with pytest.raises(ValueError, match="Incomplete"):
        DronePlaygroundBackend().normalize_eval(SimpleNamespace(root=tmp_path))
