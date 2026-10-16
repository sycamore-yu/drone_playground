"""External runner translation without importing either simulator stack."""

import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_dashboard_uses_upstream_assets(tmp_path, monkeypatch):
    """Serve the UI without copying the upstream checkout into the project."""
    from tools import autoresearch

    project = tmp_path / "project"
    upstream = tmp_path / "autoresearch-gym"
    project.mkdir()
    (upstream / "dashboard").mkdir(parents=True)
    (upstream / "dashboard" / "index.html").write_text("dashboard")
    monkeypatch.setattr(autoresearch, "ROOT", project)
    monkeypatch.setattr(autoresearch, "UPSTREAM", upstream)
    autoresearch.ensure_dashboard_assets()
    autoresearch.ensure_dashboard_assets()
    assert (project / "dashboard").is_symlink()
    assert (project / "dashboard" / "index.html").read_text() == "dashboard"


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
                "experiment": "navigation_procedural_depth_ppo_collision_free",
                "algorithm": "ppo",
                "eval_scenes": ["S01", "D01"],
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
    assert "++learning.budget_only=true" not in train
    assert "mode=benchmark" in evaluate
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


@pytest.mark.parametrize("algorithm", ["ppo", "bptt", "shac"])
def test_public_training_recipe_prepares_an_external_candidate(tmp_path, monkeypatch, algorithm):
    """Every local algorithm uses the same external orchestration and explicit budget."""
    from tools import autoresearch

    project = tmp_path / "project"
    project.mkdir()
    (project / "src").symlink_to(autoresearch.ROOT / "src", target_is_directory=True)
    monkeypatch.setattr(autoresearch, "ROOT", project)
    arguments = autoresearch.prepare_training(
        [
            "experiment=tracking",
            f"learning={algorithm}",
            "simulation.num_envs=4",
            "learning.options.horizon=8",
            "learning.max_interactions=64",
            "benchmark.episodes=2",
        ]
    )
    task = json.loads(Path(arguments[arguments.index("--benchmark") + 1]).read_text())
    assert task["execution_backend"]["train_env_steps"] == 64
    assert task["env_kwargs"]["algorithm"] == algorithm
    assert task["env_kwargs"]["eval_scenes"] == ["empty"]
    assert task["eval_episodes"] == 2
    session = Path(arguments[arguments.index("--session-dir") + 1])
    assert (project / "autoresearch_runs/sessions" / session.name).resolve() == session
    assert Path(arguments[arguments.index("--candidate") + 1]).is_file()


def test_native_curve_windows_use_complete_episode_totals(tmp_path):
    """Missing log intervals must not lose episode sums or weight windows equally."""
    from tools.autoresearch import native_training_view

    rows = [
        dict(
            event="update",
            completed_episodes=2,
            episode_return_total=10.0,
            episode_length_total=20,
            episode_success_total=1,
            interactions=30,
            wall_seconds=1.0,
        ),
        dict(
            event="update",
            completed_episodes=5,
            episode_return_total=40.0,
            episode_length_total=80,
            episode_success_total=3,
            interactions=100,
            wall_seconds=2.0,
        ),
        dict(
            event="selection",
            source="evaluation",
            mean_reward=8.0,
            interactions=100,
            wall_seconds=3.0,
            episode_records=[dict(return_=8.0, length=30, success=True, seed=1000000)],
        ),
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    view = native_training_view(tmp_path)
    curves = view["episode_records"]
    assert [row["return"] for row in curves] == [5.0, 10.0, 8.0]
    assert curves[1]["episodes_in_window"] == 3
    assert curves[1]["success_rate"] == 2 / 3
    assert curves[2]["record_type"] == "policy_probe"
    assert view["completed_episodes"] == 5 and view["avg_return"] == 8.0
    assert view["avg_length"] == 16.0 and view["success_rate"] == 0.6
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "status": "budget_exhausted",
                "interactions": 100,
                "updates": 10,
                "num_envs": 4,
                "rollout_horizon": 8,
                "algorithm": "ppo",
                "task": "navigation",
                "wall_seconds": 4.0,
                "training_wall_seconds": 4.0,
                "selection": {
                    "episodes_completed": 5,
                    "episode_return_sum": 40.0,
                    "episode_length_sum": 80,
                    "episode_successes": 3,
                },
            }
        )
    )
    final = native_training_view(tmp_path)
    assert final["policy_updates"] == 10
    assert final["num_envs"] == 4 and final["rollout_horizon"] == 8
    assert final["algorithm"] == "ppo" and final["task"] == "navigation"
    for field in (
        "completed_episodes",
        "avg_return",
        "avg_length",
        "success_rate",
        "episode_records",
    ):
        assert final[field] == view[field]


def test_native_view_keeps_missing_returns_null(tmp_path):
    """Pending runs and unfinished episodes have no measured episode mean."""
    from tools.autoresearch import native_training_view

    view = native_training_view(tmp_path)
    assert view["completed_episodes"] == 0
    assert view["avg_return"] is view["avg_length"] is view["success_rate"] is None
    (tmp_path / "metrics.jsonl").write_text('{"event": "update"')
    assert native_training_view(tmp_path)["completed_episodes"] == 0
    (tmp_path / "metrics.jsonl").write_text('{"event": "update"\n')
    with pytest.raises(json.JSONDecodeError):
        native_training_view(tmp_path)


def test_dashboard_patch_declares_environment_step_and_episode_metrics():
    """The pinned dashboard patch must preserve the public native metric units."""
    from tools import autoresearch

    patch = (autoresearch.TASKS / "upstream.patch").read_text()
    for label in (
        "Completed Episodes",
        "Policy Updates",
        "Training Mean Episode Return",
        "Environment steps",
        "Best Candidate Score",
    ):
        assert re.search(rf"(?m)^\+.*{re.escape(label)}", patch)
