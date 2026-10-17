"""Publish Task Preview only from physical Episodes with available WASM/RScope."""

import csv
import json

import pytest

from tools import build_task_preview_catalog as preview


def write_csv(path, rows):
    """Write one native Episode table with the relevant physical columns."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene", "episode", "event", "flight_seconds"])
        writer.writeheader()
        writer.writerows(rows)


def test_catalog_preserves_native_episode_status_and_scene_names(tmp_path, monkeypatch):
    """Mixed Benchmark results must map to their exact saved scene, not a fake success."""
    monkeypatch.setattr(preview, "REPO", tmp_path)
    root = tmp_path / "results" / "acceptance"
    root.mkdir(parents=True)
    cases = [
        {
            "mode": "benchmark",
            "method": "planner",
            "task": "static_oa",
            "planner": "ego",
            "controller": "sampling_mpc",
            "suite": "core",
            "command": "pixi run benchmark task=navigation sensor=depth method=ego",
        },
        {
            "mode": "play",
            "method": "controller",
            "task": "racing",
            "controller": "racing_mpcc",
            "command": "pixi run play task=racing controller=racing_mpcc",
        },
    ]
    (root / "catalog.json").write_text(json.dumps({"cases": cases}))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "cases": {
                    "000_ego": {
                        "status": "method_failure",
                        "run_relative": "runs/000_ego",
                        "task_duration_seconds": 300.0,
                        "source_revision": "feature-test",
                        "browser_replays": [
                            "replays/S01-0000/browser_replay.json",
                            "replays/S06-0000/browser_replay.json",
                        ],
                    },
                    "001_racing": {
                        "status": "completed",
                        "run_relative": "runs/001_racing",
                        "task_duration_seconds": 60.0,
                        "browser_replays": ["replays/racing-0000/browser_replay.json"],
                    },
                }
            }
        )
    )
    ego = root / "runs/000_ego"
    ego.mkdir(parents=True)
    write_csv(
        ego / "episodes.csv",
        [
            {"scene": "S01", "episode": "0", "event": "SUCCESS", "flight_seconds": "32.4"},
            {"scene": "S06", "episode": "0", "event": "METHOD_FAILURE", "flight_seconds": "6.4"},
        ],
    )
    (ego / "decisions.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {"scene": "S06", "episode": 0, "detail": "NO_SOLUTION: original EGO failed"}
                ]
            }
        )
    )
    (ego / "report.json").write_text("{}")
    for scene in ["S01", "S06"]:
        saved = ego / f"replays/{scene}-0000"
        saved.mkdir(parents=True)
        (saved / "browser_replay.json").write_text("{}")
        (saved / "episode.mj_unroll").write_bytes(b"physical-poses")

    racing = root / "runs/001_racing"
    racing.mkdir(parents=True)
    write_csv(
        racing / "episodes.csv",
        [{"scene": "racing", "episode": "0", "event": "SUCCESS", "flight_seconds": "14.72"}],
    )
    saved = racing / "replays/racing-0000"
    saved.mkdir(parents=True)
    (saved / "browser_replay.json").write_text("{}")
    (saved / "episode.mj_unroll").write_bytes(b"physical-poses")

    document = preview.build(root)
    assert document["summary"] == {
        "native_methods": 2,
        "recorded_methods": 2,
        "recorded_episodes": 3,
        "terminal_events": {"SUCCESS": 2, "METHOD_FAILURE": 1},
    }
    a, b = document["cases"]
    assert [x["scene"] for x in a["recordings"]] == ["S01", "S06"]
    assert a["recordings"][1]["elapsed_seconds"] == 6.4
    assert a["recordings"][1]["event"] == "METHOD_FAILURE"
    assert "NO_SOLUTION" in a["recordings"][1]["failure_detail"]
    assert a["recordings"][1]["browser_record"].startswith("/results/acceptance/runs/")
    assert "/tmp/" not in a["recordings"][1]["browser_record"]
    assert b["recordings"][0]["elapsed_seconds"] == 14.72
    output = preview.publish(root)
    assert json.loads(output.read_text()) == document


def test_source_outside_checkout_is_rejected(tmp_path, monkeypatch):
    """Never publish arbitrary filesystem locations as same-origin previews."""
    monkeypatch.setattr(preview, "REPO", tmp_path / "checkout")
    with pytest.raises(ValueError, match="outside the repository"):
        preview._url(tmp_path / "secret.txt")
