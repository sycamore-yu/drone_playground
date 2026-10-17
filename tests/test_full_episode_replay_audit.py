"""Replay coverage is judged against actual termination, not the task time limit."""

import csv
import json

from tools.audit_full_episode_replays import _browser_record, audit


def _write_record(path, *, last: float, source: int = 61):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format": "drone-playground-mujoco-replay-v1",
                "original_frame_count": source,
                "frames": [{"time": 0.0}, {"time": last}],
            }
        )
    )


def test_true_1p2_second_failure_recording_is_not_a_missing_300_seconds(tmp_path):
    """A real 1.2 s terminated flight with matching saved frames is faithful."""
    record = tmp_path / "browser_replay.json"
    _write_record(record, last=1.2)
    outcome = _browser_record(record, episode_seconds=1.2)
    assert outcome["status"] == "covers_recorded_episode"
    assert outcome["last_time"] == 1.2
    assert outcome["source_frame_count"] == 61


def test_actual_truncated_or_missing_record_is_identified(tmp_path):
    """Do not incorrectly accept a 1.0 s video for a 1.2 s flight."""
    record = tmp_path / "browser_replay.json"
    assert _browser_record(record, 1.2)["status"] == "missing_record"
    _write_record(record, last=1.0)
    assert _browser_record(record, 1.2)["status"] == "incomplete_recording"


def test_replay_audit_preserves_planner_error_and_short_episode(tmp_path):
    """The index must expose failure reason separately from recorder integrity."""
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "mode": "play",
                        "method": "planner",
                        "task": "static_oa",
                        "planner": "ego",
                        "controller": "full_state_nmpc",
                    }
                ]
            }
        )
    )
    run = tmp_path / "runs/000_fail"
    run.mkdir(parents=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "cases": {
                    "000_fail": {
                        "status": "method_failure",
                        "run_relative": "runs/000_fail",
                        "task_duration_seconds": 300,
                        "browser_replays": ["replays/S06-0000/browser_replay.json"],
                    }
                }
            }
        )
    )
    with (run / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["episode", "scene", "event", "flight_seconds"])
        writer.writeheader()
        writer.writerow(
            {"episode": 0, "scene": "S06", "event": "METHOD_FAILURE", "flight_seconds": 1.2}
        )
    (run / "decisions.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "status": "RuntimeError",
                        "detail": "Planner trajectory does not cover the current controller step",
                    }
                ]
            }
        )
    )
    record = run / "replays/S06-0000/browser_replay.json"
    _write_record(record, last=1.2)
    result = audit(tmp_path)
    assert result["replay_integrity"] == {"covers_recorded_episode": 1}
    assert result["event_counts"] == {"METHOD_FAILURE": 1}
    row = result["rows"][0]
    assert row["ended_early"] is True
    assert row["task_limit_seconds"] == 300
    assert row["physical_flight_seconds"] == 1.2
    assert row["failure_detail"].startswith("Planner trajectory")


def test_running_episode_is_not_counted_as_missing_replay(tmp_path):
    """Ignore an in-flight Episode while the native replay serializer runs."""
    (tmp_path / "catalog.json").write_text(
        json.dumps({"cases": [{"mode": "play", "method": "planner", "task": "static_oa"}]})
    )
    directory = tmp_path / "runs/000_inflight"
    directory.mkdir(parents=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "cases": {
                    "000_inflight": {
                        "status": "running",
                        "run_relative": "runs/000_inflight",
                        "browser_replays": [],
                    }
                }
            }
        )
    )
    (directory / "episodes.csv").write_text(
        "episode,scene,event,flight_seconds\n0,S06,SUCCESS,32.0\n"
    )
    result = audit(tmp_path)
    assert result["in_progress_cases"] == ["000_inflight"]
    assert result["event_counts"] == {}
    assert result["replay_integrity"] == {}
    assert result["rows"] == []
