"""Distinguish a short *physical episode* from an incomplete MuJoCo recording.

Read-only inspection of the native flight ledger, per-episode CSV and browser
records. Never edit or synthesize recorded poses; preserve all historical runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def _episodes(path: Path) -> list[dict]:
    file = path / "episodes.csv"
    if not file.is_file():
        return []
    with file.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _failure_details(path: Path) -> list[str]:
    file = path / "decisions.json"
    if not file.is_file():
        return []
    try:
        decisions = json.loads(file.read_text())
        if isinstance(decisions, dict):
            decisions = decisions.get("decisions", [])
        return [
            row.get("detail", "")
            for row in decisions
            if row.get("status", "") in {"RuntimeError", "RosPlannerError"} and row.get("detail")
        ]
    except (OSError, ValueError):
        return ["unreadable decision trace"]


def _browser_record(path: Path, episode_seconds: float) -> dict:
    if not path.is_file():
        return {"status": "missing_record", "path": str(path)}
    try:
        document = json.loads(path.read_text())
        frames = document.get("frames", [])
        if not frames:
            return {"status": "missing_frames", "path": str(path)}
        start = float(frames[0]["time"])
        end = float(frames[-1]["time"])
        original_count = int(document.get("original_frame_count", 0))
        valid = (
            math.isfinite(start)
            and math.isfinite(end)
            and start <= end
            and original_count >= len(frames)
        )
        # The first 20ms physics step and the final recorded sample can differ
        # by one simulation step. That is a recording tolerance, not a longer
        # permissible flight or a change to the task termination condition.
        aligned = valid and start <= 0.025 and abs(end - episode_seconds) <= 0.025
        return {
            "status": "covers_recorded_episode" if aligned else "incomplete_recording",
            "path": str(path),
            "frame_count": len(frames),
            "source_frame_count": original_count,
            "first_time": start,
            "last_time": end,
            "episode_seconds": episode_seconds,
            "end_difference_seconds": round(end - episode_seconds, 6),
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"status": "invalid_record", "path": str(path), "error": str(error)}


def audit(root: Path) -> dict:
    """Classify each physical terminal event, planner failure and replay span."""
    manifest = json.loads((root / "manifest.json").read_text())
    catalog = json.loads((root / "catalog.json").read_text())["cases"]
    rows = []
    in_progress = []
    for name, item in manifest["cases"].items():
        # Episode CSV can be written before the full RScope/MuJoCo export.
        # A running case is pending, not evidence of a truncated recording.
        if item.get("status") == "running":
            in_progress.append(name)
            continue
        index = int(name.split("_", 1)[0])
        original = catalog[index]
        run = root / item["run_relative"]
        episodes = _episodes(run)
        records = [run / name for name in item.get("browser_replays", [])]
        details = _failure_details(run)
        for episode in episodes:
            seconds = float(episode["flight_seconds"])
            paths = [
                record
                for record in records
                if record.parent.name.endswith(f"-{int(episode['episode']):04d}")
                and record.parent.name.startswith(episode["scene"] + "-")
            ]
            # Some record producers may use an opaque directory name. For the
            # single-episode case, the whole record is unambiguous.
            if not paths and len(records) == 1 and len(episodes) == 1:
                paths = records
            if not paths:
                replay = {"status": "missing_record"}
            else:
                replay = _browser_record(paths[0], seconds)
                native = paths[0].parent / "episode.mj_unroll"
                replay["native_rscope_present"] = native.is_file() and native.stat().st_size > 0
            outcome = episode["event"]
            task_limit = float(item.get("task_duration_seconds", 0))
            rows.append(
                {
                    "index": index,
                    "case": name,
                    "method": original["method"],
                    "planner": original.get("planner"),
                    "controller": original.get("controller"),
                    "scene": episode["scene"],
                    "task_event": outcome,
                    "task_limit_seconds": task_limit,
                    "physical_flight_seconds": seconds,
                    "ended_early": outcome == "METHOD_FAILURE" and seconds < task_limit,
                    "failure_detail": details[-1] if details else None,
                    "run_relative": item["run_relative"],
                    "replay": replay,
                }
            )
    # Keep superseded/failed attempts visible after a corrected run replaces
    # the manifest's current selection. Historical files must not disappear
    # from the audit just because their case passed on a later revision.
    selected = {item["run_relative"] for item in manifest["cases"].values()}
    historical = []
    for run in sorted((root / "runs").iterdir()):
        if not run.is_dir() or not run.name[:3].isdigit():
            continue
        relative = str(run.relative_to(root))
        if relative in selected:
            continue
        episodes = _episodes(run)
        details = _failure_details(run)
        if not episodes:
            console = run / "console.log"
            error = ""
            if console.is_file():
                content = console.read_text(errors="replace")
                for code in ("RESOURCE_EXHAUSTED", "STALE_INPUT", "STALE_OUTPUT"):
                    if code in content:
                        error = code
                        break
            historical.append(
                {
                    "case_index": int(run.name[:3]),
                    "run_relative": relative,
                    "task_event": None,
                    "failure_detail": error or "No completed physical episode",
                }
            )
        for episode in episodes:
            historical.append(
                {
                    "case_index": int(run.name[:3]),
                    "run_relative": relative,
                    "task_event": episode["event"],
                    "physical_flight_seconds": float(episode["flight_seconds"]),
                    "failure_detail": details[-1] if details else None,
                    "browser_record_count": len(list(run.rglob("browser_replay.json"))),
                }
            )
    return {
        "schema": "full-episode-replay-audit-v1",
        "catalog_cases": len(catalog),
        "attempted_cases": len(manifest["cases"]),
        "cases_with_episodes": len({row["index"] for row in rows}),
        "in_progress_cases": in_progress,
        "event_counts": dict(Counter(row["task_event"] for row in rows)),
        "failure_details": dict(
            Counter(row["failure_detail"] for row in rows if row["failure_detail"])
        ),
        "replay_integrity": dict(Counter(row["replay"]["status"] for row in rows)),
        "historical_attempts": historical,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("results/command-builder-native-full-20261017"),
    )
    options = parser.parse_args()
    root = options.directory.resolve()
    result = audit(root)
    target = root / "replay-audit.json"
    temporary = target.with_name("." + target.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "catalog_cases",
                    "attempted_cases",
                    "cases_with_episodes",
                    "event_counts",
                    "failure_details",
                    "replay_integrity",
                )
            },
            indent=2,
        )
    )
