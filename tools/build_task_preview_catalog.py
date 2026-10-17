"""Publish an Isaac Lab-style task gallery from actual native MuJoCo recordings.

The source is the frozen Command Builder catalog and independent full-Episode
results. A playable preview is never synthesized, lengthened or attributed to
a policy/checkpoint that was not actually evaluated. Safe to rerun after new
records have been published; writes one atomic JSON file into ignored results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _failures(path: Path) -> dict[tuple[str, int], str]:
    if not path.is_file():
        return {}
    try:
        raw = _json(path)
        decisions = raw.get("decisions", []) if isinstance(raw, dict) else raw
        result = {}
        for item in decisions:
            if item.get("detail"):
                result[(str(item.get("scene", "")), int(item.get("episode", 0)))] = str(
                    item["detail"]
                )
        return result
    except (KeyError, OSError, TypeError, ValueError):
        return {}


def _record_key(path: Path) -> tuple[str, int]:
    try:
        scene, episode = path.parent.name.rsplit("-", 1)
        return scene, int(episode)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Unrecognized replay episode directory: {path.parent.name}") from error


def _url(file: Path) -> str:
    """Produce a same-origin path, never an absolute host filesystem path."""
    file = file.resolve()
    if not file.is_relative_to(REPO.resolve()):
        raise ValueError(f"Preview artifact is outside the repository: {file}")
    return "/" + file.relative_to(REPO.resolve()).as_posix()


def build(source: Path) -> dict:
    """Map each published native command to its own authentic flight examples."""
    source = source.resolve()
    commands = _json(source / "catalog.json")["cases"]
    manifest = _json(source / "manifest.json")
    runs = manifest["cases"]
    entries = []
    outcome_counts = Counter()
    for index, command in enumerate(commands):
        matches = [(key, row) for key, row in runs.items() if key.startswith(f"{index:03d}_")]
        if len(matches) > 1:
            raise ValueError(f"Multiple live manifest entries for case {index}")
        record = {
            "id": index,
            **command,
            "status": "not_recorded",
            "recordings": [],
        }
        if matches:
            _, current = matches[0]
            record["status"] = current["status"]
            record["source_revision"] = current.get("source_revision")
            record["run_relative"] = current.get("run_relative")
            directory = source / current["run_relative"]
            episodes = {
                (row["scene"], int(row["episode"])): row
                for row in _rows(directory / "episodes.csv")
            }
            failures = _failures(directory / "decisions.json")
            for relative in current.get("browser_replays", []):
                browser = directory / relative
                key = _record_key(browser)
                episode = episodes.get(key)
                if episode is None or not browser.is_file():
                    continue
                native = browser.with_name("episode.mj_unroll")
                if not native.is_file():
                    continue
                elapsed = float(episode["flight_seconds"])
                if elapsed < 0:
                    raise ValueError("Invalid negative flight duration")
                event = episode["event"]
                reason = failures.get(key, "") if event == "METHOD_FAILURE" else ""
                # A native `NO_SOLUTION` is an honest ending at its recorded
                # timestamp, not collision and not missing WASM video frames.
                record["recordings"].append(
                    {
                        "scene": key[0],
                        "episode": key[1],
                        "event": event,
                        "elapsed_seconds": elapsed,
                        "task_limit_seconds": float(current.get("task_duration_seconds", 0)),
                        "failure_detail": reason,
                        "browser_record": _url(browser),
                        "rscope_record": _url(native),
                        "report": _url(directory / "report.json")
                        if (directory / "report.json").is_file()
                        else None,
                    }
                )
                outcome_counts[event] += 1
        entries.append(record)
    return {
        "format": "drone-playground-task-preview-v1",
        "source": {
            "command_catalog": _url(source / "catalog.json"),
            "execution_manifest": _url(source / "manifest.json"),
        },
        "summary": {
            "native_methods": len(entries),
            "recorded_methods": sum(bool(x["recordings"]) for x in entries),
            "recorded_episodes": sum(len(x["recordings"]) for x in entries),
            "terminal_events": dict(outcome_counts),
        },
        "cases": entries,
    }


def publish(source: Path, output: Path | None = None) -> Path:
    """Write one complete catalog to the static HTTP server's results tree."""
    document = build(source)
    output = (output or source / "preview-catalog.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("results/command-builder-native-full-20261017"),
    )
    options = parser.parse_args()
    path = publish(options.directory)
    print(f"Published {path}")
    print(json.dumps(_json(path)["summary"], indent=2))
