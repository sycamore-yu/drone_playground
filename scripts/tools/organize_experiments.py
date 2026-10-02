"""Migrate run storage and maintain lightweight selections over immutable runs."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from drone_playground.artifacts.layout import (  # noqa: E402
    experiment_scope,
    find_experiment,
    resolve_artifact,
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_date(path):
    value = read(path / "manifest.json")["started_at"]
    return datetime.fromisoformat(value).astimezone(timezone.utc).strftime("%y%m%d")


def _relative(root, path):
    """Prefer the current repository path even when it is a symlink to frozen evidence."""
    root = Path(root).resolve()
    path = Path(path)
    local = path if path.is_absolute() else root / path
    try:
        return str(local.relative_to(root))
    except ValueError:
        return str(local.resolve())


def _run_config(run):
    config = run / "resolved-config.json"
    if config.is_file():
        return read(config)
    manifest = run / "manifest.json"
    if manifest.is_file():
        return read(manifest).get("config", {})
    return {}


def _checkpoint_reference(root, checkpoint, report):
    checkpoint = resolve_artifact(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = Path(root) / checkpoint
    metadata_path = checkpoint.with_suffix(".json")
    metadata = read(metadata_path)
    if digest(checkpoint) != metadata["sha256"]:
        raise ValueError(f"Checkpoint digest mismatch: {checkpoint}")
    if (
        report.get("parameter_sha256")
        and metadata.get("parameter_sha256") != report["parameter_sha256"]
    ):
        raise ValueError(f"Selected weights do not match evaluation: {checkpoint}")
    return {
        "path": _relative(root, checkpoint),
        "sha256": digest(checkpoint),
        "metadata": _relative(root, metadata_path),
        "parameter_sha256": metadata.get("parameter_sha256"),
    }


def build_selection(root, progress, goal, pointcloud_run=None):
    """Write one lightweight selection manifest; never copy run artifacts."""
    root = Path(root)
    directory = root / "results/selected"
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for number, cell in enumerate(progress["cells"], 1):
        name = f"{number:02d}-{cell['method']}-{cell['task']}"
        records = cell.get("training_seeds", [])
        if not records:
            evidence = cell.get("evidence")
            run_id = cell.get("run_id") or (
                evidence.get("run_id") if isinstance(evidence, dict) else None
            )
            run_id = run_id or cell.get("development", {}).get("run_id")
            if cell["method"] == "pointcloud" and pointcloud_run:
                run_id = pointcloud_run
            records = [dict(run_id=run_id)] if run_id else []

        chosen = []
        for record in records:
            run = find_experiment(root, record["run_id"])
            if run is None:
                raise FileNotFoundError(record["run_id"])
            config = read(run / "resolved-config.json")
            checkpoint_eval_only = (
                cell["method"] == "pointcloud"
                and config.get("mode") == "train"
            )
            if checkpoint_eval_only:
                result = read(run / "result.json")
                report_source = (
                    run / "eval" / f"update-{result['selected']['updates']:07d}.json"
                )
            else:
                report_source = run / "eval/report.json"
            report = read(report_source)

            seed = record.get("training_seed", record.get("seed"))
            if seed is None and config.get("method", {}).get("trainable"):
                seed = config.get("training", {}).get("seed")
            manifest = read(run / "manifest.json")
            checkpoint = report.get("checkpoint")
            chosen.append(
                {
                    "run_id": run.name,
                    "source_run": _relative(root, run),
                    "training_seed": seed,
                    "started_date": run_date(run),
                    "source_revision": manifest.get("code", {}).get("commit"),
                    "checkpoint_eval_only": checkpoint_eval_only,
                    "report": _relative(root, report_source),
                    "parameter_sha256": report.get("parameter_sha256"),
                    "checkpoint": (
                        _checkpoint_reference(root, checkpoint, report)
                        if checkpoint
                        else None
                    ),
                    "quality_passed": cell["passed"],
                    "status": cell["status"],
                }
            )

        index.append(
            {
                "cell": name,
                "method": cell["method"],
                "task": cell["task"],
                "status": cell["status"],
                "quality_passed": cell["passed"],
                "release_requirement_satisfied": bool(
                    cell["passed"] or cell.get("release_requirement_satisfied")
                ),
                "criterion": cell.get("criterion"),
                "runs": chosen,
            }
        )

    document = {
        "goal": goal,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            key: progress[key]
            for key in (
                "passed_cells",
                "user_accepted_cells",
                "release_requirement_satisfied_cells",
                "required_cells",
                "remaining_quality_cells",
            )
            if key in progress
        },
        "cells": index,
    }
    write(directory / f"{goal}.json", document)

    lines = [f"# {goal}", "", "| Method | Task | Status | Runs |", "|---|---|---|---:|"]
    for row in index:
        lines.append(
            f"| {row['method']} | {row['task']} | {row['status']} | {len(row['runs'])} |"
        )
    lines.extend(
        [
            "",
            "This file is a selection view over immutable runs. "
            "Reports, checkpoints and replays remain in their source run directories.",
            "",
        ]
    )
    (directory / f"{goal}.md").write_text("\n".join(lines))
    return index


def _move_symlink(source, destination):
    target = source.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(
        os.path.relpath(target, destination.parent),
        target_is_directory=True,
    )
    source.unlink()


def _move(source, destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        _move_symlink(source, destination)
    else:
        source.rename(destination)


def migration_plan(root):
    """Return deterministic old->new moves without mutating the filesystem."""
    root = Path(root)
    experiments = root / "results"
    moves = []

    legacy_runs = experiments / "tmp"
    if legacy_runs.is_dir():
        for date in sorted(legacy_runs.iterdir()):
            if not date.is_dir() or date.is_symlink():
                continue
            for run in sorted(date.iterdir()):
                if not (run.is_dir() or run.is_symlink()):
                    continue
                config = _run_config(run)
                if config:
                    task, method = experiment_scope(config)
                    destination = experiments / "runs" / task / method / run.name
                else:
                    destination = (
                        experiments / "scratch" / "legacy-runs" / date.name / run.name
                    )
                moves.append((run, destination))

    known = {
        "mid360-paper-replay": experiments / "scratch/replays/mid360-paper-replay",
        "mid360-pointcloud-preview": experiments / "scratch/previews/mid360-pointcloud-preview",
        "mid360-pointcloud-preview-test": experiments / "scratch/previews/mid360-pointcloud-preview-test",
    }
    for name, destination in known.items():
        source = experiments / name
        if source.exists() or source.is_symlink():
            moves.append((source, destination))
    return moves


def migrate_layout(root):
    """Move legacy results into runs/selected/scratch without deleting evidence."""
    root = Path(root)
    experiments = root / "results"
    (experiments / "runs").mkdir(parents=True, exist_ok=True)
    (experiments / "selected").mkdir(parents=True, exist_ok=True)
    (experiments / "scratch").mkdir(parents=True, exist_ok=True)

    plan = migration_plan(root)
    seen = {}
    for source, destination in plan:
        if destination in seen and seen[destination] != source:
            raise ValueError(f"Migration collision: {destination}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        seen[destination] = source

    for source, destination in plan:
        _move(source, destination)

    legacy_tmp = experiments / "tmp"
    if legacy_tmp.is_dir():
        migration = experiments / "scratch/migration"
        migration.mkdir(parents=True, exist_ok=True)
        for item in list(legacy_tmp.iterdir()):
            if item.is_dir() and not item.is_symlink():
                try:
                    item.rmdir()
                    continue
                except OSError:
                    pass
            destination = migration / item.name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            _move(item, destination)
        try:
            legacy_tmp.rmdir()
        except OSError:
            pass

    legacy_selected = experiments / "main_result"
    if legacy_selected.exists() or legacy_selected.is_symlink():
        destination = experiments / "scratch/legacy/main_result"
        _move(legacy_selected, destination)

    return [
        {"source": str(source.relative_to(root)), "destination": str(destination.relative_to(root))}
        for source, destination in plan
    ]


def import_legacy_selection(root, goal):
    """Compact the preserved historical selection into references only."""
    root = Path(root)
    legacy_root = root / "results/scratch/legacy/main_result" / goal
    legacy_index = legacy_root / "index.json"
    if not legacy_index.is_file():
        raise FileNotFoundError(legacy_index)
    original = read(legacy_index)
    cells = []
    for cell in original["cells"]:
        packages = {}
        cell_dir = legacy_root / cell["cell"]
        if cell_dir.is_dir():
            for selection in cell_dir.glob("*/selection.json"):
                try:
                    record = read(selection)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if record.get("run_id"):
                    packages[record["run_id"]] = selection.parent
        runs = []
        for record in cell.get("runs", []):
            run_id = record["run_id"]
            run = find_experiment(root, run_id)
            package = packages.get(run_id)
            if run is None and package is not None and (package / "original-run").is_dir():
                run = package / "original-run"
            report = None
            if run is not None and (run / "eval/report.json").is_file():
                report = run / "eval/report.json"
            elif package is not None and (package / "report.json").is_file():
                report = package / "report.json"
            checkpoint = None
            if package is not None:
                weights = sorted((package / "weights").glob("*.pkl"))
                if len(weights) == 1:
                    checkpoint = {
                        "path": _relative(root, weights[0]),
                        "sha256": digest(weights[0]),
                    }
            runs.append(
                {
                    "run_id": run_id,
                    "source_run": _relative(root, run) if run is not None else None,
                    "training_seed": record.get("training_seed"),
                    "source_revision": record.get("source_revision"),
                    "parameter_sha256": record.get("parameter_sha256"),
                    "report": _relative(root, report) if report is not None else None,
                    "checkpoint": checkpoint,
                    "legacy_package": (
                        _relative(root, package) if package is not None else None
                    ),
                    "quality_passed": record.get(
                        "quality_passed", cell.get("quality_passed")
                    ),
                    "status": record.get("status", cell.get("status")),
                }
            )
        cells.append(
            {
                key: cell.get(key)
                for key in (
                    "cell",
                    "method",
                    "task",
                    "status",
                    "quality_passed",
                    "release_requirement_satisfied",
                )
            }
            | {"runs": runs}
        )
    document = {
        "goal": goal,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "origin": "legacy-main-result-import",
        "legacy_index": _relative(root, legacy_index),
        "legacy_index_sha256": digest(legacy_index),
        "cells": cells,
    }
    directory = root / "results/selected"
    write(directory / f"{goal}.json", document)
    lines = [
        f"# {goal}",
        "",
        "| Method | Task | Status | Runs |",
        "|---|---|---|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['method']} | {row['task']} | {row['status']} | {len(row['runs'])} |"
        )
    lines += [
        "",
        "Selection only: raw runs stay under results/runs; "
        "the preserved pre-migration evidence package stays under results/scratch/legacy.",
        "",
    ]
    (directory / f"{goal}.md").write_text("\n".join(lines))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--goal", default="v1-18-cells")
    parser.add_argument("--pointcloud-run")
    parser.add_argument("--migrate-layout", action="store_true")
    parser.add_argument(
        "--import-legacy", action="store_true",
        help="Refresh the preserved historical selection rather than current progress.",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    if args.migrate_layout:
        plan = migration_plan(root)
        if not args.apply:
            print(
                json.dumps(
                    {
                        "moves": [
                            {
                                "source": str(source.relative_to(root)),
                                "destination": str(destination.relative_to(root)),
                            }
                            for source, destination in plan
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        moved = migrate_layout(root)
        print(json.dumps({"moved": len(moved)}, ensure_ascii=False))
        return

    progress_path = root / "artifacts/verification/release-progress.json"
    legacy_index = (
        root
        / "results/scratch/legacy/main_result"
        / args.goal
        / "index.json"
    )
    if progress_path.is_file() and not args.import_legacy:
        progress = read(progress_path)
        source = str(progress_path.relative_to(root))
        if not args.apply:
            print(
                json.dumps(
                    {
                        "goal": args.goal,
                        "cells": len(progress["cells"]),
                        "pointcloud_run": args.pointcloud_run,
                        "source": source,
                        "output": f"results/selected/{args.goal}.json",
                    },
                    ensure_ascii=False,
                )
            )
            return
        cells = build_selection(root, progress, args.goal, args.pointcloud_run)
    else:
        if not legacy_index.is_file():
            raise FileNotFoundError(
                "No current release-progress.json or preserved legacy selection index"
            )
        if not args.apply:
            print(
                json.dumps(
                    {
                        "goal": args.goal,
                        "source": str(legacy_index.relative_to(root)),
                        "output": f"results/selected/{args.goal}.json",
                    },
                    ensure_ascii=False,
                )
            )
            return
        cells = import_legacy_selection(root, args.goal)
    print(
        json.dumps(
            {
                "cells": len(cells),
                "selected_run_records": sum(len(row["runs"]) for row in cells),
                "selection": f"results/selected/{args.goal}.json",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
