"""Copy an inactive legacy run into the current layout without altering its evidence."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


def fingerprints(root):
    """Hash every original file, retaining symbolic links as links."""
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            result[str(path.relative_to(root))] = "symlink:" + os.readlink(path)
        elif path.is_file():
            with path.open("rb") as stream:
                result[str(path.relative_to(root))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return result


def write_json(path, data):
    """Write finite JSON after the destination has been allocated."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def flatten_evaluation(directory):
    """Merge scene tables and arrays, preserving all original report metadata."""
    summary_path = directory / "report.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    reports = summary.get("reports", {})
    rows, arrays, decisions, sources = [], {}, [], {}
    for scene, report in reports.items():
        folder = directory / scene
        if not folder.is_dir():
            continue
        for source in folder.rglob("*.json"):
            sources[str(source.relative_to(directory))] = json.loads(source.read_text())
        tables = (
            [folder / "episodes.csv"]
            if (folder / "episodes.csv").exists()
            else sorted(folder.glob("episode-*/episodes.csv"))
        )
        replays = []
        for table in tables:
            case = table.parent
            index = (
                int(case.name.removeprefix("episode-")) if case.name.startswith("episode-") else 0
            )
            with table.open(newline="") as stream:
                case_rows = list(csv.DictReader(stream))
            for row_index, row in enumerate(case_rows):
                row.setdefault("scene", scene)
                row.setdefault("episode", str(index if case != folder else row_index))
            rows.extend(case_rows)
            table.unlink()
            trace = case / "trajectories.npz"
            if trace.exists():
                with np.load(trace, allow_pickle=False) as values:
                    for key in values.files:
                        arrays[f"{scene}__{index:04d}__{key}"] = values[key]
                trace.unlink()
            decision = case / "decisions.json"
            if decision.exists():
                original = json.loads(decision.read_text())
                decisions.extend(
                    {"scene": scene, "episode": index, **x} for x in original.get("decisions", [])
                )
                decision.unlink()
            for replay in sorted((case / "rollouts").glob("episode-*")):
                episode = int(replay.name.removeprefix("episode-")) if case == folder else index
                target = directory / "replays" / f"{scene}-{episode:04d}"
                target.parent.mkdir(exist_ok=True)
                if target.exists():
                    raise FileExistsError(f"Duplicate replay identity: {target}")
                replay.rename(target)
                for unroll in target.glob("*.mj_unroll"):
                    replays.append(str(unroll.relative_to(directory)))
        report["replays"] = replays
        report.pop("runs", None)
        for path in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_file() and path.suffix == ".json":
                path.unlink()  # Original contents are retained in summary.migration_source_reports.
            elif path.is_file() or path.is_symlink():
                target = directory / "evidence" / path.relative_to(directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
            elif path.is_dir():
                path.rmdir()
        folder.rmdir()
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    if arrays:
        np.savez_compressed(directory / "trajectories.npz", **arrays)
    if decisions:
        write_json(directory / "decisions.json", {"decisions": decisions})
    summary["migration_source_reports"] = sources
    write_json(summary_path, summary)


def active_run(source):
    """Reject live CLI processes even when their run header is temporarily stale."""
    for process in Path("/proc").glob("[0-9]*"):
        try:
            args = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any("drone-playground" in a or "drone_playground.cli" in a for a in args):
                continue
            cwd = (process / "cwd").resolve()
            if any(a.startswith("output=") and (cwd / a[7:]).resolve() == source for a in args):
                return True
        except (OSError, RuntimeError):
            continue
    return False


def migrate_run(source, destination, *, apply=False):
    """Create a checked migrated copy; a changing source or existing target is rejected."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("The destination must not be inside the source run")
    if destination.exists():
        raise FileExistsError(destination)
    header = json.loads((source / "run.json").read_text())
    if header.get("status") == "running" or active_run(source):
        raise ValueError(f"Cannot migrate a running experiment: {source}")
    if header.get("layout_version") == 2:
        raise ValueError("This run already uses layout version 2")
    before = fingerprints(source)
    result = {
        "source": str(source),
        "destination": str(destination),
        "files": len(before),
        "applied": apply,
    }
    if not apply:
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".migrate-", dir=destination.parent))
    working = temporary / "run"
    try:
        shutil.copytree(source, working, symlinks=True)
        if fingerprints(working) != before:
            raise RuntimeError("Copied files differ from their source fingerprints")
        for path in working.rglob("*"):
            if path.is_symlink() and (path.is_dir() or path.name in {"run.json", "report.json"}):
                raise ValueError(f"Cannot migrate a directory or report symlink: {path}")
        moves = {}
        for archive in sorted((working / "checkpoints").glob("update-*.policy.zip")):
            match = re.fullmatch(r"update-(\d+)\.policy\.zip", archive.name)
            if match is None:
                continue
            target = archive.parent / f"step-{int(match[1]):06d}" / "policy.zip"
            target.parent.mkdir(exist_ok=True)
            moves[str(archive.relative_to(working))] = str(target.relative_to(working))
            archive.rename(target)
        for evaluation in sorted((working / "checkpoint_eval").glob("update-*")):
            if not evaluation.is_dir() or not evaluation.name.removeprefix("update-").isdigit():
                continue
            target = (
                working / "checkpoints" / f"step-{int(evaluation.name.removeprefix('update-')):06d}"
            )
            target.mkdir(parents=True, exist_ok=True)
            moves[str(evaluation.relative_to(working))] = str(target.relative_to(working))
            for item in evaluation.iterdir():
                if (target / item.name).exists():
                    raise FileExistsError(target / item.name)
                item.rename(target / item.name)
            evaluation.rmdir()
            flatten_evaluation(target)
        for index, evaluation in enumerate(
            sorted(p for p in working.glob("benchmark*") if p.is_dir()), 1
        ):
            target = working / "eval" / f"{index:03d}"
            target.parent.mkdir(exist_ok=True)
            moves[str(evaluation.relative_to(working))] = str(target.relative_to(working))
            evaluation.rename(target)
            flatten_evaluation(target)

        def relocate(value):
            if isinstance(value, str):
                parts = value.split("/")
                relative = value
                if source.name in parts:
                    relative = "/".join(parts[parts.index(source.name) + 1 :])
                for old in sorted(moves, key=len, reverse=True):
                    if relative == old or relative.startswith(old + "/"):
                        return moves[old] + relative[len(old) :]
                return value
            if isinstance(value, list):
                return [relocate(x) for x in value]
            if isinstance(value, dict):
                return {k: relocate(v) for k, v in value.items()}
            return value

        selection = working / "selection.json"
        original_header = dict(header)
        if selection.exists():
            header["selection"] = json.loads(selection.read_text())
            selection.unlink()
        events = working / "events/metrics.jsonl"
        if events.exists():
            events.rename(working / "metrics.jsonl")
        header = relocate(header)
        header["layout_version"] = 2
        write_json(working / "run.json", header)
        for name in ("events", "rollouts", "checkpoint_eval"):
            path = working / name
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        # Archives remain opaque: preserve their bytes and the original embedded provenance.
        for old, checksum in before.items():
            if old.endswith(".zip"):
                new = working / moves.get(old, old)
                with new.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                        raise RuntimeError(f"Archive changed during migration: {old}")
        if fingerprints(source) != before:
            raise RuntimeError("Source changed during migration; original was left untouched")
        write_json(
            working / "migration.json",
            {
                "source": str(source),
                "source_files": before,
                "path_mapping": moves,
                "original_run": original_header,
            },
        )
        working.rename(destination)
    finally:
        shutil.rmtree(temporary)
    return result


def main():
    """Plan by default; --apply explicitly creates the independent migrated copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate_run(args.source, args.destination, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
