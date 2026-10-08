"""Publish completed RScope bundles without changing their saved source."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from drone_playground.visualization._rscope_files import _atomic_write_bytes, _exclusive_rscope

_PUBLISH_MARKER = ".drone_playground_publish.json"


def _publication_files(directory: Path) -> list[Path]:
    meta = directory / "rscope_meta.pkl"
    unrolls = sorted(directory.glob("*.mj_unroll"))
    if not meta.is_file():
        raise FileNotFoundError(f"missing rscope metadata: {meta}")
    if not unrolls:
        raise FileNotFoundError(f"no .mj_unroll files found in {directory}")
    files = [meta, *unrolls]
    xml = directory / "scene.xml"
    if xml.is_file():
        files.append(xml)
    for name in ("components.json", "replay-visualization.json"):
        identity = directory / name
        if identity.is_file():
            files.append(identity)
    assets = directory / "assets"
    if assets.is_dir():
        files.extend(sorted(path for path in assets.rglob("*") if path.is_file()))
    return files


def _validate_active_directory(active_dir: Path) -> None:
    existing = {
        path.relative_to(active_dir).as_posix() for path in active_dir.rglob("*") if path.is_file()
    }
    if not existing:
        return
    marker_path = active_dir / _PUBLISH_MARKER
    if not marker_path.is_file():
        raise RuntimeError(
            "active rscope directory contains unknown files and has no publish marker: "
            f"{active_dir}"
        )
    marker = json.loads(marker_path.read_text())
    managed = set(marker.get("files", [])) | {_PUBLISH_MARKER}
    unknown = sorted(existing - managed)
    if unknown:
        raise RuntimeError(f"active rscope directory contains unknown files: {', '.join(unknown)}")


def publish_run(
    directory: Path,
    active_dir: Path = Path("/tmp/rscope/active_run"),
) -> Path:
    """Publish a selected run into rscope's active directory without mutating the source run."""
    directory = Path(directory).resolve()
    active_dir = Path(active_dir).resolve()
    if directory == active_dir:
        raise ValueError("source rollout directory and active directory must differ")
    files = _publication_files(directory)

    with _exclusive_rscope():
        active_dir.parent.mkdir(parents=True, exist_ok=True)
        if active_dir.exists():
            _validate_active_directory(active_dir)
        stage = Path(tempfile.mkdtemp(prefix=f".{active_dir.name}.publish-", dir=active_dir.parent))
        backup = active_dir.parent / f".{active_dir.name}.backup-{uuid.uuid4().hex}"
        try:
            published: list[str] = []
            for source in files:
                relative = source.relative_to(directory)
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                published.append(relative.as_posix())
            marker = {
                "source": str(directory),
                "files": sorted(published),
            }
            (stage / _PUBLISH_MARKER).write_text(
                json.dumps(marker, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            )

            had_active = active_dir.exists()
            if had_active:
                os.replace(active_dir, backup)
            try:
                os.replace(stage, active_dir)
            except BaseException:
                if had_active and backup.exists() and not active_dir.exists():
                    os.replace(backup, active_dir)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return active_dir
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            if backup.exists() and active_dir.exists():
                shutil.rmtree(backup, ignore_errors=True)


def append_rollout(directory: Path, active_dir: Path = Path("/tmp/rscope/active_run")) -> Path:
    """Append complete snapshots without replacing the directory watched by rscope.

    Native rscope 0.0.8 handles created events, not moved events. A hard link to
    a fully written temporary file emits creation only after the payload exists.
    """
    directory, active_dir = (
        Path(directory).resolve(),
        Path(active_dir).resolve(),
    )
    files = _publication_files(directory)
    with _exclusive_rscope():
        _validate_active_directory(active_dir)
        if (directory / "scene.xml").read_bytes() != (active_dir / "scene.xml").read_bytes():
            raise ValueError("Live snapshots must use the same model; select a new run explicitly")
        for source in files:
            if "assets" in source.relative_to(directory).parts:
                target = active_dir / source.relative_to(directory)
                if not target.exists() or source.read_bytes() != target.read_bytes():
                    raise ValueError("Live snapshots must use identical model assets")
        marker_path = active_dir / _PUBLISH_MARKER
        marker = json.loads(marker_path.read_text())
        names = set(marker["files"])
        for source in files:
            if source.suffix != ".mj_unroll":
                continue
            target = active_dir / source.name
            if target.exists():
                target = active_dir / f"{source.stem}-{uuid.uuid4().hex[:8]}.mj_unroll"
            fd, temporary_name = tempfile.mkstemp(prefix=".complete-", dir=active_dir)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                shutil.copy2(source, temporary)
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            names.add(target.name)
        marker.update(files=sorted(names), latest_source=str(directory))
        _atomic_write_bytes(marker_path, (json.dumps(marker, indent=2) + "\n").encode())
    return active_dir


def publish_snapshot(
    directory: Path,
    run_directory: Path,
    *,
    first: bool = False,
    active_dir: Path = Path("/tmp/rscope/active_run"),
) -> dict:
    """Publish optional live output without changing the outcome of saved training.

    Selecting another run is an ordinary viewer operation. Its directory remains
    selected until an explicit first publication chooses a new run. Publication
    failures are returned for logging; checkpoint and evaluation errors stay fatal
    at their own, mandatory boundaries.
    """
    directory, run_directory, active_dir = (
        Path(directory).resolve(),
        Path(run_directory).resolve(),
        Path(active_dir).resolve(),
    )
    try:
        if not directory.is_relative_to(run_directory):
            raise ValueError("Snapshot directory must belong to its run")
        if not first:
            marker_path = active_dir / _PUBLISH_MARKER
            if not marker_path.is_file():
                return {
                    "status": "not-selected",
                    "reason": "no active owned selection",
                }
            marker = json.loads(marker_path.read_text())
            selected = Path(marker["source"]).resolve()
            if not selected.is_relative_to(run_directory):
                return {
                    "status": "not-selected",
                    "selected_source": str(selected),
                }
        (publish_run if first else append_rollout)(directory, active_dir)
        return {
            "status": "published",
            "directory": str(directory),
            "active_dir": str(active_dir),
        }
    except Exception as error:
        return {
            "status": "error",
            "error": repr(error),
            "directory": str(directory),
        }
