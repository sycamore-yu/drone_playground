"""Create reproducible source snapshots for experiment artifacts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

MANIFEST = "SOURCE_MANIFEST.json"


def _path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not name or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"invalid source manifest path: {name!r}")
    return path


def _identity(kind: str, mode: int, data: bytes) -> dict:
    return {
        "kind": kind,
        "mode": mode,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_archive(output: Path, files: dict, timestamp: int = 0) -> str:
    """Write normalized metadata; gzip headers do not depend on the output name."""
    with (
        output.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=timestamp) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, (identity, data) in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.mode = identity["mode"]
            info.mtime = timestamp
            if identity["kind"] == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = os.fsdecode(data)
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def export_source(root: Path, output: Path, revision: str = "HEAD") -> dict:
    """Export only committed files, with their origin and content identities."""
    root, output = Path(root).resolve(), Path(output)
    commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "--verify",
            "--end-of-options",
            revision + "^{commit}",
        ],
        cwd=root,
        text=True,
    ).strip()
    timestamp = int(
        subprocess.check_output(
            ["git", "show", "-s", "--format=%ct", commit],
            cwd=root,
            text=True,
        )
    )
    source = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=root)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        for item in archive:
            if item.isdir():
                continue
            _path(item.name)
            if item.name == MANIFEST:
                raise ValueError(f"{MANIFEST} is reserved for generated export metadata")
            if item.issym():
                kind, data, mode = "symlink", os.fsencode(item.linkname), 0o777
            elif item.isfile():
                kind, data = "file", archive.extractfile(item).read()
                mode = 0o755 if item.mode & 0o111 else 0o644
            else:
                raise ValueError(f"unsupported source entry: {item.name}")
            files[item.name] = (_identity(kind, mode, data), data)
    manifest = {
        "schema_version": 1,
        "source_revision": commit,
        "files": {name: identity for name, (identity, _) in sorted(files.items())},
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    files[MANIFEST] = (_identity("file", 0o644, data), data)
    digest = _write_archive(output, files, timestamp)
    return {
        "source_revision": commit,
        "archive_sha256": digest,
        "files": len(manifest["files"]),
    }


def _read_source(root: Path, name: str) -> tuple[dict, bytes] | None:
    relative = _path(name)
    path = root / relative
    # A changed directory symlink must never pull files from outside the source tree.
    if any((root / parent).is_symlink() for parent in relative.parents):
        return None
    if path.is_symlink():
        data = os.fsencode(os.readlink(path))
        return _identity("symlink", 0o777, data), data
    if not path.is_file():
        return None
    data = path.read_bytes()
    mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
    return _identity("file", mode, data), data


def capture_source_archive(root: Path, destination: Path) -> dict | None:
    """Save the actual source, including edits, when an export manifest is present.

    This is a content identity, not authentication of a downloaded distribution.
    Git supplies ignore semantics through an isolated temporary index/repository.
    """
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        return None
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    if manifest.get("schema_version") != 1 or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", manifest.get("source_revision", "")
    ):
        raise ValueError("invalid source manifest version or revision")
    baseline = manifest["files"]
    for name, identity in baseline.items():
        _path(name)
        if (
            name == MANIFEST
            or identity.get("kind") not in {"file", "symlink"}
            or identity.get("mode") not in {0o644, 0o755, 0o777}
            or not re.fullmatch(r"[0-9a-f]{64}", identity.get("sha256", ""))
        ):
            raise ValueError(f"invalid source manifest entry: {name!r}")
    with tempfile.TemporaryDirectory(prefix="drone-source-index-") as temporary:
        subprocess.run(
            ["git", "init", "--bare", "-q", temporary],
            check=True,
            capture_output=True,
        )
        names = subprocess.check_output(
            [
                "git",
                "--git-dir=" + temporary,
                "--work-tree=" + str(root),
                "-c",
                "core.excludesFile=/dev/null",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
        )
    candidates = {os.fsdecode(name) for name in names.split(b"\0") if name}
    # Runs themselves are never source, even if a user removes that ignore rule.
    candidates = {name for name in candidates if not name.startswith("results/")}
    candidates.update(baseline)
    candidates.discard(MANIFEST)
    files = {}
    for name in sorted(candidates):
        value = _read_source(root, name)
        if value is not None:
            files[name] = value
    modified = sorted(
        name for name in baseline if name in files and files[name][0] != baseline[name]
    )
    removed = sorted(baseline.keys() - files.keys())
    added = sorted(files.keys() - baseline.keys())
    files[MANIFEST] = (_identity("file", 0o644, original), original)
    snapshot = destination / "source-snapshot.tar.gz"
    digest = _write_archive(snapshot, files)
    return {
        "source_revision": manifest["source_revision"],
        "manifest_sha256": hashlib.sha256(original).hexdigest(),
        "dirty": bool(modified or removed or added),
        "modified": modified,
        "removed": removed,
        "added": added,
        "snapshot_path": snapshot.name,
        "snapshot_sha256": digest,
    }
