"""Describe installed source packages without requiring a Git checkout."""

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path


def package_provenance(module):
    """Record installed bytes and optional checkout information."""
    package = Path(module.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".toml", ".xml"):
            digest.update(path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes())
    record = dict(
        package=module.__name__,
        version=importlib.metadata.version(module.__name__),
        source_sha256=digest.hexdigest(),
        commit=None,
    )
    patch = b""
    checkout = package.parent
    if (checkout / ".git").exists():
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, check=False
        )
        diff = subprocess.run(
            ["git", "-C", str(checkout), "diff", "HEAD", "--"], capture_output=True, check=False
        )
        if revision.returncode == 0 and diff.returncode == 0:
            record["commit"] = revision.stdout.decode().strip()
            patch = diff.stdout
        else:
            record["git_metadata_available"] = False
    record["working_tree_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    return record, patch
