"""Materialize pinned source dependencies into the project's ignored cache.

Run before ``pixi install``. Existing cache checkouts are verified in place; this
tool never resets a dirty tree or switches an unrelated checkout.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def verify_checkout(directory, revision, patch, name):
    """Compare against the patched tree using a private, disposable Git index.

    Plain ``git diff HEAD`` omits newly added patch files. An alternate index
    includes those files while leaving the user's checkout and staging intact.
    """
    with tempfile.TemporaryDirectory(prefix="drone-source-index-") as temporary:
        environment = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}

        def check(args):
            return subprocess.run(["git", *args], cwd=directory, env=environment,
                                  capture_output=True, text=True, check=True).stdout

        check(["read-tree", revision])
        if patch:
            check(["apply", "--cached", str(patch)])
        difference = check(["diff", "--binary", "--no-ext-diff", "--"])
        if difference:
            raise ValueError(f"Cached modifications differ from the declared patch: {name}")
        untracked = set(filter(None, check(["ls-files", "--others", "--exclude-standard", "-z"]).split("\0")))
        if untracked - {".drone-playground-source.json"}:
            raise ValueError(f"Unexpected cached source files: {name}")


def fetch(root=ROOT):
    root = Path(root).resolve()
    # JSON is a YAML subset; the single manifest is readable before dependencies
    # have been installed, using only the Python standard library.
    sources = json.loads((root / "third_party/sources.yaml").read_text())
    results = {}
    for name, specification in sources.items():
        relative = specification.get("checkout")
        if not relative:
            continue
        directory = (root / relative).resolve()
        if not directory.is_relative_to(root / "tmp/sources"):
            raise ValueError(f"Source cache must be under tmp/sources: {name}")
        revision = specification["revision"]
        patch = root / specification["patch"] if specification.get("patch") else None
        expected_patch = hashlib.sha256(patch.read_bytes()).hexdigest() if patch else None
        marker = directory / ".drone-playground-source.json"
        if directory.exists():
            if run(["git", "rev-parse", "HEAD"], directory) != revision:
                raise ValueError(f"Cached revision differs for {name}")
            if (
                not marker.exists()
                or json.loads(marker.read_text()).get("patch_sha256") != expected_patch
            ):
                raise ValueError(f"Existing cache needs explicit reconciliation: {directory}")
        else:
            directory.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--no-checkout", specification["repository"], str(directory)],
                check=True,
            )
            subprocess.run(["git", "checkout", "--detach", revision], cwd=directory, check=True)
            if patch:
                subprocess.run(["git", "apply", "--check", str(patch)], cwd=directory, check=True)
                subprocess.run(["git", "apply", str(patch)], cwd=directory, check=True)
            marker.write_text(
                json.dumps(dict(revision=revision, patch_sha256=expected_patch), indent=2)
            )
        verify_checkout(directory, revision, patch, name)
        results[name] = dict(path=relative, revision=revision, patch_sha256=expected_patch)
    return results


if __name__ == "__main__":
    try:
        print(json.dumps(fetch(), indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
