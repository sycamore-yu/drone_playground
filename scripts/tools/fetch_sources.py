"""Materialize pinned source dependencies into the project's ignored cache.

Run before ``pixi install``. Existing cache checkouts are verified in place; this
tool never resets a dirty tree or switches an unrelated checkout.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


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
        diff = run(["git", "diff", "--binary", "HEAD"], directory)
        if patch and diff.strip() != patch.read_text().strip():
            raise ValueError(f"Cached modifications differ from the declared patch: {name}")
        if not patch and diff:
            raise ValueError(f"Unexpected cached source modification: {name}")
        results[name] = dict(path=relative, revision=revision, patch_sha256=expected_patch)
    return results


if __name__ == "__main__":
    try:
        print(json.dumps(fetch(), indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
