"""Set up pinned source dependencies and optional native solver dependencies."""

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACADOS_VERSION = "v0.5.1"


def run(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def verify_checkout(directory, revision, patch, name):
    """Compare against the patched tree using a private disposable Git index."""
    with tempfile.TemporaryDirectory(prefix="drone-source-index-") as temporary:
        environment = {
            **os.environ,
            "GIT_INDEX_FILE": str(Path(temporary) / "index"),
        }

        def check(args):
            return subprocess.run(
                ["git", *args],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            ).stdout

        check(["read-tree", revision])
        if patch:
            check(["apply", "--cached", str(patch)])
        difference = check(["diff", "--binary", "--no-ext-diff", "--"])
        if difference:
            raise ValueError(
                f"Cached modifications differ from the declared patch: {name}"
            )
        untracked = set(
            filter(
                None,
                check(
                    ["ls-files", "--others", "--exclude-standard", "-z"]
                ).split("\0"),
            )
        )
        if untracked - {".drone-playground-source.json"}:
            raise ValueError(f"Unexpected cached source files: {name}")


def fetch(root=ROOT):
    """Materialize pinned editable source dependencies into tmp/sources."""
    root = Path(root).resolve()
    sources = json.loads((root / "third_party/sources.json").read_text())
    results = {}
    for name, specification in sources.items():
        relative = specification.get("checkout")
        if not relative:
            continue
        directory = (root / relative).resolve()
        if not directory.is_relative_to(root / "tmp/sources"):
            raise ValueError(f"Source cache must be under tmp/sources: {name}")
        revision = specification["revision"]
        patch = (
            root / specification["patch"]
            if specification.get("patch")
            else None
        )
        expected_patch = (
            hashlib.sha256(patch.read_bytes()).hexdigest() if patch else None
        )
        marker = directory / ".drone-playground-source.json"
        if directory.exists():
            if run(["git", "rev-parse", "HEAD"], directory) != revision:
                raise ValueError(f"Cached revision differs for {name}")
            if (
                not marker.exists()
                or json.loads(marker.read_text()).get("patch_sha256")
                != expected_patch
            ):
                raise ValueError(
                    f"Existing cache needs explicit reconciliation: {directory}"
                )
        else:
            directory.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    specification["repository"],
                    str(directory),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "checkout", "--detach", revision],
                cwd=directory,
                check=True,
            )
            if patch:
                subprocess.run(
                    ["git", "apply", "--check", str(patch)],
                    cwd=directory,
                    check=True,
                )
                subprocess.run(
                    ["git", "apply", str(patch)],
                    cwd=directory,
                    check=True,
                )
            marker.write_text(
                json.dumps(
                    dict(revision=revision, patch_sha256=expected_patch),
                    indent=2,
                )
            )
        verify_checkout(directory, revision, patch, name)
        results[name] = dict(
            path=relative,
            revision=revision,
            patch_sha256=expected_patch,
        )
    return results


def setup_acados(root=ROOT):
    """Build the pinned optional acados dependency in the project cache."""
    root = Path(root).resolve()
    destination = root / "tmp/p3p4/optimization/acados"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").is_dir():
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                ACADOS_VERSION,
                "--depth",
                "1",
                "--recursive",
                "--shallow-submodules",
                "https://github.com/acados/acados.git",
                str(destination),
            ],
            check=True,
        )
    if (
        run(["git", "describe", "--tags", "--exact-match"], destination)
        != ACADOS_VERSION
    ):
        raise ValueError(f"Expected acados {ACADOS_VERSION}: {destination}")

    build = destination / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(destination),
            "-B",
            str(build),
            f"-DCMAKE_INSTALL_PREFIX={destination}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DACADOS_WITH_QPOASES=ON",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DBLASFEO_TARGET=GENERIC",
            "-DHPIPM_TARGET=GENERIC",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "install",
            "--parallel",
            "4",
        ],
        check=True,
    )

    renderer = destination / "bin/t_renderer"
    marker = destination / "bin/renderer-0.2.0"
    if not marker.is_file():
        renderer.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/acados/tera_renderer/releases/download/"
            "v0.2.0/t_renderer-v0.2.0-linux-amd64",
            renderer,
        )
        renderer.chmod(
            renderer.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        marker.touch()

    pixi_python = root / ".pixi/envs/default/bin/python"
    if not pixi_python.is_file():
        raise FileNotFoundError(
            "Pixi environment is required before building acados; "
            "run pixi install --locked first"
        )
    subprocess.run(
        [
            str(pixi_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(destination.parent / "python-deps"),
            "future-fstrings==1.2.0",
            "matplotlib==3.10.6",
        ],
        check=True,
    )
    return {
        "version": ACADOS_VERSION,
        "source_dir": str(destination),
        "install_dir": str(destination),
        "revision": run(["git", "rev-parse", "HEAD"], destination),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "sources",
        help="materialize pinned editable sources before Pixi installation",
    )
    subparsers.add_parser(
        "acados",
        help="build the optional pinned acados dependency after Pixi installation",
    )
    args = parser.parse_args()
    result = fetch() if args.command == "sources" else setup_acados()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
