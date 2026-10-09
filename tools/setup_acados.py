"""Build the optional pinned acados dependency without changing the core environment."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Install acados v0.5.1 or use an existing matching native build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/acados"))
    parser.add_argument("--python-path", type=Path, default=Path(".cache/acados-python"))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source = args.source.resolve()
    if not source.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "v0.5.1",
                "--depth",
                "1",
                "--recursive",
                "https://github.com/acados/acados.git",
                str(source),
            ],
            check=True,
        )
    version = subprocess.check_output(
        ["git", "-C", str(source), "describe", "--tags", "--exact-match"], text=True
    ).strip()
    if version != "v0.5.1":
        raise ValueError(f"Expected acados v0.5.1, found {version}")
    if not (source / "lib/libacados.so").is_file():
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(source / "build"),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBLASFEO_TARGET=GENERIC",
                f"-DCMAKE_INSTALL_PREFIX={source}",
            ],
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(source / "build"),
                "--target",
                "install",
                "--parallel",
                str(args.jobs),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            str(args.python_path),
            str(source / "interfaces/acados_template"),
        ],
        check=True,
    )
    print(f"export ACADOS_SOURCE_DIR={source}")
    print(f"export LD_LIBRARY_PATH={source}/lib:${{LD_LIBRARY_PATH:-}}")
    print("Optional interface installed. Select controller=attitude_mpc with these variables.")


if __name__ == "__main__":
    main()
