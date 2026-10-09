"""Reproducible run identities, event logs and per-episode evaluation records."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf


def atomic_json(path: Path, data: dict) -> None:
    """Replace a JSON record after a complete, finite serialization."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_identity() -> dict:
    """Fingerprint installed source independently of availability of a Git checkout."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml", ".xml", ".toml"}:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    result = {"source_sha256": digest.hexdigest()}
    checkout = root.parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True
    )
    result["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    return result


def gpu_usage() -> dict:
    """Read current GPU memory/utilization; no fabricated measurement when unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        memory, utilization = map(float, result.stdout.splitlines()[0].split(","))
        return {"gpu_memory_mib": memory, "gpu_utilization_percent": utilization}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"gpu_memory_mib": None, "gpu_utilization_percent": None}


class RunRecord:
    """Own one run's resolved recipe, identity, append-only metrics and status."""

    def __init__(self, directory: str | Path, config, *, resume: bool = False):
        """Create or resume the run identity, resolved recipe and append-only events."""
        self.directory = Path(directory)
        if (self.directory / "run.json").exists() and not resume:
            raise FileExistsError(f"Run already exists: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ("events", "checkpoints", "rollouts"):
            (self.directory / name).mkdir(exist_ok=True)
        resolved = OmegaConf.to_container(config, resolve=True)
        config_path = self.directory / "config.yaml"
        if not resume:
            OmegaConf.save(OmegaConf.create(resolved), config_path)
        self.start = time.monotonic()
        versions = {}
        for package in ("drone-playground", "crazyflow", "jax", "flax", "mujoco", "rscope"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "source"
        self.identity = {
            "schema_version": 1,
            "status": "running",
            "started_unix": time.time(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "versions": versions,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "devices": [str(device) for device in jax.devices()],
            **source_identity(),
        }
        session_identity = dict(self.identity)
        if resume and (self.directory / "run.json").exists():
            self.identity = json.loads((self.directory / "run.json").read_text())
            self.identity.update(status="running", resumed_unix=time.time())
        atomic_json(self.directory / "run.json", self.identity)
        self.event("session_started", resumed=resume, identity=session_identity)

    def event(self, event: str, **metrics) -> None:
        """Append one finite metric record, preserving earlier failed attempts."""
        data = {"event": event, "wall_seconds": time.monotonic() - self.start, **metrics}
        with (self.directory / "events" / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False) + "\n")

    def finish(self, status: str, **metrics) -> None:
        """Record measured outcome; only the caller's acceptance checker may declare a pass."""
        self.identity.update(status=status, wall_seconds=time.monotonic() - self.start, **metrics)
        atomic_json(self.directory / "run.json", self.identity)


def write_episodes(path: Path, episodes: list[dict]) -> None:
    """Write all outcomes, including failures, with an explicit stable denominator."""
    if not episodes:
        raise ValueError("Cannot report an empty evaluation as an experiment")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)


def acceptance(task: str, episodes: list[dict]) -> dict:
    """Apply C2/C3/C4 exactly; incomplete episode sets cannot pass."""
    successes = [episode for episode in episodes if episode["event"] == "SUCCESS"]
    success_rate = len(successes) / len(episodes) if episodes else 0.0
    rmse = (
        float(np.mean([episode["position_rmse"] for episode in successes]))
        if successes and task == "tracking"
        else None
    )
    if task == "tracking":
        passed = len(episodes) == 100 and len(successes) >= 95 and rmse is not None and rmse <= 0.25
    elif task == "racing":
        passed = len(episodes) == 100 and len(successes) >= 90
    elif task == "navigation":
        passed = len(episodes) == 25 and len(successes) >= 23
    else:
        raise ValueError(f"Unknown task: {task}")
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": success_rate,
        "successful_position_rmse": rmse,
        "passed": bool(passed),
    }
