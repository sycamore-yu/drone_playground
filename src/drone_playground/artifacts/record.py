"""Durable evidence recording for Drone Playground runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

from tensorboardX import SummaryWriter

from drone_playground.artifacts.layout import experiment_directory
from drone_playground.artifacts.source_snapshot import capture_source_archive

_HEARTBEAT_INTERVAL_SECONDS = 60.0


def _utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically and durably enough for run-state recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_snapshot(root: Path) -> dict[str, Any]:
    """Capture the current commit, full worktree patch, and status before run files exist."""
    top = _run_command(["git", "rev-parse", "--show-toplevel"], root)
    if top is None or top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
        return {"commit": None, "dirty": None, "status": "", "patch": ""}
    head = _run_command(["git", "rev-parse", "HEAD"], root)
    if head is None or head.returncode != 0:
        return {"commit": None, "dirty": None, "status": "", "patch": ""}

    status = _run_command(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    patch = _run_command(["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"], root)
    untracked = _run_command(["git", "ls-files", "--others", "--exclude-standard", "-z"], root)
    status_text = status.stdout if status is not None and status.returncode == 0 else ""
    patch_text = patch.stdout if patch is not None and patch.returncode == 0 else ""
    if untracked is not None and untracked.returncode == 0:
        for relative_path in filter(None, untracked.stdout.split("\0")):
            added = _run_command(
                [
                    "git",
                    "diff",
                    "--binary",
                    "--no-index",
                    "--",
                    "/dev/null",
                    relative_path,
                ],
                root,
            )
            if added is not None and added.returncode in (0, 1):
                patch_text += added.stdout
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(status_text.strip()),
        "status": status_text,
        "patch": patch_text,
    }


def _process_command() -> str:
    """Return the actual process command line when the operating system exposes it."""
    proc_cmdline = Path("/proc/self/cmdline")
    try:
        args = [
            arg.decode(errors="surrogateescape") for arg in proc_cmdline.read_bytes().split(b"\0")
        ]
        args = [arg for arg in args if arg]
        if args:
            return shlex.join(args)
    except OSError:
        pass
    return shlex.join([sys.executable, *sys.argv])


def _process_start_marker() -> str:
    """Return a PID-reuse-resistant process start marker on Linux."""
    pid = os.getpid()
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text().split()
        start_ticks = stat_fields[21]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{boot_id}:{pid}:{start_ticks}"
    except (OSError, IndexError):
        return f"{pid}:{time.time_ns()}"


def _dependencies() -> dict[str, str]:
    """Return installed Python distribution versions for reproducibility."""
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


class RunRecorder:
    """Record one experiment's configuration, progress, metrics, and terminal result.

    Args:
        root: Project root. Runs are created below ``results/runs/<task>/<method>``.
        run_id: Stable identifier for this run. Existing run directories are rejected.
        config: Fully resolved run configuration.
        task_id: Project task identifier associated with the run.
    """

    def __init__(
        self,
        root: Path,
        run_id: str,
        config: dict[str, Any],
        task_id: str = "01",
    ) -> None:
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.task_id = task_id
        self._started_at = _utc_now()
        self.path = experiment_directory(self.root, run_id, config=config)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._finished = False
        self._started_monotonic = time.monotonic()
        self._process = {
            "pid": os.getpid(),
            "start_marker": _process_start_marker(),
        }

        if self.path.exists():
            raise FileExistsError(f"run already exists: {self.path}")
        protocol = None
        canonical = config.get("components", config)
        self.conditions = None
        if "env" in canonical:
            from drone_playground.artifacts.conditions import experiment_conditions

            self.conditions = experiment_conditions(canonical)
        if canonical.get("evaluation", {}).get("protocol"):
            from drone_playground.benchmarks import protocol_identity

            protocol = protocol_identity(canonical)
        self.root.mkdir(parents=True, exist_ok=True)
        git = _git_snapshot(self.root)
        self.path.mkdir(parents=True, exist_ok=False)
        archive = capture_source_archive(self.root, self.path)
        (self.path / "metrics").mkdir()
        (self.path / "checkpoints").mkdir()
        (self.path / "eval").mkdir()
        (self.path / "rollouts").mkdir()

        config_text = _json_text(config)
        _atomic_write_text(self.path / "resolved-config.json", config_text)
        command = _process_command()
        _atomic_write_text(self.path / "command.txt", command + "\n")
        _atomic_write_text(self.path / "code.patch", git["patch"])
        _atomic_write_text(self.path / "git-status.txt", git["status"])

        dependencies = _dependencies()
        _atomic_write_text(self.path / "dependencies.json", _json_text(dependencies))
        patch_hash = hashlib.sha256(git["patch"].encode()).hexdigest()
        config_hash = hashlib.sha256(config_text.encode()).hexdigest()
        manifest = {
            "run_id": run_id,
            "task_id": task_id,
            "started_at": self._started_at,
            "config": config,
            "config_path": "resolved-config.json",
            "config_sha256": config_hash,
            "command": command,
            "command_path": "command.txt",
            "process": self._process,
            "code": {
                "commit": git["commit"],
                "dirty": git["dirty"],
                "patch_path": "code.patch",
                "patch_sha256": patch_hash,
                "status_path": "git-status.txt",
            },
            "dependencies": dependencies,
            "dependencies_path": "dependencies.json",
            "benchmark_protocol": protocol,
            "experiment_conditions": self.conditions,
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
            },
        }
        if archive is not None:
            manifest["code"]["archive"] = archive
        _atomic_write_text(self.path / "manifest.json", _json_text(manifest))

        self._state: dict[str, Any] = {
            "run_id": run_id,
            "task_id": task_id,
            "status": "running",
            "phase": "starting",
            "step": None,
            "details": {},
            "started_at": self._started_at,
            "updated_at": self._started_at,
            "heartbeat_interval_s": _HEARTBEAT_INTERVAL_SECONDS,
            "heartbeat_count": 0,
            "process": self._process,
        }
        _atomic_write_text(self.path / "state.json", _json_text(self._state))
        self._metric_stream = (self.path / "metrics" / "metrics.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        self._writer = SummaryWriter(logdir=str(self.path / "metrics"), flush_secs=10)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"run-heartbeat-{run_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError(f"run is already finished: {self.run_id}")

    def _write_state(self) -> None:
        _atomic_write_text(self.path / "state.json", _json_text(self._state))

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(_HEARTBEAT_INTERVAL_SECONDS):
            with self._lock:
                if self._finished:
                    return
                self._state["heartbeat_count"] += 1
                self._state["updated_at"] = _utc_now()
                self._write_state()

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Record scalar metrics at an exact training or evaluation step."""
        if not isinstance(step, int):
            raise TypeError("step must be an int")
        values = {str(name): float(value) for name, value in metrics.items()}
        row = {"step": step, "metrics": values}
        with self._lock:
            self._ensure_open()
            self._metric_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._metric_stream.flush()
            for name, value in values.items():
                self._writer.add_scalar(name, value, global_step=step)
            self._writer.flush()
            self._state["step"] = step
            self._state["updated_at"] = _utc_now()
            self._write_state()

    def phase(self, name: str, step: int | None = None, **details: Any) -> None:
        """Record the current execution phase and optional structured details."""
        with self._lock:
            self._ensure_open()
            self._state["phase"] = name
            if step is not None:
                self._state["step"] = step
            self._state["details"] = details
            self._state["updated_at"] = _utc_now()
            self._write_state()

    def finish(self, status: str, **result: Any) -> None:
        """Write the terminal result and close metric writers and heartbeat activity."""
        with self._lock:
            self._ensure_open()
            self._stop_event.set()
        self._heartbeat_thread.join(timeout=2.0)

        finished_at = _utc_now()
        result_payload = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": status,
            "started_at": self._started_at,
            "finished_at": finished_at,
            "elapsed_s": time.monotonic() - self._started_monotonic,
            "experiment_conditions": self.conditions,
            **result,
        }
        with self._lock:
            self._state["status"] = status
            self._state["updated_at"] = finished_at
            self._state["finished_at"] = finished_at
            self._write_state()
            _atomic_write_text(self.path / "result.json", _json_text(result_payload))
            self._writer.flush()
            self._writer.close()
            self._metric_stream.flush()
            self._metric_stream.close()
            self._finished = True

    def record_environment(self, env) -> None:
        """Record executed clocks and sensing after construction, before rollouts."""
        if self.conditions is None:
            return
        self.conditions.update(
            control_frequency_hz=1 / env.dt,
            task_duration_s=env.episode_length * env.dt,
            sensor_timing=getattr(env, "sensor_timing", None),
            time_limit_kind=getattr(env, "time_limit_kind", None),
            environment_effects=getattr(env, "environment_effects", {}),
            command_distribution=getattr(env, "command_distribution", {}),
        )
        self.conditions["scene_distribution"] = getattr(env, "scene_distribution", None)
        self.conditions["scene"] = env.component_identity["scene"]
        if getattr(env, "sim", None) is not None:
            self.conditions["physics_frequency_hz"] = env.sim.freq
        self.conditions["safety_margin"]["body_radius_m"] = getattr(env, "body_radius", None)
        fixed = {
            key: value for key, value in self.conditions.items() if key not in ("method", "study")
        }
        self.conditions["conditions_sha256"] = hashlib.sha256(
            _json_text(fixed).encode()
        ).hexdigest()
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["experiment_conditions"] = self.conditions
        _atomic_write_text(manifest_path, _json_text(manifest))

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if self._finished:
            return False
        if exc is None:
            self.finish("completed")
            return False
        exception = {
            "type": exc_type.__name__ if exc_type is not None else type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
        }
        self.finish("failed", exception=exception)
        return False
