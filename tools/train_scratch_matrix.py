"""Run the 3-algorithm x 4-condition scratch-learning matrix on a shared GPU.

No checkpoint initialization or historical policy transfer is permitted.
The project CLI remains the sole training lifecycle owner. This scheduler only
starts individual CLI runs, observes their recorded outcomes, and chooses a
new *scratch* trial after a measured failure. All mutable artifacts live under
the ignored results directory rather than accumulating transient documentation.

Example:
  python tools/train_scratch_matrix.py run --results-root /path/to/results/scratch_v02
  python tools/train_scratch_matrix.py status --results-root /path/to/results/scratch_v02
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    """A shared training recipe evaluated on three independent random seeds."""

    task: str
    algorithm: str
    experiment: str
    worlds: int
    horizon: int
    learning_rate: float

    @property
    def key(self):
        """Return the stable logical name of an algorithm/task condition."""
        return f"{self.task}_{self.algorithm}"


def cases():
    """Build 12 jobs per seed; Navigation static/dynamic share trained policies."""
    recipes = {
        "tracking": ("tracking_gpu", {"ppo": 1024, "apg": 256, "shac": 256}),
        "racing": ("racing", {"ppo": 512, "apg": 256, "shac": 256}),
        "depth": (
            "navigation_depth_scratch",
            {"ppo": 128, "apg": 128, "shac": 128},
        ),
        "lidar": (
            "navigation_lidar_scratch_safe",
            {"ppo": 64, "apg": 64, "shac": 64},
        ),
    }
    for task, (experiment, sizes) in recipes.items():
        for algorithm in ("ppo", "apg", "shac"):
            selected = experiment
            if task == "depth" and algorithm == "ppo":
                selected = "navigation_depth_scratch_ppo_asymmetric"
            elif task == "depth" and algorithm == "shac":
                selected = "navigation_depth_scratch_safe"
            elif task == "lidar" and algorithm == "ppo":
                selected = "navigation_lidar_scratch_ppo_asymmetric"
            learning_rate = (
                0.001 if task in ("tracking", "racing") and algorithm != "ppo" else 0.0003
            )
            yield Case(
                task,
                algorithm,
                selected,
                sizes[algorithm],
                64 if task in ("tracking", "racing") and algorithm == "ppo" else 32,
                learning_rate,
            )


def overrides(case, variant):
    """Propose bounded, named variations using measured previous trials.

    These are recipe alternatives, not hidden changes to the test protocol.
    Each alternative starts from independently sampled network weights.
    """
    result = [
        f"experiment={case.experiment}",
        f"learning={case.algorithm}",
        f"simulation.num_envs={case.worlds}",
        f"learning.options.horizon={case.horizon}",
        f"learning.options.lr={case.learning_rate}",
        "initial_checkpoint=null",
        "resume=null",
        "checkpoint=null",
        "learning.evaluation_interval=" + ("160" if case.task in ("depth", "lidar") else "100"),
        "learning.checkpoint_interval=" + ("80" if case.task in ("depth", "lidar") else "50"),
        "learning.log_interval=10",
    ]
    if variant == 0:
        return result
    if variant == 1:
        if case.task in ("depth", "lidar"):
            result.extend(
                [
                    "++learning.options.progress_reward_scale=0.5",
                    "++learning.options.height_boundary_weight=1.0",
                    "learning.options.lr=" + ("0.0001" if case.algorithm == "ppo" else "0.0005"),
                ]
            )
        elif case.task == "racing":
            result.extend(
                [
                    "++learning.options.progress_reward_scale=0.5",
                    "learning.options.lr=" + ("0.0001" if case.algorithm == "ppo" else "0.0005"),
                ]
            )
        else:
            result.append(
                "learning.options.lr=" + ("0.0001" if case.algorithm == "ppo" else "0.0005")
            )
    elif variant == 2:
        if case.task in ("depth", "lidar"):
            result.extend(
                [
                    "simulation.navigation_goal_observation=true",
                    "++learning.options.progress_reward_scale=0.5",
                    "++learning.options.height_boundary_weight=2.0",
                    "++learning.options.altitude_weight=2.0",
                    "learning.options.lr=0.0003",
                ]
            )
        elif case.task == "racing":
            result.extend(
                [
                    "++learning.options.progress_reward_scale=1.0",
                    "learning.options.horizon=32",
                    "learning.options.lr=0.0003",
                ]
            )
        else:
            result.extend(["learning.options.horizon=32", "learning.options.lr=0.0003"])
    else:
        raise ValueError(f"Unknown optimization candidate: {variant}")
    return result


def directory(root, case, variant, seed):
    """Identify a candidate without rewriting old results."""
    suffix = "" if variant == 0 else f"_trial{variant}"
    return root / f"{case.key}_s{seed}{suffix}"


def read_json(path):
    """Return validated JSON objects or None for absent and partial files."""
    try:
        result = json.loads(path.read_text())
        return result if isinstance(result, dict) else None
    except (OSError, ValueError):
        return None


def active_runs(root):
    """Observe actual Python processes, including runs started before this manager."""
    processes = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            parts = (process / "cmdline").read_bytes().split(b"\x00")
            command = [part.decode(errors="replace") for part in parts if part]
        except (OSError, PermissionError):
            continue
        if "-m" not in command or "drone_playground.cli" not in command:
            continue
        if "mode=train" not in command:
            continue
        for argument in command:
            if argument.startswith("output="):
                path = Path(argument.partition("=")[-1])
                if path.parent == root:
                    processes[str(path)] = int(process.name)
    return processes


def last_update(path):
    """Read the most recent measured update, never guess progress from elapsed time."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        try:
            data = json.loads(line)
            if data.get("event") == "update":
                return int(data["update"])
        except (ValueError, KeyError, TypeError):
            continue
    return 0


def consecutive_passes(path):
    """Recover the genuine consecutive success streak from evaluation events."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        try:
            data = json.loads(line)
            if data.get("event") == "checkpoint_evaluation":
                return int(data["consecutive_passes"])
        except (ValueError, KeyError, TypeError):
            continue
    return 0


def trial_status(path, running):
    """Trust the real run record and detect interrupted/vanished processes."""
    report = read_json(path / "run.json")
    active = str(path) in running
    if report is not None:
        status = report.get("status", "unknown")
        if status == "accepted":
            if report.get("benchmark_passed") is True and "initialization" not in report:
                return "accepted"
            return "invalid"
        if status in {"failed", "interrupted", "frozen_benchmark_failed"}:
            return status
        if active:
            return "running"
        if status == "running":
            state_archive = path / "checkpoints" / "latest.training.zip"
            proposal = read_json(path / "proposal.json")
            restarts = read_json(path / "restarts.json") or {}
            if state_archive.is_file() and proposal is not None and restarts.get("attempts", 0) < 3:
                return "recoverable"
            return "stale"
    if active:
        return "running"
    if path.exists() and (path / "console.log").exists():
        return "stale"
    return "missing"


def best_evaluation_score(path):
    """Return the best measured mean main-scene success rate, if recorded."""
    values = []
    for report_path in (path / "checkpoints").glob("step-*/report.json"):
        report = read_json(report_path)
        if report is None:
            continue
        scene_reports = report.get("reports", {})
        if not isinstance(scene_reports, dict):
            continue
        rates = [
            float(result["success_rate"])
            for scene, result in scene_reports.items()
            if scene in ("empty", "racing", "S01", "S02", "S03", "D01", "D02", "D03")
            and isinstance(result, dict)
            and isinstance(result.get("success_rate"), (float, int))
        ]
        if rates:
            values.append(sum(rates) / len(rates))
    return max(values) if values else None


def eligible(case, root, running, *, candidates):
    """Choose the next testable seed/recipe, never copying a checkpoint."""
    history = []
    for variant in range(candidates):
        previous = []
        for seed in (0, 1, 2):
            path = directory(root, case, variant, seed)
            status = trial_status(path, running)
            previous.append({"seed": seed, "status": status, "score": best_evaluation_score(path)})
        history.append({"variant": variant, "runs": previous})
        # Screen each variant with seed 0 before spending seeds 1 and 2.
        if previous[0]["status"] in (
            "failed",
            "interrupted",
            "frozen_benchmark_failed",
            "invalid",
            "stale",
        ):
            continue
        if previous[0]["status"] in ("missing", "running", "recoverable"):
            return (
                (
                    "launch"
                    if previous[0]["status"] == "missing"
                    else "resume"
                    if previous[0]["status"] == "recoverable"
                    else "waiting"
                ),
                variant,
                0,
                history,
            )
        if any(
            item["status"]
            in ("failed", "interrupted", "frozen_benchmark_failed", "invalid", "stale")
            for item in previous[1:]
        ):
            continue
        for seed in (1, 2):
            if previous[seed]["status"] != "accepted":
                return (
                    (
                        "launch"
                        if previous[seed]["status"] == "missing"
                        else "resume"
                        if previous[seed]["status"] == "recoverable"
                        else "waiting"
                    ),
                    variant,
                    seed,
                    history,
                )
        return "accepted", variant, None, history
    return "unresolved", None, None, history


def resources():
    """Get available VRAM and host RAM for admission control."""
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    memory_mib, gpu_util = [int(float(value.strip())) for value in output.split(",")]
    mem = Path("/proc/meminfo").read_text()
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", mem, flags=re.MULTILINE)
    available_ram = int(match.group(1)) // 1024 if match else 0
    return {"gpu_free_mib": memory_mib, "gpu_percent": gpu_util, "ram_free_mib": available_ram}


def tmux_launch(project, path, runner, session_name, *, append=False):
    """Launch the configured original CLI without owning its training lifecycle."""
    environment = (
        "env SCIPY_ARRAY_API=1 XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "PYTHONPATH=src "
        f"JAX_COMPILATION_CACHE_DIR={shlex.quote(str(project / '.cache' / 'jax'))} "
    )
    redirect = ">>" if append else ">"
    script = (
        f"cd {shlex.quote(str(project))} && "
        f"{environment}{shlex.join(runner)} "
        f"{redirect} {shlex.quote(str(path / 'console.log'))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, script], check=True)


def launch(project, root, python, case, variant, seed):
    """Run the original project CLI with explicit scratch initialization."""
    path = directory(root, case, variant, seed)
    if path.exists():
        raise RuntimeError(f"Refusing to reuse an existing training directory: {path}")
    path.mkdir(parents=True)
    overrides_list = ["mode=train", *overrides(case, variant), f"seed={seed}", f"output={path}"]
    name = f"dp-scr-{case.task[:2]}-{case.algorithm}-{seed}-{variant}"
    if (
        subprocess.run(["tmux", "has-session", "-t", f"={name}"], capture_output=True).returncode
        == 0
    ):
        raise RuntimeError(f"Refusing to overwrite tmux session: {name}")
    runner = [str(python), "-m", "drone_playground.cli", *overrides_list]
    (path / "proposal.json").write_text(
        json.dumps(
            {
                "task": case.task,
                "algorithm": case.algorithm,
                "variant": variant,
                "seed": seed,
                "from_scratch": True,
                "initial_checkpoint": None,
                "command": runner,
                "selection": "three_consecutive_checkpoint_evaluations",
            },
            indent=2,
        )
    )
    tmux_launch(project, path, runner, name)
    return name


def resume(project, root, case, variant, seed):
    """Restore only the same trial's complete own state after an abnormal stop."""
    path = directory(root, case, variant, seed)
    proposal = read_json(path / "proposal.json")
    archive = path / "checkpoints" / "latest.training.zip"
    if proposal is None or not archive.is_file() or proposal.get("initial_checkpoint") is not None:
        raise RuntimeError(f"No valid scratch recovery state for {path}")
    runner = list(proposal["command"])
    if "resume=null" not in runner or "initial_checkpoint=null" not in runner:
        raise RuntimeError(f"Refusing non-scratch resumption for {path}")
    runner[runner.index("resume=null")] = f"resume={archive}"
    restart_file = path / "restarts.json"
    state = read_json(restart_file) or {"attempts": 0}
    state["attempts"] += 1
    atomic_json(restart_file, state)
    name = f"dp-scr-{case.task[:2]}-{case.algorithm}-{seed}-{variant}-resume"
    tmux_launch(project, path, runner, name, append=True)
    return name


def report(root, running, candidates):
    """Return one compact matrix snapshot; stage logs stay inside each trial."""
    rows = []
    for case in cases():
        state, variant, seed, history = eligible(case, root, running, candidates=candidates)
        rows.append(
            {
                "cell": case.key,
                "outcome": state,
                "next_trial": variant,
                "next_seed": seed,
                "variants": history,
            }
        )
    return {
        "source": "random_initialization_only",
        "matrix_cells": 18,
        "train_families": 12,
        "required_seeds": [0, 1, 2],
        "active_runs": list(running),
        "families_accepted": sum(row["outcome"] == "accepted" for row in rows),
        "families_unresolved": sum(row["outcome"] == "unresolved" for row in rows),
        "rows": rows,
    }


def atomic_json(path, payload):
    """Publish status without exposing a half-written snapshot."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def supervise(root, max_parallel, update_budget, candidates):
    """Observe measured outcomes and admit new trials while GPU capacity exists."""
    project = Path(__file__).resolve().parents[1]
    python = Path(
        os.environ.get("DRONE_MATRIX_PYTHON", str(project / ".pixi/envs/default/bin/python"))
    )
    if not python.is_file():
        raise RuntimeError(f"Python environment not installed: {python}")
    root.mkdir(parents=True, exist_ok=True)
    while True:
        running = active_runs(root)
        for path_string, pid in list(running.items()):
            path = Path(path_string)
            streak = consecutive_passes(path / "metrics.jsonl")
            allowance = 160 * streak if streak else 0
            if (
                last_update(path / "metrics.jsonl") >= update_budget + allowance
                and not (path / "stop_requested.json").exists()
            ):
                (path / "stop_requested.json").write_text(
                    json.dumps(
                        {"reason": "trial_update_budget", "threshold": update_budget, "pid": pid}
                    )
                )
                os.kill(pid, signal.SIGINT)
        snapshot = report(root, running, candidates)
        snapshot["resources"] = resources()
        atomic_json(root / "matrix.json", snapshot)
        print(
            f"accepted={snapshot['families_accepted']}/12 "
            f"active={len(running)} "
            f"VRAM_free={snapshot['resources']['gpu_free_mib']}MiB "
            f"GPU={snapshot['resources']['gpu_percent']}%",
            flush=True,
        )
        if snapshot["families_accepted"] + snapshot["families_unresolved"] == 12:
            return
        if (
            len(running) < max_parallel
            and snapshot["resources"]["gpu_free_mib"] >= 6500
            and snapshot["resources"]["ram_free_mib"] >= 10000
        ):
            for case in cases():
                action, variant, seed, _ = eligible(case, root, running, candidates=candidates)
                if action not in ("launch", "resume"):
                    continue
                if action == "resume":
                    session = resume(project, root, case, variant, seed)
                else:
                    session = launch(project, root, python, case, variant, seed)
                print(
                    f"{action} {session}: {case.key} seed={seed} trial={variant}",
                    flush=True,
                )
                break
        time.sleep(20)


def main():
    """Run the persistent scheduler or display the authoritative matrix snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status", "plan"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--update-budget", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=3)
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    if args.python is not None:
        os.environ["DRONE_MATRIX_PYTHON"] = str(args.python.resolve())
    if args.mode == "plan":
        for case in cases():
            print(case.key, overrides(case, 0), "seeds=0,1,2")
    elif args.mode == "status":
        print(json.dumps(report(root, active_runs(root), args.candidates), indent=2))
    else:
        if args.max_parallel < 1 or args.update_budget < 160 or not 1 <= args.candidates <= 3:
            parser.error("Invalid concurrency, update budget or number of candidates")
        supervise(root, args.max_parallel, args.update_budget, args.candidates)


if __name__ == "__main__":
    main()
