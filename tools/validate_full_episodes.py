"""Run the Command Builder's native methods through complete task termination.

Unlike the historical 0.14-second integration smoke, this tool NEVER overrides
task.duration. A task TIMEOUT therefore means the task actually exhausted its
configured horizon (Tracking 20 s, Racing 60 s, Navigation 300 s).

Use a separate results directory; the old short-flight evidence is untouched.
An operating-system wall limit is recorded as PROCESS_TIMEOUT, never as a
successful or timed-out physical episode. The runner is resumable and serial.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
TASK_DURATION = {
    name: float(
        OmegaConf.load(ROOT / "src/drone_playground/configs/task" / f"{name}.yaml").duration
    )
    for name in ("tracking", "racing", "navigation")
}
CATEGORIES = ("play:controller", "play:planner", "benchmark:controller", "benchmark:planner")


def atomic_json(path: Path, data: dict) -> None:
    """Publish complete records even if the process is interrupted mid-run."""
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def builder_catalog() -> list[dict]:
    """Use the authoritative command generator, not a second hardcoded matrix."""
    html = (ROOT / "docs/command-builder.html").read_text()
    source = html.split("<script>", 1)[1].split("</script>", 1)[0]
    process = subprocess.run(
        ["node"],
        input=source + "\nprocess.stdout.write(JSON.stringify(CommandBuilder.combinations()));\n",
        text=True,
        capture_output=True,
        check=True,
    )
    cases = json.loads(process.stdout)
    return [
        case
        for case in cases
        if case["mode"] in {"play", "benchmark"} and case["method"] in {"controller", "planner"}
    ]


def case_slug(index: int, case: dict) -> str:
    """Stable unique case ID shared by HTML, manifest and flight artifacts."""
    parts = [
        case["mode"],
        case["method"],
        case["task"],
        case.get("planner"),
        case.get("controller"),
        case.get("scene"),
        case.get("suite"),
    ]
    label = "_".join(str(value) for value in parts if value is not None)
    return f"{index:03d}_{re.sub(r'[^a-zA-Z0-9_-]', '_', label)}"


def physical_duration(case: dict) -> float:
    """Use the real task horizon, independent of how many scenes are selected."""
    return TASK_DURATION["navigation" if case["method"] == "planner" else case["task"]]


def build_command(case: dict, target: Path, args: argparse.Namespace) -> list[str]:
    """Preserve builder Hydra values and add only execution/capture controls."""
    original = shlex.split(case["command"])
    if original[:2] != ["pixi", "run"] or original[2] not in {"play", "benchmark"}:
        raise ValueError("Not a native Command Builder Play or Benchmark command")
    if any(field.startswith("task.duration=") for field in original):
        raise ValueError("Full-task acceptance must not override task duration")
    overrides = [
        "mode=" + original[2],
        *original[3:],
        "benchmark.episodes=1",
        "benchmark.record_replay=true",
        "benchmark.replay_episodes=1",
        "simulation.num_envs=1",
        "simulation.device=cpu",
        "output=" + str(target),
    ]
    if case["method"] == "planner":
        port = args.ego_port if case["planner"] == "ego" else args.super_port
        overrides.append(f"method.implementation.planner.address=127.0.0.1:{port}")
    if any(field.startswith("task.duration=") for field in overrides):
        raise AssertionError("Task duration was truncated")
    return [sys.executable, "-m", "drone_playground.cli", *overrides]


def runtime_environment() -> dict:
    """Run the same source via the active Pixi Python without GPU contention."""
    env = os.environ.copy()
    source = str(ROOT / "src")
    acados_python = env.get("ACADOS_PYTHON_PATH")
    extras = [source, *([acados_python] if acados_python else [])]
    env["PYTHONPATH"] = os.pathsep.join([*extras, env.get("PYTHONPATH", "")])
    env.update(
        JAX_PLATFORMS="cpu",
        SCIPY_ARRAY_API="1",
        MUJOCO_GL="egl",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        JAX_COMPILATION_CACHE_DIR=str(ROOT / ".cache/jax"),
    )
    if acados := env.get("ACADOS_SOURCE_DIR"):
        env["LD_LIBRARY_PATH"] = (
            str(Path(acados) / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        )
    return env


def load_episodes(directory: Path) -> list[dict]:
    """Read actual per-scene task termination reasons and flight durations."""
    file = directory / "episodes.csv"
    if not file.is_file():
        return []
    with file.open(newline="") as stream:
        return list(csv.DictReader(stream))


def perform(case: dict, index: int, output: Path, args: argparse.Namespace, attempt: int) -> dict:
    """Execute one untruncated case, preserving its complete logs and report."""
    slug = case_slug(index, case)
    if attempt:
        slug += f"_retry_{attempt}"
    directory = output / "runs" / slug
    directory.mkdir(parents=True, exist_ok=False)
    command = build_command(case, directory, args)
    start = time.monotonic()
    returncode = None
    infrastructure_error = None
    with (directory / "console.log").open("w") as stream:
        stream.write(
            "# Original Command Builder command\n# " + case["command"] + "\n"
            "# Full-task invocation: no task.duration override\n# " + shlex.join(command) + "\n\n"
        )
        stream.flush()
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=runtime_environment(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.wall_limit if args.wall_limit else None,
                check=False,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            infrastructure_error = "PROCESS_TIMEOUT: exceeded infrastructure wall limit"
        except OSError as error:
            infrastructure_error = f"PROCESS_ERROR: {type(error).__name__}: {error}"

    report_file = directory / "report.json"
    try:
        report = json.loads(report_file.read_text()) if report_file.is_file() else {}
    except (OSError, ValueError):
        report = {}
    episodes = load_episodes(directory)
    replays = [
        name
        for group in report.get("reports", {}).values()
        for name in group.get("browser_replays", [])
    ]
    duration = physical_duration(case)
    # A genuine task TIMEOUT is only legitimate when the entire configured
    # horizon was simulated. This explicitly rejects the old 0.14s smoke case.
    invalid_timeouts = [
        row
        for row in episodes
        if row["event"] == "TIMEOUT" and abs(float(row["flight_seconds"]) - duration) > 0.03
    ]
    missing_replay = any(not (directory / name).is_file() for name in replays)
    complete = bool(report.get("complete") and returncode == 0 and episodes)
    if infrastructure_error:
        status = (
            "process_timeout" if infrastructure_error.startswith("PROCESS_TIMEOUT") else "error"
        )
    elif not complete or invalid_timeouts or missing_replay:
        status = "error"
    elif any(row["event"] == "METHOD_FAILURE" for row in episodes):
        status = "method_failure"
    elif len(replays) != len(episodes):
        status = "missing_replay"
    else:
        status = "completed"
    return {
        "status": status,
        "command": case["command"],
        "executed_command": shlex.join(command),
        "source_case": index,
        "task_duration_seconds": duration,
        "simulated_flight_seconds": [float(row["flight_seconds"]) for row in episodes],
        "events": dict(Counter(row["event"] for row in episodes)),
        "episodes": len(episodes),
        "browser_replays": replays,
        "has_report": bool(report),
        "returncode": returncode,
        "process_error": infrastructure_error,
        "invalid_short_timeouts": len(invalid_timeouts),
        "run_relative": "runs/" + slug,
        "wall_seconds": round(time.monotonic() - start, 2),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "finished_at": time.time(),
    }


def write_index(directory: Path) -> None:
    """Publish the standard results viewer next to the finished flight artifacts."""
    (directory / "index.html").write_text((ROOT / "docs/full-episode-results.html").read_text())


def main() -> None:
    """Keep progressing through real task episodes until the requested subset ends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/command-builder-native-full-20261017")
    )
    parser.add_argument("--group", choices=("all", *CATEGORIES), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Number of new cases, 0 means all")
    parser.add_argument("--case", type=int, action="append", help="Exact catalog index, repeatable")
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry selected cases in new folders; preserve old runs",
    )
    parser.add_argument(
        "--wall-limit", type=int, default=0, help="Optional OS watchdog seconds; 0 disables"
    )
    parser.add_argument("--ego-port", type=int, default=50151)
    parser.add_argument("--super-port", type=int, default=50152)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = builder_catalog()
    current = {
        "schema": "native-full-episode-v1",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cases": cases,
    }
    catalog_path = output / "catalog.json"
    if catalog_path.is_file():
        existing = json.loads(catalog_path.read_text())
        if [x["command"] for x in existing["cases"]] != [x["command"] for x in cases]:
            raise ValueError("Command Builder changed: use a new output directory")
    else:
        atomic_json(catalog_path, current)
    manifest_path = output / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.is_file()
        else {
            "schema": "native-full-episode-v1",
            "task_duration_seconds": TASK_DURATION,
            "episode_count_per_scene": 1,
            "replay": "complete RScope trajectory and sampled browser poses",
            "smoke_duration_override": None,
            "cases": {},
        }
    )
    write_index(output)
    atomic_json(manifest_path, manifest)
    completed = 0
    for index, case in enumerate(cases):
        if args.case and index not in args.case:
            continue
        if args.group != "all" and case["mode"] + ":" + case["method"] != args.group:
            continue
        slug = case_slug(index, case)
        if (
            slug in manifest["cases"]
            and not args.retry
            and manifest["cases"][slug]["status"] != "running"
        ):
            continue
        if args.limit and completed >= args.limit:
            break
        print(f"START {index:03d} {case['mode']} {case['method']} {case['task']}", flush=True)
        previous = manifest["cases"].get(slug)
        attempt = 0
        while (output / "runs" / (slug + (f"_retry_{attempt}" if attempt else ""))).exists():
            attempt += 1
        manifest["cases"][slug] = {
            "status": "running",
            "run_relative": "runs/" + slug + (f"_retry_{attempt}" if attempt else ""),
            "events": {},
            "browser_replays": [],
        }
        atomic_json(manifest_path, manifest)
        result = perform(case, index, output, args, attempt)
        if previous is not None:
            result["previous_run_relative"] = previous.get("run_relative")
        manifest["cases"][slug] = result
        atomic_json(manifest_path, manifest)
        print(
            f"DONE {index:03d} {result['status']} events={result['events']} "
            f"flight_s={result['simulated_flight_seconds']} "
            f"replays={len(result['browser_replays'])} wall_s={result['wall_seconds']}",
            flush=True,
        )
        completed += 1
    print("SUMMARY", dict(Counter(v["status"] for v in manifest["cases"].values())), flush=True)


if __name__ == "__main__":
    main()
