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
import hashlib
import json
import os
import random
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
    elif 3 <= variant <= 8:
        if case.task == "racing":
            # DiffAero racing rewards place substantial weight on progress through gates.
            # It is dense training reward only; gate order and C3 remain unchanged.
            tuning = {
                3: (4.0, 0.0003, 32, 50.0),
                4: (10.0, 0.0001, 32, 50.0),
                5: (6.0, 0.0005, 16, 30.0),
                6: (2.0, 0.001, 64, 50.0),
                7: (10.0, 0.0003, 64, 50.0),
                8: (4.0, 0.0001, 64, 30.0),
            }
            progress, lr, horizon, failure_cost = tuning[variant]
            result.extend(
                [
                    f"++learning.options.progress_reward_scale={progress}",
                    f"++learning.options.failure_cost={failure_cost}",
                    f"learning.options.lr={lr}",
                    f"learning.options.horizon={horizon}",
                ]
            )
            if case.algorithm == "ppo":
                result.append("learning.options.entropy_weight=0.002")
        elif case.task == "tracking":
            result.extend(
                [
                    f"learning.options.lr={0.0002 if variant % 2 else 0.0005}",
                    f"learning.options.horizon={16 if variant % 3 == 0 else 32}",
                ]
            )
        elif case.task in ("depth", "lidar"):
            # Network structures stay identical across PPO/APG/SHAC. Variants
            # differ only in recorded optimization and reward schedules.
            grid = {
                3: (0.25, 1.0, 0.0001, 0.5),
                4: (0.75, 2.0, 0.0002, 1.0),
                5: (1.0, 2.0, 0.0001, 0.5),
                6: (0.25, 1.0, 0.0003, 0.5),
                7: (1.0, 3.0, 0.00015, 0.75),
                8: (0.5, 0.5, 0.0002, 1.0),
            }
            progress, boundary, lr, perception_weight = grid[variant]
            result.extend(
                [
                    f"++learning.options.progress_reward_scale={progress}",
                    f"++learning.options.height_boundary_weight={boundary}",
                    f"learning.options.lr={lr}",
                    f"learning.options.perception_weight={perception_weight}",
                ]
            )
            if variant in (4, 5, 7):
                result.append("simulation.navigation_goal_observation=true")
            if case.algorithm == "ppo":
                # Liu point-cloud PPO struggled with early zero-success
                # privileged value fits; evaluate an actual sensor-critic
                # alternative without granting the Actor privileged truth.
                if variant in (3, 5, 6, 8):
                    result.extend(
                        [
                            "learning.options.critic_uses_privileged=false",
                            "++learning.options.critic_uses_sensor=true",
                        ]
                    )
                result.extend(
                    [
                        "learning.options.initial_log_std=-2.5",
                        "learning.options.critic_lr=0.0005",
                        "learning.options.clip_epsilon=0.15",
                    ]
                )
            else:
                result.append(f"learning.options.temporal_gradient_alpha={0.916290731874155}")
            if case.task == "lidar" or variant >= 5:
                # More frequent independent evaluations can detect a genuine
                # three-checkpoint stability streak without cherry-picking
                # individual episodes. Final benchmarks stay untouched.
                result.extend(
                    ["learning.evaluation_interval=80", "learning.checkpoint_interval=40"]
                )
        else:
            raise ValueError(f"Unsupported task: {case.task}")
    else:
        raise ValueError(f"Unknown optimization candidate: {variant}")
    # Hydra override order is significant. Keep the final intentional value
    # for each option, but do not pass two spellings of the same key.
    seen = set()
    normalized = []
    for option in reversed(result):
        name = option.partition("=")[0].lstrip("+")
        if name not in seen:
            normalized.append(option)
            seen.add(name)
    return list(reversed(normalized))


def adaptive_overrides(case, variant, root, seed):
    """Ask for the next bounded recipe using genuine prior checkpoint results.

    This follows the Optim-Agent ask/tell principle: candidate choices depend
    on measured checkpoint evaluations, all parameters are bounded, and the
    final result is reported by the real environment. It never transfers
    network or optimizer parameters.
    """
    if variant < 9:
        return overrides(case, variant), None
    if seed > 0:
        pilot = read_proposal(directory(root, case, variant, 0))
        if pilot is None or pilot.get("initial_checkpoint") is not None:
            raise RuntimeError(f"Missing seed-0 recipe for {case.key} candidate {variant}")
        options = [
            arg
            for arg in pilot["command"][3:]
            if not arg.startswith(("seed=", "output=")) and arg != "mode=train"
        ]
        return options, pilot.get("parent_variant")

    history = []
    for previous in range(variant):
        p = directory(root, case, previous, 0)
        value = best_evaluation_score(p)
        if value is None:
            continue
        recipe = read_proposal(p)
        if recipe is not None:
            history.append((value, previous, recipe))
    history.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if history:
        _, parent_variant, parent = history[0]
        options = [
            arg
            for arg in parent["command"][3:]
            if not arg.startswith(
                (
                    "seed=",
                    "output=",
                    "learning.selection=",
                    "learning.required_consecutive_passes=",
                )
            )
            and arg != "mode=train"
        ]
    else:
        parent_variant = 0
        options = overrides(case, 0)

    def numeric(name, default):
        for option in reversed(options):
            key, separator, value = option.partition("=")
            if separator and key.lstrip("+") == name:
                return float(value)
        return default

    digest = hashlib.sha256(f"{case.key}:{variant}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    base_lr = numeric("learning.options.lr", case.learning_rate)
    chosen_lr = min(0.0015, max(0.00003, base_lr * rng.choice((0.4, 0.65, 0.8, 1.25, 1.6))))
    options.append(f"learning.options.lr={chosen_lr:.8f}")
    horizon_space = (
        (32, 64, 128) if case.task == "racing" and case.algorithm == "shac" else (16, 32, 64)
    )
    options.append(f"learning.options.horizon={rng.choice(horizon_space)}")

    if case.task == "racing":
        progress = rng.choice((2.0, 4.0, 6.0, 10.0, 15.0))
        options.extend(
            [
                f"++learning.options.progress_reward_scale={progress}",
                f"++learning.options.failure_cost={rng.choice((20.0, 40.0, 80.0))}",
            ]
        )
    elif case.task in ("depth", "lidar"):
        options.extend(
            [
                f"++learning.options.progress_reward_scale={rng.choice((0.25, 0.5, 1.0, 2.0))}",
                f"++learning.options.height_boundary_weight={rng.choice((0.5, 1.0, 2.0, 4.0))}",
                f"learning.options.perception_weight={rng.choice((0.25, 0.5, 0.75, 1.0))}",
                f"learning.scene_updates={rng.choice((20, 40, 80))}",
            ]
        )
        if rng.random() < 0.75:
            options.append("simulation.navigation_goal_observation=true")
        options.extend(
            [
                f"learning.evaluation_interval={rng.choice((80, 160))}",
                "learning.checkpoint_interval=40",
            ]
        )
    if case.algorithm == "ppo":
        options.extend(
            [
                f"learning.options.initial_log_std={rng.choice((-3.5, -3.0, -2.5, -2.0))}",
                f"learning.options.critic_lr={rng.choice((0.0003, 0.0005, 0.001))}",
                f"learning.options.clip_epsilon={rng.choice((0.1, 0.15, 0.2))}",
                f"learning.options.entropy_weight={rng.choice((0.0, 0.001, 0.005))}",
            ]
        )
        if case.task in ("depth", "lidar"):
            critic_type = rng.choice(("state", "sensor", "privileged"))
            privileged = str(critic_type == "privileged").lower()
            sensor = str(critic_type == "sensor").lower()
            options.extend(
                [
                    f"++learning.options.critic_uses_privileged={privileged}",
                    f"++learning.options.critic_uses_sensor={sensor}",
                ]
            )
    if case.algorithm == "shac":
        options.extend(
            [
                f"learning.options.target_alpha={rng.choice((0.95, 0.99, 0.995, 0.999))}",
                f"learning.options.critic_lr={rng.choice((0.0003, 0.001, 0.003))}",
                f"learning.options.critic_epochs={rng.choice((2, 4, 8))}",
            ]
        )
    if case.algorithm in ("apg", "shac") and case.task in ("depth", "lidar"):
        options.append(
            f"learning.options.temporal_gradient_alpha={rng.choice((0.0, 0.35, 0.916290731874155))}"
        )

    seen = set()
    unique = []
    for option in reversed(options):
        key = option.partition("=")[0].lstrip("+")
        if key not in seen:
            unique.append(option)
            seen.add(key)
    return list(reversed(unique)), parent_variant


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


def read_proposal(path):
    """Find a trial proposal in the compact scheduler directory or old layout."""
    return read_json(path.parent / ".scheduler" / "proposals" / f"{path.name}.json") or read_json(
        path / "proposal.json"
    )


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
        header = read_json(path.parent / "run.json") or {}
        return int(header.get("last_update") or header.get("updates") or 0)
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
        header = read_json(path.parent / "run.json") or {}
        evaluation = header.get("last_evaluation", {})
        return int(evaluation.get("consecutive_passes", 0))
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
            selection = report.get("selection", {})
            try:
                configured = (path / "config.yaml").read_text()
            except OSError:
                configured = ""
            fresh = re.search(r"^initial_checkpoint:\s*null\s*$", configured, re.MULTILINE)
            rule = selection.get("rule")
            benchmark_dir = report.get("benchmark_directory")
            benchmark = (
                read_json(path / benchmark_dir / "report.json")
                if isinstance(benchmark_dir, str)
                else None
            )
            frozen_verified = (
                benchmark is not None
                and benchmark.get("mode") == "benchmark"
                and benchmark.get("complete") is True
                and benchmark.get("passed") is True
                and benchmark.get("seed_base", 0) >= 2_000_000
            )
            if rule == "first_full_validation_pass":
                selected_report = read_json(path / selection.get("full_validation", "missing"))
                validated = (
                    selection.get("consecutive_passes") == 1
                    and selected_report is not None
                    and selected_report.get("passed") is True
                    and selected_report.get("complete") is True
                    and selected_report.get("mode") == "checkpoint_eval"
                    and 1_500_000 <= selected_report.get("seed_base", 0) < 2_000_000
                )
            else:
                validated = (
                    rule == "last_of_first_three_consecutive_passes"
                    and selection.get("consecutive_passes", 0) >= 3
                )
            if (
                report.get("benchmark_passed") is True
                and frozen_verified
                and validated
                and fresh is not None
                and "initialization" not in report
            ):
                return "accepted"
            return "invalid"
        if status == "interrupted":
            proposal = read_proposal(path)
            restarts = read_json(path / "restarts.json") or {}
            archive = latest_training_checkpoint(path)
            stopped_by_budget = (path / "stop_requested.json").is_file()
            score = best_evaluation_score(path)
            promising = score is not None and score >= 0.65
            if (
                stopped_by_budget
                and proposal is not None
                and archive is not None
                and promising
                and restarts.get("attempts", 0) < 4
            ):
                return "recoverable"
            return status
        if status in {"failed", "frozen_benchmark_failed"}:
            return status
        if active:
            return "running"
        if status == "running":
            state_archive = latest_training_checkpoint(path)
            proposal = read_proposal(path)
            restarts = read_json(path / "restarts.json") or {}
            if (
                state_archive is not None
                and proposal is not None
                and restarts.get("attempts", 0) < 3
            ):
                return "recoverable"
            return "stale"
    if active:
        return "running"
    if path.exists() and (
        (path / "console.log").exists()
        or (path.parent / ".scheduler" / "logs" / f"{path.name}.log").exists()
    ):
        return "stale"
    return "missing"


def best_evaluation_score(path):
    """Rank candidates using measured diagnostics or complete validation only.

    The score is for tuning, never for declaring C2-C5 acceptance.
    """
    values = []
    reports = list((path / "checkpoints").glob("step-*/report.json"))
    reports.extend((path / "checkpoints").glob("step-*/screening/report.json"))
    reports.extend((path / "checkpoints").glob("model_*/report.json"))
    reports.extend((path / "checkpoints").glob("model_*/screening/report.json"))
    for report_path in reports:
        report = read_json(report_path)
        if report is None:
            continue
        scene_reports = report.get("reports", {})
        if not isinstance(scene_reports, dict):
            continue
        rates = []
        for scene, result in scene_reports.items():
            if scene not in ("empty", "racing", "S01", "S02", "S03", "D01", "D02", "D03"):
                continue
            if not isinstance(result, dict):
                continue
            success_rate = result.get("success_rate")
            if not isinstance(success_rate, (float, int)):
                continue
            score = float(success_rate)
            if scene == "empty":
                rmse = result.get("successful_position_rmse")
                score *= min(1.0, 0.25 / rmse) if isinstance(rmse, (float, int)) and rmse > 0 else 0
            rates.append(score)
        if rates:
            values.append(sum(rates) / len(rates))
    return max(values) if values else None


def latest_training_checkpoint(path):
    """Find a complete original zip or newest Orbax training state in this run."""
    legacy = path / "checkpoints" / "latest.training.zip"
    if legacy.is_file():
        return legacy
    candidates = sorted((path / "checkpoints").glob("model_*/training"), reverse=True)
    for candidate in candidates:
        if (candidate / "commit_success.txt").is_file() and (
            candidate.parent / "metadata.json"
        ).is_file():
            return candidate.parent
    return None


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
    # A fully accepted three-seed recipe must WIN, even if another earlier
    # candidate has a recoverable but unfinished run. Otherwise progress can
    # regress from accepted to pending after extending the search space.
    for item in history:
        if all(entry["status"] == "accepted" for entry in item["runs"]):
            return "accepted", item["variant"], None, history
    for item in history:
        variant, previous = item["variant"], item["runs"]
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
    logs = path.parent / ".scheduler" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logfile = logs / f"{path.name}.log"
    if append and (path / "console.log").is_file() and not logfile.is_file():
        logfile.write_bytes((path / "console.log").read_bytes())
    script = (
        f"cd {shlex.quote(str(project))} && "
        f"{environment}{shlex.join(runner)} "
        f"{redirect} {shlex.quote(str(logfile))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, script], check=True)


def launch(project, root, python, case, variant, seed):
    """Run the original project CLI with explicit scratch initialization."""
    path = directory(root, case, variant, seed)
    if path.exists():
        raise RuntimeError(f"Refusing to reuse an existing training directory: {path}")
    path.mkdir(parents=True)
    suggested, parent_variant = adaptive_overrides(case, variant, root, seed)
    if seed > 0:
        # Previously completed seed-0 runs can predate the three-stage
        # protocol. Keep the same acceptance contract across all three seeds
        # of that candidate rather than changing it after seeing results.
        origin = directory(root, case, variant, 0) / "config.yaml"
        if origin.is_file():
            match = re.search(r"^\s+selection:\s*(\S+)\s*$", origin.read_text(), re.MULTILINE)
            if match and match.group(1) == "last_of_first_three_consecutive_passes":
                suggested.extend(
                    [
                        "learning.selection=last_of_first_three_consecutive_passes",
                        "learning.required_consecutive_passes=3",
                    ]
                )
    overrides_list = ["mode=train", *suggested, f"seed={seed}", f"output={path}"]
    name = f"dp-scr-{case.task[:2]}-{case.algorithm}-{seed}-{variant}"
    if (
        subprocess.run(["tmux", "has-session", "-t", f"={name}"], capture_output=True).returncode
        == 0
    ):
        raise RuntimeError(f"Refusing to overwrite tmux session: {name}")
    runner = [str(python), "-m", "drone_playground.cli", *overrides_list]
    proposal_dir = root / ".scheduler" / "proposals"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    (proposal_dir / f"{path.name}.json").write_text(
        json.dumps(
            {
                "task": case.task,
                "algorithm": case.algorithm,
                "variant": variant,
                "parent_variant": parent_variant,
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
    proposal = read_proposal(path)
    archive = latest_training_checkpoint(path)
    if proposal is None or archive is None or proposal.get("initial_checkpoint") is not None:
        raise RuntimeError(f"No valid scratch recovery state for {path}")
    runner = list(proposal["command"])
    if "resume=null" not in runner or "initial_checkpoint=null" not in runner:
        raise RuntimeError(f"Refusing non-scratch resumption for {path}")
    runner[runner.index("resume=null")] = f"resume={archive}"
    # Strict v2 restoration checks the entire saved Learning recipe. Old
    # in-flight runs must keep the exact old protocol when the project default
    # changes. Only source/CLI defaults are upgraded; saved model weights and
    # optimizer/RNG histories are not rewritten.
    config_file = path / "config.yaml"
    saved = config_file.read_text()
    match = re.search(r"^\s+selection:\s*(\S+)\s*$", saved, re.MULTILINE)
    if match and match.group(1) == "last_of_first_three_consecutive_passes":
        runner.extend(
            [
                "learning.selection=last_of_first_three_consecutive_passes",
                "learning.required_consecutive_passes=3",
            ]
        )
    old_budget = read_json(path / "budget.json") or {}
    last = last_update(path / "metrics.jsonl")
    # Resume the SAME randomly initialized run, including its optimizer, RNG
    # and recurrent history. Never import parameters from another experiment.
    budget_limit = max(last, int(old_budget.get("limit", 1000))) + 1000
    atomic_json(path / "budget.json", {"limit": budget_limit, "source": "same_run_resume"})
    marker = path / "stop_requested.json"
    if marker.exists():
        with (path / "stop_history.jsonl").open("a") as stream:
            stream.write(marker.read_text() + "\n")
        marker.unlink()
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
    cells = []
    for row in rows:
        if row["cell"].startswith(("depth_", "lidar_")):
            cells.extend(
                {"cell": f"{row['cell']}_{kind}", "outcome": row["outcome"]}
                for kind in ("static", "dynamic")
            )
        else:
            cells.append({"cell": row["cell"], "outcome": row["outcome"]})
    return {
        "source": "random_initialization_only",
        "matrix_cells": len(cells),
        "train_families": 12,
        "required_seeds": [0, 1, 2],
        "active_runs": list(running),
        "families_accepted": sum(row["outcome"] == "accepted" for row in rows),
        "families_unresolved": sum(row["outcome"] == "unresolved" for row in rows),
        "accepted_cells": sum(cell["outcome"] == "accepted" for cell in cells),
        "cells": cells,
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
            limit = (read_json(path / "budget.json") or {}).get("limit", update_budget)
            if (
                last_update(path / "metrics.jsonl") >= limit + allowance
                and not (path / "stop_requested.json").exists()
            ):
                (path / "stop_requested.json").write_text(
                    json.dumps({"reason": "trial_update_budget", "threshold": limit, "pid": pid})
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
            # Prioritize recipes that already demonstrated main-scene arrivals
            # before exploring families with zero successes. This uses
            # checkpoint evaluations, never frozen benchmark results for tuning.
            priority = {
                "depth_shac": 0,
                "lidar_apg": 1,
                "lidar_shac": 2,
                "depth_ppo": 3,
                "racing_ppo": 4,
                "racing_shac": 5,
                "lidar_ppo": 6,
            }
            for case in sorted(cases(), key=lambda item: priority.get(item.key, 7)):
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
    parser.add_argument("--candidates", type=int, default=48)
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
        if args.max_parallel < 1 or args.update_budget < 160 or not 1 <= args.candidates <= 256:
            parser.error("Invalid concurrency, update budget or number of candidates")
        supervise(root, args.max_parallel, args.update_budget, args.candidates)


if __name__ == "__main__":
    main()
