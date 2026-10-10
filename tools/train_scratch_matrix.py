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
import math
import os
import random
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
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


def cases(regime="fixed", only=()):
    """Build requested families without mixing procedural and old results."""
    if regime not in {"fixed", "procedural"}:
        raise ValueError("Training regime must be fixed or procedural")
    only = frozenset(only)
    allowed = {
        f"{task}_{algorithm}"
        for task in ("tracking", "racing", "depth", "lidar")
        for algorithm in ("ppo", "apg", "shac")
    }
    if only - allowed:
        raise ValueError(f"Unknown training family: {sorted(only - allowed)}")
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
    if regime == "procedural":
        recipes["depth"] = (
            "navigation_procedural_depth",
            {"ppo": 16, "apg": 16, "shac": 16},
        )
        recipes["lidar"] = (
            "navigation_procedural_lidar",
            {"ppo": 16, "apg": 16, "shac": 16},
        )
    for task, (experiment, sizes) in recipes.items():
        for algorithm in ("ppo", "apg", "shac"):
            if only and f"{task}_{algorithm}" not in only:
                continue
            selected = experiment
            if regime == "procedural" and task in ("depth", "lidar"):
                selected = experiment + ("_ppo" if algorithm == "ppo" else "")
            elif regime == "procedural" and task == "racing" and algorithm == "ppo":
                selected = "racing_ppo_stable"
            elif regime == "procedural" and task == "racing" and algorithm == "shac":
                selected = "racing_shac_stable"
            elif task == "depth" and algorithm == "ppo":
                selected = "navigation_depth_scratch_ppo_asymmetric"
            elif task == "depth" and algorithm == "shac":
                selected = "navigation_depth_scratch_safe"
            elif task == "lidar" and algorithm == "ppo":
                selected = "navigation_lidar_scratch_ppo_asymmetric"
            learning_rate = (
                0.0001
                if regime == "procedural" and task in ("depth", "lidar") and algorithm == "ppo"
                else 0.0001
                if regime == "procedural" and task == "racing" and algorithm == "shac"
                else 0.001
                if task in ("tracking", "racing") and algorithm != "ppo"
                else 0.0003
            )
            yield Case(
                task,
                algorithm,
                selected,
                sizes[algorithm],
                32
                if regime == "procedural" and task == "racing" and algorithm == "ppo"
                else 64
                if task in ("tracking", "racing") and algorithm == "ppo"
                else 32,
                learning_rate,
            )


def overrides(case, variant):
    """Propose bounded, named variations using measured previous trials.

    These are recipe alternatives, not hidden changes to the test protocol.
    Each alternative starts from independently sampled network weights.
    """
    if (
        variant == 4
        and case.algorithm == "ppo"
        and case.task in ("depth", "lidar")
        and "procedural" in case.experiment
    ):
        # Controlled old-recipe ablation, not an automatic replacement of PPO:
        # same frozen evaluation, different velocity/yaw Action, motion-aware
        # reward, per-episode free-start sampling, and PPO update batch size.
        return [
            f"experiment=navigation_procedural_{case.task}_ppo_velocity",
            "learning=ppo",
            "simulation.num_envs=64",
            "learning.options.horizon=32",
            "learning.options.lr=0.0005",
            "initial_checkpoint=null",
            "resume=null",
            "checkpoint=null",
            "learning.evaluation_interval=160",
            "learning.checkpoint_interval=80",
            "learning.log_interval=10",
        ]
    experiment = case.experiment
    if case.task == "lidar" and variant >= 3 and "procedural" not in experiment:
        experiment = (
            "navigation_lidar_scratch_ppo_asymmetric_sparse"
            if case.algorithm == "ppo"
            else "navigation_lidar_scratch_sparse"
        )
    result = [
        f"experiment={experiment}",
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
    if variant < 9:
        return overrides(case, variant), None

    if (
        variant in (9, 10)
        and case.algorithm == "ppo"
        and case.task in ("depth", "lidar")
        and "procedural" in case.experiment
    ):
        # Paired start-distribution test: keep all optimization settings fixed
        # between the control and candidate, including PPO network/critic options.
        if variant == 9:
            evidence = [
                (score, previous, proposal)
                for previous in range(9)
                if (score := best_evaluation_score(directory(root, case, previous, 0))) is not None
                and (proposal := read_proposal(directory(root, case, previous, 0))) is not None
                and not any("_ppo_velocity" in arg for arg in proposal["command"])
            ]
            if evidence:
                _, parent_variant, parent = max(evidence, key=lambda item: (item[0], item[1]))
                options = parent["command"][3:]
            else:
                parent_variant, options = 0, overrides(case, 0)
        else:
            parent_variant = 9
            parent = read_proposal(directory(root, case, 9, 0))
            if parent is None:
                raise RuntimeError("Paired PPO trial requires its recorded control proposal")
            options = parent["command"][3:]
        options = [
            arg
            for arg in options
            if not arg.startswith(("experiment=", "seed=", "output=")) and arg != "mode=train"
        ]
        label = "validation" if variant == 9 else "stratified"
        options.insert(0, f"experiment=navigation_procedural_{case.task}_ppo_{label}")
        return options, parent_variant

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
    # Never leak fixed benchmark geometry into the procedural training pool.
    # The legacy sparse LiDAR preset belongs exclusively to fixed-map runs.
    if case.task == "lidar" and "procedural" not in case.experiment:
        sparse = (
            "navigation_lidar_scratch_ppo_asymmetric_sparse"
            if case.algorithm == "ppo"
            else "navigation_lidar_scratch_sparse"
        )
        options.append(f"experiment={sparse}")

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
                *(
                    [f"learning.scene_updates={rng.choice((20, 40, 80))}"]
                    if "procedural" not in case.experiment
                    else []
                ),
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


def trial_status(path, running):
    """Trust the real run record and detect interrupted/vanished processes."""
    report = read_json(path / "run.json")
    active = str(path) in running
    # A pre-v3 seed with an independently frozen *first passing checkpoint*
    # may be adjudicated once without rewriting the historical ZIP run.
    audit_dir = path.parent / ".scheduler" / "frozen_audits" / path.name
    audit = read_json(audit_dir / "audit.json")
    if audit is not None:
        if audit.get("status") == "running":
            return "auditing"
        if audit.get("status") == "failed":
            return "frozen_benchmark_failed"
        if audit.get("status") == "accepted":
            original_checkpoint = path / audit.get("checkpoint", "missing")
            validation = read_json(original_checkpoint.parent / "report.json")
            frozen = read_json(audit_dir / "report.json")
            try:
                configured = (path / "config.yaml").read_text()
            except OSError:
                configured = ""
            if (
                original_checkpoint.is_file()
                and re.search(r"^initial_checkpoint:\s*null\s*$", configured, re.MULTILINE)
                and validation is not None
                and validation.get("mode") == "checkpoint_eval"
                and validation.get("complete") is True
                and validation.get("passed") is True
                and 1_000_000 <= validation.get("seed_base", 0) < 2_000_000
                and frozen is not None
                and frozen.get("mode") == "benchmark"
                and frozen.get("complete") is True
                and frozen.get("passed") is True
                and frozen.get("seed_base", 0) >= 2_000_000
            ):
                return "accepted"
            return "invalid"
    if report is not None:
        status = report.get("status", "unknown")
        # Old runs remain immutable evidence; do not extend the obsolete
        # three-consecutive-validations protocol or pre-Patience v3 recipes.
        # A fully accepted legacy seed (or verified one-time frozen audit) is
        # still valid and is checked before this retirement rule.
        try:
            configured = (path / "config.yaml").read_text()
        except OSError:
            configured = ""
        legacy_recipe = (
            report.get("layout_version") == 2
            or re.search(
                r"^\s+selection:\s*last_of_first_three_consecutive_passes\s*$",
                configured,
                re.MULTILINE,
            )
            is not None
        )
        pre_patience_recipe = (
            not legacy_recipe
            and re.search(r"^early_stopping:\s*$", configured, re.MULTILINE) is None
        )
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
        if not active and status in {"interrupted", "running"}:
            if legacy_recipe:
                return "legacy_retired"
            if pre_patience_recipe:
                return "pre_patience_retired"
        if status == "interrupted":
            proposal = read_proposal(path)
            restarts = read_json(path / "restarts.json") or {}
            archive = latest_training_checkpoint(path)
            if proposal is not None and archive is not None and restarts.get("attempts", 0) < 3:
                return "recoverable"
            return status
        if status in {"failed", "frozen_benchmark_failed", "early_stopped"}:
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
    manifest = read_json(path / "run.json") or {}
    recorded_best = manifest.get("best_screen_score")
    if isinstance(recorded_best, (float, int)) and math.isfinite(recorded_best):
        values.append(float(recorded_best))
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
            average = sum(rates) / len(rates)
            if (
                report_path.parent.name == "screening"
                and report.get("complete")
                and all(
                    item.get("task") == "navigation" and item.get("episodes") == 8
                    for item in scene_reports.values()
                )
            ):
                # Use the same score that drives navigation Patience. Historical
                # best screens survive rolling checkpoint retention in run.json.
                values.append(min(rates) + 0.25 * average)
            else:
                values.append(average)
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
            "legacy_retired",
            "pre_patience_retired",
            "frozen_benchmark_failed",
            "early_stopped",
            "invalid",
            "stale",
        ):
            continue
        if previous[0]["status"] in ("missing", "running", "recoverable", "auditing"):
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
            in (
                "failed",
                "interrupted",
                "legacy_retired",
                "pre_patience_retired",
                "frozen_benchmark_failed",
                "invalid",
                "stale",
            )
            or item["status"] == "early_stopped"
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
    # Training hyperparameters stay identical to the accepted earlier seeds,
    # but a NEW independently initialized seed uses the current checkpoint
    # selection/stopping protocol. Original ZIP runs retain their legacy
    # contract only when actually resuming the *same* original run.
    suggested = [
        override
        for override in suggested
        if not override.startswith(("learning.selection=", "learning.required_consecutive_passes="))
    ]
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
                "selection": (
                    "last_of_first_three_consecutive_passes"
                    if "learning.selection=last_of_first_three_consecutive_passes" in overrides_list
                    else "first_full_validation_pass"
                ),
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
    config_file = path / "config.yaml"
    saved = config_file.read_text()
    match = re.search(r"^\s+selection:\s*(\S+)\s*$", saved, re.MULTILINE)
    if (
        not archive.is_dir()
        or match is None
        or match.group(1) != "first_full_validation_pass"
        or re.search(r"^early_stopping:\s*$", saved, re.MULTILINE) is None
    ):
        raise ValueError(f"Refusing obsolete or incomplete training protocol: {path}")
    runner[runner.index("resume=null")] = f"resume={archive}"
    # Resume the SAME randomly initialized run, including its optimizer, RNG
    # and recurrent history. Never import parameters from another experiment.
    # No new per-trial update budget: progress-based Patience owns termination.
    # Keep historical budget records for audit only; do not extend them.
    restart_file = path / "restarts.json"
    state = read_json(restart_file) or {"attempts": 0}
    state["attempts"] += 1
    atomic_json(restart_file, state)
    name = f"dp-scr-{case.task[:2]}-{case.algorithm}-{seed}-{variant}-resume"
    tmux_launch(project, path, runner, name, append=True)
    return name


def report(root, running, candidates, regime="fixed", only=()):
    """Return one compact matrix snapshot; stage logs stay inside each trial."""
    rows = []
    for case in cases(regime, only):
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
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "source": "random_initialization_only",
        "regime": regime,
        "selected_families": sorted(only) if only else [],
        "matrix_cells": len(cells),
        "train_families": len(rows),
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


def supervise(root, max_parallel, candidates, regime="fixed", only=()):
    """Observe measured outcomes and admit new trials while GPU capacity exists."""
    project = Path(__file__).resolve().parents[1]
    python = Path(
        os.environ.get("DRONE_MATRIX_PYTHON", str(project / ".pixi/envs/default/bin/python"))
    )
    if not python.is_file():
        raise RuntimeError(f"Python environment not installed: {python}")
    root.mkdir(parents=True, exist_ok=True)
    previous = read_json(root / "matrix.json")
    if previous is not None and previous.get("regime", "fixed") != regime:
        raise ValueError("Refusing to mix different training regimes in one results directory")
    if previous is not None and previous.get("selected_families", []) != (
        sorted(only) if only else []
    ):
        raise ValueError("Refusing to change the selected families in an existing queue")
    n_families = len(list(cases(regime, only)))
    while True:
        running = active_runs(root)
        snapshot = report(root, running, candidates, regime=regime, only=only)
        snapshot["resources"] = resources()
        atomic_json(root / "matrix.json", snapshot)
        print(
            f"accepted={snapshot['families_accepted']}/{n_families} "
            f"active={len(running)} "
            f"VRAM_free={snapshot['resources']['gpu_free_mib']}MiB "
            f"GPU={snapshot['resources']['gpu_percent']}%",
            flush=True,
        )
        if snapshot["families_accepted"] + snapshot["families_unresolved"] == n_families:
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
            if regime == "procedural":
                priority = {
                    "depth_ppo": 0,
                    "lidar_ppo": 1,
                    "racing_shac": 2,
                    "racing_ppo": 3,
                    "depth_apg": 4,
                    "lidar_apg": 5,
                    "depth_shac": 6,
                    "lidar_shac": 7,
                }

            # Rotate among unresolved families before spending many trials on
            # a single hard PPO cell. Each family gets an independent seed-0
            # attempt before a second attempt is admitted for another family.
            # This affects admission order only, never checkpoint selection.
            def admission_key(case, priority_map=priority):
                started = sum(
                    directory(root, case, variant, seed).is_dir()
                    for variant in range(candidates)
                    for seed in (0, 1, 2)
                )
                return started, priority_map.get(case.key, 7)

            for case in sorted(cases(regime, only), key=admission_key):
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
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--regime", choices=("fixed", "procedural"), default="fixed")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated logical train families, e.g. depth_ppo,lidar_ppo,racing_shac",
    )
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    only = tuple(x.strip() for x in args.only.split(",") if x.strip())
    if len(only) != len(set(only)):
        parser.error("--only cannot contain duplicate families")
    if args.python is not None:
        os.environ["DRONE_MATRIX_PYTHON"] = str(args.python.resolve())
    if args.mode == "plan":
        for case in cases(args.regime, only):
            print(case.key, overrides(case, 0), "seeds=0,1,2")
    elif args.mode == "status":
        print(
            json.dumps(
                report(root, active_runs(root), args.candidates, args.regime, only), indent=2
            )
        )
    else:
        if args.max_parallel < 1 or not 1 <= args.candidates <= 256:
            parser.error("Invalid concurrency or number of candidates")
        supervise(root, args.max_parallel, args.candidates, args.regime, only)


if __name__ == "__main__":
    main()
