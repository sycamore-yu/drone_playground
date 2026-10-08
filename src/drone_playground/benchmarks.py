"""Load and verify the exact external conditions of a named benchmark."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from drone_playground.resources import resource_path

NAVIGATION_BENCHMARK = "benchmarks/navigation.yaml"


def benchmark_id(config):
    """Resolve the benchmark protocol identifier from experiment settings."""
    evaluation = config.get("evaluation", config)
    if "release_validation" in evaluation:
        raise ValueError("release_validation was removed; use benchmark_id")
    return evaluation.get("benchmark_id")


def load_protocol(path=NAVIGATION_BENCHMARK):
    """Load and validate the named benchmark protocol description."""
    specification = yaml.safe_load(_path(path).read_bytes())
    if specification.get("name") != "navigation" or specification.get("version") != 2:
        raise ValueError("Unsupported benchmark protocol version")
    scenes = [scene for group in specification["scenes"].values() for scene in group]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("Protocol scenes must be nonempty and unique")
    if not set(specification["primary_scenes"]) <= set(scenes):
        raise ValueError("Primary scenes must be declared protocol scenes")
    if any(not isinstance(n, int) or n < 1 for n in specification["episodes_per_scene"].values()):
        raise ValueError("Protocol episode counts must be positive integers")
    return specification


def protocol_scenes(protocol):
    """Extract the scene identifiers required by a benchmark protocol."""
    return [scene for group in protocol["scenes"].values() for scene in group]


def resolve_protocol_settings(config):
    """Fill omitted phase settings from the selected, recorded protocol."""
    evaluation = config.get("evaluation", {})
    path = evaluation.get("protocol")
    if not path:
        return
    protocol = load_protocol(path)
    if evaluation.get("episodes") is None:
        evaluation["episodes"] = protocol["episodes_per_scene"]["benchmark"]
    if evaluation.get("seed_start") is None:
        evaluation["seed_start"] = protocol["seeds"]["benchmark"]
    if "initial_conditions" in evaluation and evaluation["initial_conditions"] is None:
        evaluation["initial_conditions"] = protocol["initial_conditions"]
    training = config.get("training", {})
    for field, value in (
        (
            "checkpoint_eval_episodes",
            protocol["episodes_per_scene"]["checkpoint_eval"],
        ),
        ("checkpoint_eval_seed_start", protocol["seeds"]["checkpoint_eval"]),
        ("checkpoint_eval_initial_conditions", protocol["initial_conditions"]),
    ):
        if field in training and training[field] is None:
            training[field] = value


def apply_quality(report, config):
    """Project policy consumes measurements; evaluators never decide acceptance."""
    config = config.get("components", config)
    rules = config.get("evaluation", {}).get("acceptance")
    if not rules:
        report.update(quality_passed=None, quality_rule=None)
        return
    if "success_rate" in report:
        rates = report.get("scene_success_rates", {})
        if not rates and "cells" in report:
            rates = {
                scene: rate
                for cell in report["cells"].values()
                for scene, rate in cell.get("scene_success_rates", {}).items()
            }
        passed = report["success_rate"] >= rules["navigation_success_rate"]
        if rates:
            passed = passed and min(rates.values()) >= rules["navigation_scene_success_rate"]
    else:
        passed = report["completion_rate"] >= rules["completion_rate"]
        error = report.get("rmse_completed_mean")
        if "rmse_completed_mean" in report:
            passed = passed and error is not None and error <= rules["tracking_rmse_m"]
    report.update(quality_passed=bool(passed), quality_rule=dict(rules))


def navigation_checkpoint_eval_selection(
    report,
    criterion="navigation-checkpoint_eval-v2",
    *,
    protocol=None,
    episodes_per_scene=None,
):
    """Apply the declared selection rule to all independently reset cases."""
    specification = load_protocol() if protocol is None else protocol
    try:
        rule = specification["success_threshold"][criterion]
    except KeyError as exc:
        raise ValueError("Unknown navigation checkpoint_eval criterion") from exc
    scenes = protocol_scenes(specification)
    repeats = (
        specification["episodes_per_scene"]["checkpoint_eval"]
        if episodes_per_scene is None
        else episodes_per_scene
    )
    rows = report["episodes"]
    count = len(scenes) * repeats
    seeds = [row["seed"] for row in rows]
    reset = report.get("initial_conditions")
    if (
        repeats < 1
        or report["num_trials"] != count
        or len(rows) != count
        or Counter(row["scene_id"] for row in rows) != dict.fromkeys(scenes, repeats)
        or len(set(seeds)) != count
        or not reset
        or reset["seeds"] != seeds
    ):
        raise ValueError(
            "Checkpoint evaluation must contain every configured scene/reset exactly once"
        )
    successes = {
        scene: sum(bool(row["arrived"]) for row in rows if row["scene_id"] == scene)
        for scene in scenes
    }
    selected = specification["primary_scenes"] if rule["scope"] == "primary" else scenes
    worst = min(successes[scene] for scene in selected)
    total = sum(successes[scene] for scene in selected)
    selected_count = repeats * len(selected)
    # One worst-scene success outweighs every possible aggregate improvement.
    weight = selected_count + 1
    result = dict(
        score=[worst, total],
        pilot_objective=(weight * worst + total) / (weight * repeats + selected_count),
        worst_scene_success_rate=worst / repeats,
        overall_success_rate=sum(successes.values()) / count,
        selection_rule=criterion,
    )
    if rule["scope"] == "primary":
        result.update(
            primary_success_rate=total / selected_count,
            primary_checkpoint_eval_passed=worst / repeats >= rule["per_scene"],
        )
    return result


def validate_navigation_report(
    report,
    minimum_per_task=None,
    tasks=None,
    criterion="navigation-v1",
    *,
    protocol=None,
):
    """Apply project acceptance after checking the complete measurement evidence."""
    specification = load_protocol() if protocol is None else protocol
    try:
        rule = specification["success_threshold"][criterion]
    except KeyError as exc:
        raise ValueError("Unknown navigation benchmark criterion") from exc
    if (
        report.get("role") != "eval"
        or not report.get("parameters_frozen")
        or not report.get("initial_conditions")
    ):
        raise ValueError(
            "Navigation benchmark needs frozen parameters and evaluation initial conditions"
        )
    tasks = tuple(specification["scenes"]) if tasks is None else tasks
    if not tasks or not set(tasks) <= set(specification["scenes"]):
        raise ValueError("Select declared navigation tasks")
    expected = {scene for task in tasks for scene in specification["scenes"][task]}
    rows = report["episodes"]
    if (
        len(rows) != report["num_trials"]
        or {row["scene_id"] for row in rows} != expected
        or len({row["seed"] for row in rows}) != len(rows)
    ):
        raise ValueError("Missing navigation scenes/episodes or repeated evaluation seeds")
    reset = report["initial_conditions"]
    if reset["seeds"] != [row["seed"] for row in rows] or len(reset["position_m"]) != len(rows):
        raise ValueError("Initial-condition manifest and episodes differ")
    actual = np.asarray([row["initial_position_m"] for row in rows])
    if (
        not np.isfinite(actual).all()
        or not np.allclose(actual, reset["position_m"], rtol=0, atol=1e-6)
        or len(np.unique(actual, axis=0)) != len(rows)
    ):
        raise ValueError("Recorded initial positions differ from the manifest or repeat")
    results = {}
    for task in tasks:
        scenes = specification["scenes"][task]
        cases = [row for row in rows if row["scene_id"] in scenes]
        minimum = (
            specification["episodes_per_scene"]["benchmark"] * len(scenes)
            if minimum_per_task is None
            else minimum_per_task
        )
        if len(cases) < minimum:
            raise ValueError(
                f"Navigation task {task} requires at least {minimum} benchmark episodes"
            )
        for row in cases:
            if row["outcome"] not in (
                "arrived",
                "collision",
                "out_of_bounds",
                "numerical_failure",
                "timeout",
            ):
                raise ValueError("Unknown navigation outcome")
            if bool(row["arrived"]) != (row["outcome"] == "arrived"):
                raise ValueError("Inconsistent navigation outcome")
        successes = sum(row["arrived"] for row in cases)
        rates = {
            scene: sum(row["arrived"] for row in cases if row["scene_id"] == scene)
            / sum(row["scene_id"] == scene for row in cases)
            for scene in scenes
        }
        result = dict(
            num_trials=len(cases),
            arrived=successes,
            success_rate=successes / len(cases),
            scene_success_rates=rates,
        )
        if rule["scope"] == "primary":
            primary = {
                scene: rate
                for scene, rate in rates.items()
                if scene in specification["primary_scenes"]
            }
            primary_cases = [row for row in cases if row["scene_id"] in primary]
            result.update(
                primary_scene_success_rates=primary,
                extension_scene_success_rates={s: r for s, r in rates.items() if s not in primary},
                primary_num_trials=len(primary_cases),
                extension_num_trials=len(cases) - len(primary_cases),
                primary_success_rate=sum(row["arrived"] for row in primary_cases)
                / len(primary_cases),
                passed=bool(primary)
                and all(rate >= rule["per_scene"] for rate in primary.values()),
            )
        else:
            result["passed"] = result["success_rate"] >= rule["aggregate"]
        results[task] = result
    return dict(
        protocol="benchmark-" + criterion,
        tasks=results,
        passed=all(result["passed"] for result in results.values()),
        parameter_sha256=report["parameter_sha256"],
    )


def _path(value):
    value = Path(value)
    return value if value.is_absolute() else resource_path(str(value))


def protocol_identity(config):
    """Derive the immutable identity of the selected benchmark protocol."""
    selected = config.get("evaluation", {}).get("protocol")
    if not selected:
        return None
    path = _path(selected)
    specification = load_protocol(path)
    task, scene = config["env"]["task"], config["env"]["scene"]
    target = config["env"]["dynamics"]["_target_"]
    condition = specification.get("execution_conditions", {}).get(target, {})
    if (
        "control_frequency_hz" in condition
        and config["env"]["freq"] != condition["control_frequency_hz"]
    ):
        raise ValueError("Benchmark protocol differs on control frequency")
    if (
        "physics_frequency_hz" in condition
        and config["env"].get("physics_freq", 500) != condition["physics_frequency_hz"]
    ):
        raise ValueError("Benchmark protocol differs on physics frequency")
    if task["time_limit_kind"] != specification["timeout"]["kind"]:
        raise ValueError("Benchmark protocol differs on time-limit interpretation")
    if task["name"] != "navigation":
        raise ValueError("Navigation protocol requires a navigation task")
    for field, expected in (
        ("duration", "timeout"),
        ("goal_radius", "goal_radius_m"),
    ):
        value = (
            specification[expected]["duration_s"]
            if expected == "timeout"
            else specification[expected]
        )
        if float(task[field]) != float(value):
            raise ValueError(f"Benchmark protocol differs on task.{field}")
    if "body_radius" in task and float(task["body_radius"]) != specification["body_radius_m"]:
        raise ValueError("Benchmark protocol differs on body collision radius")
    evaluation = config["evaluation"]
    if (
        "duration" in evaluation
        and evaluation["duration"] != specification["timeout"]["duration_s"]
    ):
        raise ValueError("Benchmark protocol differs on evaluation duration")
    if any(
        speed <= 0 or speed > specification["nominal_max_velocity_mps"]
        for speed in evaluation.get("speeds", [])
    ):
        raise ValueError("Benchmark protocol differs on commanded speed bounds")
    limits = config["method"].get("limits")
    if (
        specification["version"] == 2
        and limits is not None
        and limits["max_velocity_mps"] != specification["nominal_max_velocity_mps"]
    ):
        raise ValueError("Benchmark protocol differs on nominal planner velocity limit")
    chosen = scene.get("scene_ids", protocol_scenes(specification))
    if not chosen or not set(chosen) <= set(protocol_scenes(specification)):
        raise ValueError("Benchmark contains an undeclared scene identity")
    return dict(
        name=specification["name"],
        version=specification["version"],
        path=str(selected),
        catalog=scene.get("catalog_path") or specification["catalog"],
    )


def checkpoint_eval_score(task: str, report: dict, rule: str | None = None) -> tuple:
    """Select only on checkpoint_eval data; ties break on the declared secondary terms."""
    if rule == "release-pilot-v1":
        return (min(pilot_scene_rates(task, report).values()),)
    if task == "racing":
        return (
            report["completed"],
            report["gates_passed_mean"],
            -report["rmse_all_mean"],
        )
    if task == "navigation":
        # Declared order: macro success rate, collision rate, constrained time.
        score = (
            report["success_rate"],
            -report["collision_rate"],
            -report["constrained_time_mean_s"],
        )
        if rule is None:
            return score
        if rule not in (
            "navigation-convergence-v1",
            "navigation-convergence-v2",
            "navigation-convergence-v3",
        ):
            raise ValueError(f"Unknown checkpoint_eval selection rule: {rule}")
        rows = [row for cell in report["cells"].values() for row in cell["episodes"]]
        if rule == "navigation-convergence-v3":
            # Once arrival is established, sub-millimetre sampling differences
            # within the goal radius have no task meaning. Compare actual times.
            remaining = float(
                np.mean(
                    [
                        0.0 if row.get("arrived", False) else row["final_goal_distance_m"]
                        for row in rows
                    ]
                )
            )
            return (
                report["success_rate"],
                -report["failure_rate"],
                -remaining,
                -report["constrained_time_mean_s"],
            )
        if rule == "navigation-convergence-v2":
            return (
                report["success_rate"],
                -report["failure_rate"],
                -float(np.mean([row["final_goal_distance_m"] for row in rows])),
                -report["constrained_time_mean_s"],
            )
        return (
            *score,
            -float(np.mean([row["final_goal_distance_m"] for row in rows])),
        )
    return (report["completed"], -report["rmse_all_mean"])


def pilot_scene_rates(task: str, report: dict) -> dict:
    """Frozen pilot objective: worst scenario group, with failures in the denominator."""
    groups = {}
    if task == "navigation":
        for difficulty, cell in report["cells"].items():
            for row in cell["episodes"]:
                key = difficulty + "/" + row["subtype"]
                groups.setdefault(key, []).append(bool(row["arrived"]))
    else:
        for row in report["episodes"]:
            success = bool(row["completed"]) and not row["failed"]
            if task != "racing":
                success = success and np.isfinite(row["rmse_m"]) and row["rmse_m"] <= 0.25
            groups.setdefault(task, []).append(success)
    if not groups or any(not rows for rows in groups.values()):
        raise ValueError("Pilot objective requires nonempty episode groups")
    return {key: float(np.mean(rows)) for key, rows in groups.items()}


def checkpoint_eval_seeds(config: dict, count: int) -> list[int]:
    """Select deterministic evaluation seeds for frozen checkpoint comparison."""
    start = int(config.get("checkpoint_eval_seed_start", 20000))
    if start < 0 or count < 1:
        raise ValueError(
            "CheckpointEval requires a nonnegative seed start and positive episode count"
        )
    return list(range(start, start + count))


def load_specification(task):
    """Load the benchmark specification for the requested task."""
    if task not in ("tracking", "racing"):
        raise ValueError("This benchmark validator covers tracking and racing only")
    path = resource_path(f"benchmarks/{task}.yaml")
    specification = yaml.safe_load(path.read_bytes())
    if specification.get("name") != task or specification.get("version") != 1:
        raise ValueError("Unsupported control benchmark specification")
    return specification


def validate_control_report(report, task, specification=None):
    """Validate control metrics against the selected benchmark acceptance rules."""
    specification = load_specification(task) if specification is None else specification
    if specification.get("name") != task:
        raise ValueError("Benchmark specification and task differ")
    rows = report["episodes"]
    if report.get("role") != "eval" or not report.get("parameters_frozen"):
        raise ValueError("Benchmark evidence requires a frozen policy in eval role")
    native = report.get("parameter_identity_kind") == "resolved optimization configuration"
    if native and not report.get("runtime_identity"):
        raise ValueError("Optimization benchmark requires its actual solver runtime identity")
    minimum_episodes = int(specification["episodes"])
    if len(rows) != report["num_trials"] or len(rows) < minimum_episodes:
        raise ValueError("Insufficient or incomplete benchmark episode evidence")
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(rows):
        raise ValueError("Repeated evaluation seeds are not independent cases")
    if any(bool(row["completed"]) == bool(row["failed"]) for row in rows):
        raise ValueError("Each episode must record either completion or failure")
    if any(not math.isfinite(row["rmse_m"]) or row["rmse_m"] < 0 for row in rows):
        raise ValueError("Every episode must retain finite valid-trajectory RMSE")
    completed = [row for row in rows if row["completed"]]
    threshold = float(specification["success"]["minimum_completion_rate"])
    rate = len(completed) / len(rows)
    rmse = sum(row["rmse_m"] for row in completed) / len(completed) if completed else None
    max_rmse = specification["success"].get("maximum_completed_rmse_m")
    passed = rate >= threshold and (
        max_rmse is None or (rmse is not None and rmse <= float(max_rmse))
    )
    return dict(
        protocol=f"benchmark-{task}-v1",
        task=task,
        passed=passed,
        num_trials=len(rows),
        completed=len(completed),
        failed=len(rows) - len(completed),
        completion_rate=rate,
        minimum_completion_rate=threshold,
        rmse_completed_mean=rmse,
        rmse_all_mean=sum(row["rmse_m"] for row in rows) / len(rows),
        error_rule=(
            f"Completed mean RMSE <= {max_rmse}m; all failures retained"
            if max_rmse is not None
            else "All failures retained"
        ),
        parameter_sha256=report["parameter_sha256"],
        caveat=(
            "Frozen solver; no learning seeds required"
            if native
            else "One frozen policy; the cell additionally requires all 3 training seeds"
        ),
    )
