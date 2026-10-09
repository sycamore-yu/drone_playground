"""Collect C1-C6 evidence from an explicit manifest, without importing simulation."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

ALGORITHMS = ("ppo", "apg", "shac")
CONDITIONS = (
    ("tracking", "state", None),
    ("racing", "state", None),
    ("navigation", "depth", "static"),
    ("navigation", "depth", "dynamic"),
    ("navigation", "lidar", "static"),
    ("navigation", "lidar", "dynamic"),
)
NAV_SCENES = ("S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06")
RULE = "last_of_first_three_consecutive_passes"
EVENTS = {
    "SUCCESS",
    "COLLISION",
    "NUMERICAL_FAILURE",
    "OUT_OF_BOUNDS",
    "TIMEOUT",
    "MISSED_GATE",
    "METHOD_FAILURE",
}
GPU_SCOPE = "Whole GPU samples with concurrent jobs; not isolated process/runtime measurements."


def check(passed, *reasons):
    """Represent a verified, failed or unavailable criterion with its reasons."""
    return {"passed": passed, "reasons": list(reasons)}


def metric(value, source, reason="Not recorded or not finite"):
    """Retain a measured value together with its source and unavailable reason."""
    return {"value": value, "source": source, "reason": reason if value is None else None}


def number(value):
    """Return a finite numeric value without inventing missing measurements."""
    if type(value) is int:
        return value
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def digest(path):
    """Compute the file SHA-256 used for evidence identity."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_mapping(path, issues):
    """Read a JSON or YAML mapping and retain parsing failures as evidence issues."""
    try:
        value = (
            yaml.safe_load(path.read_text())
            if path.suffix == ".yaml"
            else json.loads(path.read_text())
        )
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError, yaml.YAMLError) as error:
        issues.append(f"{path}: {error}")
        return {}


def read_events(path, issues):
    """Read recorded events while retaining malformed-line evidence."""
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        issues.append(f"{path}: {error}")
        return []
    events = []
    for index, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("expected an object")
            events.append(event)
        except ValueError as error:
            issues.append(f"{path}:{index}: {error}")
    return events


def metrics_path(run):
    """Locate recorded logs in current or immutable legacy result packages."""
    run = Path(run)
    return (
        run / "metrics.jsonl" if (run / "metrics.jsonl").is_file() else run / "events/metrics.jsonl"
    )


def resolve_run_path(value, run, root):
    """Resolve version-2 run-local references while retaining historical evidence paths."""
    path = Path(value)
    if not path.is_absolute() and path.parts[0] in {"checkpoints", "eval"}:
        return (run / path).resolve()
    return resolve_path(value, root)


def checkpoint(path):
    """Inspect the persisted archive and payload checksum without loading an Actor."""
    result = {"path": str(path), "archive_sha256": None, "metadata": None}
    try:
        result["archive_sha256"] = digest(path)
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            payload_sha = hashlib.sha256(archive.read("variables.msgpack")).hexdigest()
        result.update(metadata=metadata, payload_sha256=payload_sha)
        result["integrity"] = check(
            metadata.get("sha256") == payload_sha and metadata.get("purpose") == "inference",
            "Compared archive payload SHA-256 and inference purpose",
        )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        result["integrity"] = check(None, str(error))
    return result


def scene_evidence(directory, scene, task, claimed):
    """Recompute quality and denominators from complete, independent CSV episodes."""
    flat = (directory / "episodes.csv").is_file()
    path = directory / "episodes.csv" if flat else directory / scene / "episodes.csv"
    result = {
        "csv": str(path),
        "claimed": claimed,
        "episodes": None,
        "successes": None,
        "outcomes": None,
        "initialization_seeds": None,
        "case_ids": None,
    }
    issues = []
    rows = None
    try:
        with path.open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if not flat or row.get("scene") == scene]
        result["csv_sha256"] = digest(path)
    except (OSError, csv.Error) as error:
        issues.append(str(error))
    rmse = flight = clearance = None
    gate_orders = None
    if rows is not None:
        result["episodes"] = len(rows)
        outcomes = Counter(row.get("event") for row in rows)
        result["outcomes"] = dict(outcomes)
        successes = [row for row in rows if row.get("event") == "SUCCESS"]
        result["successes"] = len(successes)
        if any(row.get("event") not in EVENTS or row.get("scene") != scene for row in rows):
            issues.append("Unknown/nonterminal outcome or wrong scene in CSV")
        cases = []
        try:
            cases = [(int(row["initialization_seed"]), int(row["world_index"])) for row in rows]
            if len(set(cases)) != len(cases):
                issues.append("Duplicate episode initialization_seed/world_index")
            result["initialization_seeds"] = sorted({seed for seed, _ in cases})
            result["case_ids"] = cases
        except (ValueError, KeyError, TypeError):
            issues.append("Missing/invalid episode seed or world_index")
        if task == "tracking" and successes:
            values = [number(row.get("position_rmse")) for row in successes]
            if all(value is not None and value >= 0 for value in values):
                rmse = statistics.mean(values)
            else:
                issues.append("Successful episode RMSE missing/nonfinite/negative")
        if task == "racing":
            gate_orders = dict(Counter(row.get("gate_order", "") for row in rows))
            if any(
                row.get("gate_order") != "1,2,3,4,2" or number(row.get("gates_passed")) != 5
                for row in successes
            ):
                issues.append("SUCCESS without legal gate order 1,2,3,4,2 and five gates")
        times = [number(row.get("flight_seconds")) for row in successes]
        if times and all(value is not None and value > 0 for value in times):
            flight = {"min": min(times), "mean": statistics.mean(times), "max": max(times)}
        elif task == "racing" and successes:
            issues.append("Successful Racing flight time missing/nonfinite/nonpositive")
        clearances = [number(row.get("min_clearance")) for row in rows]
        if clearances and all(value is not None for value in clearances):
            clearance = min(clearances)
        for key in ("episodes", "successes"):
            if key in claimed and claimed[key] != result[key]:
                issues.append(f"report.json {key} disagrees with CSV")
        if (
            claimed.get("successful_position_rmse") is not None
            and rmse is not None
            and not math.isclose(claimed["successful_position_rmse"], rmse, rel_tol=1e-6)
        ):
            issues.append("report.json RMSE disagrees with CSV")
    expected, minimum = (
        (25, 23) if task == "navigation" else (100, 95 if task == "tracking" else 90)
    )
    count = result["episodes"]
    if count is None:
        quality = None
    elif count != expected:
        quality = False if count > expected else None
        issues.append(f"Requires exactly {expected} episodes; CSV has {count}")
    else:
        quality = result["successes"] >= minimum and (
            task != "tracking" or (rmse is not None and rmse <= 0.25)
        )
    if issues and quality is True:
        quality = False
    result.update(
        quality=check(quality, *issues),
        success_rate=metric(result["successes"] / count if count else None, str(path)),
        successful_position_rmse=metric(rmse, str(path), "No successful Tracking RMSE available"),
        successful_flight_seconds=metric(flight, str(path), "No complete successful flight times"),
        min_clearance=metric(
            clearance, str(path), "Clearance absent/nonfinite in one or more rows"
        ),
        gate_orders=metric(gate_orders, str(path), "Not a Racing evaluation or CSV unavailable"),
    )
    return result


def combine(checks):
    """Combine criteria without treating unavailable evidence as a pass."""
    values = [entry["passed"] for entry in checks]
    return check(False if False in values else (None if None in values or not values else True))


def evaluate(directory, task, seed, mode):
    """Verify saved episode CSVs, task quality and evaluation seed partitions."""
    issues = []
    report = read_mapping(directory / "report.json", issues)
    scenes = NAV_SCENES if task == "navigation" else ("empty" if task == "tracking" else "racing",)
    evidence = {
        scene: scene_evidence(directory, scene, task, report.get("reports", {}).get(scene, {}))
        for scene in scenes
    }
    if report.get("mode") != mode or report.get("training_seed") != seed:
        issues.append("Evaluation mode/training_seed missing or mismatched")
    base = number(report.get("seed_base"))
    index = report.get("evaluation_index")
    if base is None or not (
        1_000_000 <= base < 2_000_000 if mode == "checkpoint_eval" else base >= 2_000_000
    ):
        issues.append("Report seed_base outside independent evaluation partition")
    seeds = set()
    for scene_index, entry in enumerate(evidence.values()):
        actual = entry["initialization_seeds"]
        if actual is None or not actual:
            issues.append(f"{scenes[scene_index]}: no verified CSV initialization seeds")
            continue
        seeds.update(actual)
        if any(
            not (
                1_000_000 <= value < 2_000_000 if mode == "checkpoint_eval" else value >= 2_000_000
            )
            for value in actual
        ):
            issues.append(f"{scenes[scene_index]}: actual CSV seeds outside {mode} partition")
        if base is not None and isinstance(index, int):
            if actual != [int(base + 10_000 * seed + 100 * index + scene_index)]:
                issues.append(f"{scenes[scene_index]}: actual CSV seeds disagree with report")
        else:
            issues.append("Missing evaluation_index/seed_base")
    primary = scenes[:6] if task == "navigation" else scenes
    return {
        "directory": str(directory),
        "evaluation_index": index,
        "reported_passed": report.get("passed"),
        "task_contract": report.get("task_contract"),
        "scenes": evidence,
        "quality": combine([evidence[s]["quality"] for s in primary]),
        "partition": check(not issues, *issues),
        "seeds": sorted(seeds),
    }


def cost_evidence(header, events, source):
    """Attribute actual counters, elapsed time and GPU samples to their records."""
    updates = [event for event in events if event.get("event") == "update"]
    last = updates[-1] if updates else {}
    result = {}
    for key in (
        "updates",
        "interactions",
        "wall_seconds",
        "training_wall_seconds",
        "session_interactions",
        "session_wall_seconds",
        "interactions_per_second",
    ):
        value = number(header.get(key))
        origin = f"{source}/run.json"
        if value is None:
            value = number(last.get("update" if key == "updates" else key))
            origin = f"{metrics_path(source)}:last update (lower bound while running)"
        result[key] = metric(value, origin)
    result["logged_update_events"] = len(updates)
    result["last_logged_update"] = last or None
    for key in ("gpu_memory_mib", "gpu_utilization_percent"):
        samples = [number(event.get(key)) for event in updates]
        samples = [value for value in samples if value is not None]
        result[key] = metric(
            {
                "samples": len(samples),
                "min": min(samples),
                "mean": statistics.mean(samples),
                "max": max(samples),
            }
            if samples
            else None,
            str(metrics_path(source)),
        )
    result["gpu_scope"] = GPU_SCOPE
    result["isolated_gpu_seconds"] = metric(None, None, GPU_SCOPE)
    return result


def resolve_path(value, root):
    """Resolve a recorded path against the chosen evidence root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def initialization_cost_id(path):
    """Identify the source run or reference archive for initialization cost."""
    for parent in path.parents:
        if parent.name == "checkpoints":
            return str(parent.parent)
    if (path.parent / "config.yaml").is_file() or (path.parent / "run.json").is_file():
        return str(path.parent)
    return str(path)


def collect_run(path, root):
    """Collect one selected run without replacing failures from another run."""
    issues = []
    config = read_mapping(path / "config.yaml", issues)
    header = read_mapping(path / "run.json", issues)
    events = read_events(metrics_path(path), issues)
    current_layout = header.get("layout_version") == 2
    task, sensor = config.get("task", {}).get("name"), config.get("sensor", {}).get("name")
    algorithm, seed = config.get("learning", {}).get("algorithm"), config.get("seed")
    result = {
        "path": str(path),
        "exists": path.exists(),
        "task": task,
        "sensor": sensor,
        "algorithm": algorithm,
        "seed": seed,
        "recorded_status": header.get("status"),
        "config": config or None,
        "run_record": header or None,
        "issues": issues,
        "cost": cost_evidence(header, events, path),
    }
    if (task, sensor) not in {(t, s) for t, s, _ in CONDITIONS} or algorithm not in ALGORITHMS:
        result["identity"] = check(None, "Missing or unsupported task/sensor/algorithm")
        return result
    if type(seed) is not int or seed < 0:
        raise ValueError(f"{path}: invalid training seed")
    result["identity"] = check(True)
    result["config_integrity"] = (
        check(
            digest(path / "config.yaml") == header.get("config_sha256"),
            "Compared saved config bytes with run.json config_sha256",
        )
        if header.get("config_sha256")
        else check(None, "run.json config_sha256 unavailable")
    )
    selection = (
        header.get("selection", {})
        if current_layout
        else read_mapping(path / "selection.json", issues)
    )
    result["selection"] = selection or None
    cp = (
        checkpoint(resolve_run_path(selection["checkpoint"], path, root))
        if selection.get("checkpoint")
        else {
            "path": None,
            "archive_sha256": None,
            "metadata": None,
            "integrity": check(None, "No selected checkpoint"),
        }
    )
    result["checkpoint"] = cp
    metadata = cp.get("metadata") or {}
    saved = metadata.get("config", {}).get("experiment")
    kind = config.get("method", {}).get("actor", {}).get("kind", sensor)
    cp_config = isinstance(saved, dict) and (
        {**saved, "resume": None} == {**config, "resume": None} and metadata.get("kind") == kind
    )
    result["checkpoint_config"] = check(
        cp_config if saved else None,
        "Compared checkpoint experiment/kind to config.yaml; resume is an execution entrypoint",
    )
    folder = path / ("checkpoints" if current_layout else "checkpoint_eval")
    prefix = "step-" if current_layout else "update-"
    directories = {
        int(p.name.removeprefix(prefix)): p
        for p in folder.glob(f"{prefix}*")
        if p.is_dir()
        and p.name.removeprefix(prefix).isdigit()
        and (not current_layout or (p / "report.json").is_file())
    }
    for event in events:
        if event.get("event") == "checkpoint_evaluation" and type(event.get("update")) is int:
            update = event["update"]
            name = f"step-{update:06d}" if current_layout else f"update-{update:08d}"
            directories.setdefault(update, folder / name)
    evaluations = []
    for update, directory in sorted(directories.items()):
        entry = evaluate(directory, task, seed, "checkpoint_eval")
        entry["update"] = update
        evaluations.append(entry)
    result["checkpoint_evaluations"] = evaluations
    benchmark = evaluate(
        resolve_run_path(header["benchmark_directory"], path, root)
        if header.get("benchmark_directory")
        else path / "benchmark",
        task,
        seed,
        "benchmark",
    )
    result["benchmark"] = benchmark
    streak, first, history_complete = [], None, True
    previous_index = 0
    interval = config.get("learning", {}).get("evaluation_interval")
    previous_update = 0
    for entry in evaluations:
        consecutive = entry["evaluation_index"] == previous_index + 1 and (
            interval is None or entry["update"] == previous_update + interval
        )
        if not consecutive:
            streak = []
            history_complete = False
        if entry["quality"]["passed"] is True and entry["partition"]["passed"] is True:
            streak.append(entry["update"])
        else:
            streak = []
        previous_index = entry["evaluation_index"] or 0
        previous_update = entry["update"]
        if len(streak) == 3 and first is None:
            first = list(streak)
    independent = (
        not (set(benchmark["seeds"]) & {value for entry in evaluations for value in entry["seeds"]})
        if benchmark["seeds"] and evaluations and all(e["seeds"] for e in evaluations)
        else None
    )
    c5_ok = (
        history_complete
        and first is not None
        and selection.get("update") == first[-1]
        and selection.get("rule") == RULE
        and config.get("learning", {}).get("selection") == RULE
        and config.get("learning", {}).get("required_consecutive_passes") == 3
        and selection.get("consecutive_passes") == 3
        and selection.get("evaluation_index")
        == next(
            (e["evaluation_index"] for e in evaluations if e["update"] == selection.get("update")),
            None,
        )
        and metadata.get("provenance", {}).get("updates") == selection.get("update")
        and header.get("selected_checkpoint") is not None
        and resolve_run_path(header["selected_checkpoint"], path, root) == Path(cp["path"])
        and independent
        and benchmark["partition"]["passed"] is True
    )
    result["C5"] = check(
        c5_ok if selection else None,
        "Verified first consecutive triple, selection, archive update and CSV partitions"
        if c5_ok
        else "Missing/invalid consecutive evaluations, selection or holdout",
    )
    result["C5"].update(first_passing_updates=first, independent_seed_sets=independent)
    target_event = next(
        (
            event
            for event in events
            if event.get("event") == "checkpoint_evaluation"
            and event.get("update") == selection.get("update")
        ),
        {},
    )
    result["cost_to_selection"] = {
        "wall_seconds": metric(
            number(target_event.get("wall_seconds")),
            f"{metrics_path(path)}:selected checkpoint evaluation",
            "Selected checkpoint evaluation elapsed time not recorded",
        ),
        **{
            key: metric(
                number(metadata.get("provenance", {}).get(key)), f"{cp['path']}:metadata.provenance"
            )
            for key in ("updates", "interactions")
        },
        "scope": "Recorded session elapsed time through the selected evaluation; not GPU time. "
        "A resumed session does not establish total historical time to target.",
    }
    initial = metadata.get("config", {}).get("initialization") or next(
        (event for event in events if event.get("event") == "initialized"), None
    )
    result["initialization"] = {
        "checkpoint": config.get("initial_checkpoint"),
        "record": initial,
        "shared_cost_id": None,
    }
    if config.get("initial_checkpoint"):
        initial_cp = checkpoint(resolve_path(config["initial_checkpoint"], root))
        result["initialization"]["evidence"] = initial_cp
        result["initialization"]["shared_cost_id"] = initialization_cost_id(
            Path(initial_cp["path"])
        )
        result["initialization"]["integrity"] = check(
            bool(initial)
            and initial.get("sha256") == initial_cp.get("payload_sha256")
            and initial_cp["integrity"]["passed"] is True,
            "Initialization sha256 is the payload checksum, not the archive checksum",
        )
    else:
        result["initialization"]["integrity"] = check(True, "Fresh initialization in saved config")
    cost = result["cost"]
    real_updates = cost["updates"]["value"]
    result["actual_updates"] = check(
        None if real_updates is None or not cost["logged_update_events"] else real_updates > 0,
        "Requires positive recorded counter and actual update events",
    )
    simulation = config.get("simulation", {})
    expected = {"dynamics": "first_principles", "drone": "cf21B_500", "physics_hz": 500}
    recorded = {key: simulation.get(key) for key in expected}
    duration = config.get("task", {}).get("duration")
    result["physics_config"] = check(
        None
        if any(value is None for value in recorded.values()) or duration is None
        else (
            recorded == expected
            and duration == {"tracking": 20, "racing": 60, "navigation": 300}[task]
        ),
        "Saved Crazyflow dynamics/drone/physics rate and task duration versus chosen base contract",
    )
    needed = (
        "updates",
        "interactions",
        "wall_seconds",
        "training_wall_seconds",
        "interactions_per_second",
        "gpu_memory_mib",
        "gpu_utilization_percent",
    )
    result["C6"] = check(
        True
        if all(cost[key]["value"] is not None for key in needed)
        and any("cuda" in device or "gpu" in device for device in header.get("devices", []))
        else None,
        GPU_SCOPE,
    )
    return result


def collect_initialization_costs(runs, root):
    """Follow explicit initial checkpoints, counting each source run once."""
    shared = {}
    for run in runs:
        initial = run.get("initialization")
        if initial is None:
            continue
        lineage = initial["cost_lineage"] = []
        while initial.get("shared_cost_id"):
            source = initial["shared_cost_id"]
            if source in lineage:
                shared[source]["issues"].append(f"Cyclic initialization lineage for {run['path']}")
                break
            lineage.append(source)
            if source not in shared:
                issues = []
                reference = source == initial["evidence"]["path"]
                config = {} if reference else read_mapping(Path(source) / "config.yaml", issues)
                header = {} if reference else read_mapping(Path(source) / "run.json", issues)
                events = [] if reference else read_events(metrics_path(source), issues)
                cost = cost_evidence(header, events, source)
                if reference:
                    for value in cost.values():
                        if isinstance(value, dict) and "value" in value:
                            value.update(
                                source=f"{source}:metadata.provenance",
                                reason="Historical training cost unavailable: no source run "
                                "cost records linked by this reference archive.",
                            )
                parent = {"checkpoint": config.get("initial_checkpoint"), "shared_cost_id": None}
                if parent["checkpoint"]:
                    path = resolve_path(parent["checkpoint"], root)
                    parent.update(
                        shared_cost_id=initialization_cost_id(path), evidence=checkpoint(path)
                    )
                shared[source] = {
                    "kind": "reference_archive" if reference else "training_run",
                    "config": config or None,
                    "run_record": header,
                    "cost": cost,
                    "initialization": parent,
                    "reference_archive": initial["evidence"] if reference else None,
                    "used_by": [],
                    "issues": issues,
                    "attribution": (
                        "Historical training cost unavailable; never treated as zero. "
                        "Archive provenance records transferred weights, not training from scratch."
                        if reference
                        else "Shared initialization training cost counted once; not a formal seed. "
                        "Full source run cost, not an invented prefix cost."
                    ),
                }
            shared[source]["used_by"].append(run["path"])
            initial = shared[source]["initialization"]
    return shared


def recipe(config):
    """Select configuration fields that define the three-seed training recipe."""
    return {
        key: value
        for key, value in config.items()
        if key not in {"seed", "output", "checkpoint", "resume", "replay_path", "mode"}
    }


def common_actor(runs):
    """Compare saved Actor kind and input/control configuration across algorithms."""
    signatures = []
    for run in runs:
        config = run["config"]
        signatures.append(
            {
                "task": config.get("task"),
                "sensor": config.get("sensor"),
                "method": config.get("method"),
                "simulation": {
                    key: value
                    for key, value in config.get("simulation", {}).items()
                    if key not in {"num_envs", "device"}
                },
            }
        )
    kinds = [(run["checkpoint"].get("metadata") or {}).get("kind") for run in runs]
    complete = len(runs) == 9 and all(kinds)
    equal = bool(signatures) and all(s == signatures[0] for s in signatures)
    equal = equal and len({kind for kind in kinds if kind is not None}) <= 1
    return {
        **check(
            False if signatures and not equal else (True if complete else None),
            "Saved Actor kind/task/sensor/control fields; historical implementation not replayed",
        ),
        "signatures": signatures,
        "checkpoint_actor_kinds": kinds,
        "source_identities": sorted(
            {
                (run["run_record"] or {}).get("source_sha256")
                for run in runs
                if (run["run_record"] or {}).get("source_sha256")
            }
        ),
        "available_runs": len(runs),
    }


def seed_result(run, condition):
    """Grade one selected seed using quality, selection and provenance evidence."""
    if run is None:
        return {"status": "missing", "run": None, "reason": "No chosen run for this seed"}
    task, _, partition = condition
    scenes = (
        (NAV_SCENES[:3] if partition == "static" else NAV_SCENES[3:6])
        if task == "navigation"
        else ("empty" if task == "tracking" else "racing",)
    )
    quality = combine([run["benchmark"]["scenes"][scene]["quality"] for scene in scenes])
    criteria = [
        quality,
        run["C5"],
        run["C6"],
        run["config_integrity"],
        run["checkpoint"]["integrity"],
        run["checkpoint_config"],
        run["actual_updates"],
        run["initialization"]["integrity"],
        run["physics_config"],
    ]
    passed = combine(criteria)["passed"]
    terminal_failure = run["recorded_status"] in {"failed", "error"}
    status = (
        "failed"
        if terminal_failure or (passed is False and run["recorded_status"] != "running")
        else (
            "passed"
            if passed is True and run["recorded_status"] in {"accepted", "frozen_benchmark_failed"}
            else "incomplete"
        )
    )
    return {
        "status": status,
        "run": run["path"],
        "quality": quality,
        "C5": run["C5"],
        "C6": run["C6"],
        "scenes": list(scenes),
        "updates": run["cost"]["updates"],
        "interactions": run["cost"]["interactions"],
        "wall_seconds": run["cost"]["wall_seconds"],
    }


def collect(manifest_path):
    """Return all 18 cells; only explicit paths enter the selected cohort."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object with an explicit runs list")
    paths, seeds = manifest.get("runs"), manifest.get("seeds", [0, 1, 2])
    if not isinstance(paths, list) or not all(isinstance(p, str) and p for p in paths):
        raise ValueError("manifest.runs must be an explicit list of nonempty run paths")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != 3
    ):
        raise ValueError("manifest.seeds must contain three distinct nonnegative training seeds")
    root = resolve_path(manifest.get("root", "."), manifest_path.parent)
    paths = [resolve_path(path, root) for path in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate run path in manifest")
    runs = [collect_run(path, root) for path in paths]
    indexed = {}
    for run in runs:
        if run["identity"]["passed"] is not True:
            continue
        key = (run["task"], run["sensor"], run["algorithm"], run["seed"])
        if key in indexed:
            raise ValueError(
                f"Duplicate seed group {key}: {indexed[key]['path']} and {run['path']}"
            )
        if run["seed"] not in seeds:
            raise ValueError(f"Chosen run {run['path']} has seed outside manifest.seeds")
        indexed[key] = run
    shared = collect_initialization_costs(runs, root)
    cells = []
    actors = {}
    for task, sensor, _ in CONDITIONS:
        key = f"{task}/{sensor}"
        if key not in actors:
            actors[key] = common_actor(
                [r for r in indexed.values() if r["task"] == task and r["sensor"] == sensor]
            )
    for algorithm in ALGORITHMS:
        for condition in CONDITIONS:
            task, sensor, partition = condition
            selected = [indexed.get((task, sensor, algorithm, seed)) for seed in seeds]
            recipes = [recipe(run["config"]) for run in selected if run]
            same_recipe = check(
                False
                if recipes and any(r != recipes[0] for r in recipes)
                else (True if len(recipes) == 3 else None),
                "Three seeds must use the same recipe",
            )
            values = {
                str(seed): seed_result(run, condition)
                for seed, run in zip(seeds, selected, strict=True)
            }
            statuses = [value["status"] for value in values.values()]
            comparison = combine([same_recipe, actors[f"{task}/{sensor}"]])["passed"]
            status = (
                "failed"
                if "failed" in statuses or comparison is False
                else (
                    "passed"
                    if all(s == "passed" for s in statuses) and comparison is True
                    else ("missing" if all(s == "missing" for s in statuses) else "incomplete")
                )
            )
            cells.append(
                {
                    "algorithm": algorithm,
                    "task": task,
                    "sensor": sensor,
                    "partition": partition,
                    "status": status,
                    "seeds": values,
                    "C1": check(
                        True if all(selected) else None,
                        "Distinct recorded training RNG seeds; shared initialization disclosed",
                    ),
                    "same_recipe": same_recipe,
                    "common_actor": f"{task}/{sensor}",
                }
            )
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "manifest_sha256": digest(manifest_path),
        "resolved_root": str(root),
        "all_passed": all(cell["status"] == "passed" for cell in cells),
        "summary": dict(Counter(cell["status"] for cell in cells)),
        "cells": cells,
        "runs": runs,
        "common_actor_checks": actors,
        "shared_initialization_costs": shared,
        "physics_task_contract": {
            "required": {
                "navigation_goal_radius_m": 0.5,
                "body_radius_m": 0.07,
                "navigation_duration_s": 300,
                "nominal_speed_limit_m_s": 20,
                "racing_gate_order": [1, 2, 3, 4, 2],
            },
            "actual_config": "Per run config.task, config.simulation and checkpoint_config",
            "historical_hardcoded_constants_verified": metric(
                None,
                None,
                "Not serialized in historical run records; current source is not "
                "proof of historical task constants. Collector never changes physics/thresholds.",
            ),
        },
        "native_S6": {
            "status": "not_collected",
            "reason": "Separate native evidence; see "
            "docs/native.md and docs/acceptance.md. Never counted as Learning C4.",
        },
    }


def markdown(report):
    """Render the matrix and shared initialization costs as a readable report."""
    lines = [
        "# Acceptance evidence",
        "",
        f"Snapshot: {report['generated_utc']}",
        "",
        f"All 18 cells passed: **{str(report['all_passed']).lower()}**. "
        f"Cell counts: `{json.dumps(report['summary'], sort_keys=True)}`.",
        "",
        "Each seed entry: status; benchmark successes/actual CSV episodes; updates; wall seconds.",
        "Missing metrics are null with reasons in the JSON. Mixed Navigation appears twice.",
        "",
    ]
    seeds = report["manifest"].get("seeds", [0, 1, 2])
    lines += [
        "| Algorithm | Condition | Status | " + " | ".join(f"Seed {s}" for s in seeds) + " |",
        "|---|---|---|" + "---|" * 3,
    ]
    runs = {run["path"]: run for run in report["runs"]}
    for cell in report["cells"]:
        entries = []
        for value in cell["seeds"].values():
            run = runs.get(value["run"])
            if run is None:
                entries.append(value["status"])
                continue
            scenes = run["benchmark"]["scenes"]
            counts = ", ".join(
                f"{s}:{scenes[s]['successes']}/{scenes[s]['episodes']}" for s in value["scenes"]
            )
            updates = value["updates"]["value"]
            wall = value["wall_seconds"]["value"]
            entries.append(
                f"{value['status']}; {counts}; u={updates}; "
                f"{round(wall, 2) if wall is not None else 'null'} s"
            )
        label = "/".join(str(v) for v in (cell["task"], cell["partition"], cell["sensor"]) if v)
        lines.append(
            f"| {cell['algorithm']} | {label} | {cell['status']} | " + " | ".join(entries) + " |"
        )
    lines += [
        "",
        GPU_SCOPE,
        "",
        "Shared initialization cost (full transitive ancestry, once per source run):",
        "Formal initialized runs report additional training; their costs exclude pretraining. "
        "Source costs cover full recorded runs, not inferred checkpoint prefixes. "
        "Unknown historical costs remain unavailable, not zero.",
        "",
    ]
    if not report["shared_initialization_costs"]:
        lines.append("None recorded.")
    for path, source in report["shared_initialization_costs"].items():
        header, cost = source["run_record"], source["cost"]
        if source["kind"] == "reference_archive":
            archive = source["reference_archive"]
            provenance = (archive.get("metadata") or {}).get("provenance", {})
            lines.append(
                f"- Reference archive `{path}`: historical training cost **unavailable**; "
                f"used by {len(source['used_by'])} formal runs through initialization ancestry. "
                f"Archive SHA-256 `{archive['archive_sha256']}`. "
                f"Recorded provenance: `{json.dumps(provenance, sort_keys=True)}`."
            )
            continue
        parent = source["initialization"]
        ancestry = (
            f" Initializes from `{parent['checkpoint']}` "
            f"(cost source `{parent['shared_cost_id']}`)."
            if parent["checkpoint"]
            else ""
        )
        lines.append(
            f"- `{path}`: {cost['updates']['value']} updates, "
            f"{cost['interactions']['value']} interactions, {cost['wall_seconds']['value']} s; "
            f"Python {header.get('python')}, JAX {header.get('versions', {}).get('jax')}; "
            f"used by {len(source['used_by'])} formal runs through initialization ancestry."
            f"{ancestry}"
        )
    lines += [
        "",
        "Per-scene outcomes (including S06/D06), every checkpoint evaluation, SHA-256, "
        "resolved recipes, provenance, missing reasons and costs are in the companion JSON.",
        "Historical hardcoded physical constants are not independently established by these "
        "records. Native S6 is separate and not graded here.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    """Collect the explicit run manifest and write the evidence matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="JSON with explicit runs and optional root/seeds"
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("results/acceptance/current"))
    args = parser.parse_args(argv)
    try:
        report = collect(args.manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    prefix = args.output_prefix
    protected = [run["path"] for run in report["runs"]] + list(
        report["shared_initialization_costs"]
    )
    if any(prefix.resolve().is_relative_to(path) for path in protected):
        parser.error("Output prefix must be outside selected run directories")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, content in (
        (".json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
        (".md", markdown(report)),
    ):
        destination = Path(str(prefix) + suffix)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(destination)
    print(
        json.dumps(
            {
                "all_passed": report["all_passed"],
                "cells": report["summary"],
                "output_prefix": str(prefix),
            }
        )
    )


if __name__ == "__main__":
    main()
