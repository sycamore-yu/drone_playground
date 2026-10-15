"""Collect C1-C6 evidence from an explicit manifest, without importing simulation."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
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
SINGLE_RULE = "first_full_validation_pass"
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
    if not path.exists() and list(path.parent.glob("events.out.tfevents.*")):
        return read_tensorboard_events(path.parent, issues)
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


def read_tensorboard_events(directory, issues):
    """Read existing v3 scalar updates and selection events without simulation imports."""
    from tensorboardX.proto.event_pb2 import Event

    events, updates = [], {}
    for path in sorted(directory.glob("events.out.tfevents.*")):
        try:
            with path.open("rb") as stream:
                while header := stream.read(12):
                    if len(header) != 12:
                        raise ValueError("Incomplete TensorBoard record header")
                    length = struct.unpack("<Q", header[:8])[0]
                    payload = stream.read(length)
                    if len(payload) != length or len(stream.read(4)) != 4:
                        raise ValueError("Incomplete TensorBoard record")
                    event = Event.FromString(payload)
                    for value in event.summary.value:
                        if value.tag.startswith("train/") and value.HasField("simple_value"):
                            update = updates.setdefault(
                                event.step, {"event": "update", "update": event.step}
                            )
                            update[value.tag.removeprefix("train/")] = value.simple_value
                        elif value.tag.startswith("event/") and value.tensor.string_val:
                            events.append(json.loads(value.tensor.string_val[0]))
        except (OSError, ValueError) as error:
            issues.append(f"{path}: {error}")
    return events + [updates[step] for step in sorted(updates)]


def resolve_run_path(value, run):
    """Resolve the current run's checkpoint or evaluation location."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (run / path).resolve()


def checkpoint(path):
    """Inspect the persisted archive and payload checksum without loading an Actor."""
    result = {"path": str(path), "archive_sha256": None, "metadata": None}
    if path.is_dir():
        issues = []
        metadata = read_mapping(path / "metadata.json", issues)
        result["metadata"] = metadata
        result["integrity"] = check(
            not issues
            and metadata.get("format_version") == 3
            and metadata.get("purpose") == "model"
            and metadata.get("has_inference_actor") is True
            and isinstance(metadata.get("actor"), dict)
            and all(
                (path / part / "commit_success.txt").is_file() for part in ("actor", "training")
            ),
            "Recorded v3 model identity and completed Orbax actor/training saves; "
            "payload bytes are not rehashed or replayed",
            *issues,
        )
        return result
    try:
        result["archive_sha256"] = digest(path)
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            payload_sha = hashlib.sha256(archive.read("variables.msgpack")).hexdigest()
        result.update(metadata=metadata, payload_sha256=payload_sha)
        result["integrity"] = check(
            metadata.get("sha256") == payload_sha
            and metadata.get("purpose") == "inference"
            and metadata.get("format_version") == 2
            and isinstance(metadata.get("actor"), dict),
            "Verified v2 inference metadata, Actor specification and payload SHA-256",
        )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        result["integrity"] = check(None, str(error))
    return result


def scene_evidence(directory, scene, task, claimed):
    """Recompute quality and denominators from complete, independent CSV episodes."""
    path = directory / "episodes.csv"
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
            rows = [row for row in csv.DictReader(stream) if row.get("scene") == scene]
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


def evaluate(directory, task, seed, mode, *, fixed_checkpoint_seeds=False):
    """Verify saved episode CSVs, task quality and evaluation seed partitions."""
    issues = []
    if type(fixed_checkpoint_seeds) is not bool:
        issues.append("fixed_checkpoint_seeds must be boolean")
    report = read_mapping(directory / "report.json", issues)
    scenes = NAV_SCENES if task == "navigation" else ("empty" if task == "tracking" else "racing",)
    if (
        task == "navigation"
        and mode == "checkpoint_eval"
        and set(report.get("reports", {})) == set(NAV_SCENES[:6])
    ):
        scenes = NAV_SCENES[:6]
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
            cohort = 0 if fixed_checkpoint_seeds is True and mode == "checkpoint_eval" else index
            if actual != [int(base + 10_000 * seed + 100 * cohort + scene_index)]:
                issues.append(f"{scenes[scene_index]}: actual CSV seeds disagree with report")
        else:
            issues.append("Missing evaluation_index/seed_base")
    primary = scenes[:6] if task == "navigation" else scenes
    return {
        "directory": str(directory),
        "seed_base": base,
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
    event_source = str(
        Path(source)
        / ("events.out.tfevents.*" if header.get("layout_version") == 3 else "metrics.jsonl")
    )
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
            origin = f"{event_source}:last update (lower bound while running)"
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
            event_source,
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


def check_single_full(result, screening, events):
    """Grade selection using saved episode evidence; no model or CSV reread."""
    path = Path(result["path"])
    config, header = result["config"], result["run_record"]
    task, seed = result["task"], result["seed"]
    selection = result["selection"]
    audit = result.get("selection_audit")
    cp = result["checkpoint"]
    metadata = cp.get("metadata") or {}
    evaluations, benchmark = result["checkpoint_evaluations"], result["benchmark"]
    independent = result["C5"]["independent_seed_sets"]
    chosen = next((e for e in evaluations if e["update"] == selection["update"]), None)
    prior_passes = [
        e
        for e in evaluations
        if e["update"] < selection["update"] and e["quality"]["passed"] is True
    ]
    screened = bool(audit) or (
        screening is not None
        and screening.get("complete") is True
        and screening.get("seed_base") == 1_000_000
        and chosen is not None
        and chosen["seed_base"] == 1_500_000
        and screening.get("training_seed") == seed
        and screening.get("evaluation_index") == selection["evaluation_index"]
        and all(
            screening.get("reports", {}).get(scene, {}).get("episodes") == 8
            and screening["reports"][scene].get("successes", 0)
            >= (6 if task == "navigation" else 7)
            and (
                task != "tracking"
                or (
                    number(screening["reports"][scene].get("successful_position_rmse")) is not None
                    and screening["reports"][scene]["successful_position_rmse"] <= 0.35
                )
            )
            for scene in (
                NAV_SCENES[:6]
                if task == "navigation"
                else ("empty" if task == "tracking" else "racing",)
            )
        )
    )
    selected_path = audit.get("checkpoint") if audit else header.get("selected_checkpoint")
    c5_ok = (
        chosen is not None
        and chosen["quality"]["passed"] is True
        and chosen["partition"]["passed"] is True
        and not prior_passes
        and not any(
            e.get("event") == "full_validation"
            and e.get("passed") is True
            and e.get("update", selection["update"]) < selection["update"]
            for e in events
        )
        and screened
        and (
            bool(audit)
            or (
                config.get("learning", {}).get("selection") == SINGLE_RULE
                and config.get("learning", {}).get("required_consecutive_passes") == 1
            )
        )
        and selection.get("consecutive_passes") == 1
        and selection.get("evaluation_index") == chosen["evaluation_index"]
        and metadata.get("provenance", {}).get("updates") == selection["update"]
        and selected_path is not None
        and resolve_run_path(selected_path, path) == Path(cp["path"])
        and independent is True
        and benchmark["partition"]["passed"] is True
        and benchmark.get("checkpoint_link", {"passed": True})["passed"] is True
    )
    result["C5"] = check(
        c5_ok,
        "First full validation and independent frozen benchmark; "
        "historical audit or recorded screening, without repeated training",
    )
    result["C5"].update(
        first_passing_updates=[selection["update"]],
        independent_seed_sets=independent,
        protocol=SINGLE_RULE,
        screening=screening,
    )


def collect_run(path, root, *, benchmark_run=None, allow_first_full=False):
    """Collect one selected run without replacing failures from another run."""
    issues = []
    config = read_mapping(path / "config.yaml", issues)
    header = read_mapping(path / "run.json", issues)
    events = read_events(path / "metrics.jsonl", issues)
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
    if header.get("layout_version") not in (2, 3):
        message = f"Unsupported run layout version: {header.get('layout_version')!r}"
        issues.append(message)
        result["identity"] = check(False, message)
        return result
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
    selection = header.get("selection", {})
    audit_directory = path.parent / ".scheduler" / "frozen_audits" / path.name
    audit = (
        read_mapping(audit_directory / "audit.json", issues)
        if allow_first_full and (audit_directory / "audit.json").is_file()
        else None
    )
    if audit:
        selected_report = read_mapping(path / audit["source_validation"], issues)
        selection = {
            "rule": SINGLE_RULE,
            "checkpoint": audit["checkpoint"],
            "update": int(Path(audit["checkpoint"]).parent.name.removeprefix("step-")),
            "evaluation_index": selected_report.get("evaluation_index"),
            "consecutive_passes": 1,
            "full_validation": audit["source_validation"],
        }
        result["selection_audit"] = audit
    result["selection"] = selection or None
    cp = (
        checkpoint(resolve_run_path(selection["checkpoint"], path))
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
    folder = path / "checkpoints"
    prefix = "step-"
    directories = {
        int(p.name.removeprefix(prefix)): p
        for p in folder.glob(f"{prefix}*")
        if p.is_dir() and p.name.removeprefix(prefix).isdigit() and (p / "report.json").is_file()
    }
    if header.get("layout_version") == 3:
        directories.update(
            (int(p.name.removeprefix("model_")), p)
            for p in folder.glob("model_*")
            if (p / "report.json").is_file()
        )
    for event in events:
        if (
            header.get("layout_version") == 2
            and event.get("event") == "checkpoint_evaluation"
            and type(event.get("update")) is int
        ):
            update = event["update"]
            name = f"step-{update:06d}"
            directories.setdefault(update, folder / name)
    evaluations = []
    for update, directory in sorted(directories.items()):
        entry = evaluate(
            directory,
            task,
            seed,
            "checkpoint_eval",
            fixed_checkpoint_seeds=config.get("benchmark", {}).get("fixed_checkpoint_seeds", False),
        )
        entry["update"] = update
        evaluations.append(entry)
    result["checkpoint_evaluations"] = evaluations
    if audit:
        benchmark = evaluate(audit_directory, task, seed, "benchmark")
        benchmark["checkpoint_link"] = check(
            audit.get("status") == "accepted"
            and audit.get("source_run") == str(path)
            and audit.get("benchmark_attempts") == 1
            and audit.get("benchmark_passed") is True,
            "Reuse completed one-time frozen audit of its recorded first validation candidate",
        )
    elif benchmark_run is not None:
        benchmark_issues = []
        frozen_config = read_mapping(benchmark_run / "config.yaml", benchmark_issues)
        frozen_header = read_mapping(benchmark_run / "run.json", benchmark_issues)
        benchmark = evaluate(
            resolve_run_path(frozen_header.get("evaluation_directory", "eval/001"), benchmark_run),
            task,
            seed,
            "benchmark",
        )
        loaded = frozen_header.get("evaluated_checkpoint", {})
        matched_inputs = all(
            frozen_config.get(key, {}) == config.get(key, {})
            for key in ("task", "sensor", "method", "observation", "controller")
        ) and {
            k: v
            for k, v in frozen_config.get("simulation", {}).items()
            if k not in {"num_envs", "device"}
        } == {
            k: v for k, v in config.get("simulation", {}).items() if k not in {"num_envs", "device"}
        }
        configured_checkpoint = frozen_config.get("checkpoint")
        declared_seed_base = frozen_config.get("benchmark", {}).get("seed_base")
        if declared_seed_base is None:
            declared_seed_base = 2_000_000
        linked = (
            not benchmark_issues
            and frozen_header.get("status") == "completed"
            and frozen_header.get("layout_version") == 2
            and digest(benchmark_run / "config.yaml") == frozen_header.get("config_sha256")
            and frozen_config.get("mode") == "benchmark"
            and frozen_config.get("seed") == seed
            and benchmark["seed_base"] == declared_seed_base
            and frozen_config.get("benchmark", {}).get("protocol") == "learning"
            and not frozen_config.get("benchmark", {}).get("allow_environment_change", False)
            and isinstance(configured_checkpoint, str)
            and str(resolve_run_path(configured_checkpoint, benchmark_run)) == cp["path"]
            and loaded.get("path") == cp["path"]
            and bool(cp.get("payload_sha256"))
            and loaded.get("payload_sha256") == cp.get("payload_sha256")
            and matched_inputs
        )
        benchmark["source_run"] = str(benchmark_run)
        benchmark["checkpoint_link"] = check(
            linked,
            "Verified separate frozen run config, actual loaded payload, selected model and inputs"
            if linked
            else "Separate frozen evaluation lacks matching checkpoint/input provenance",
            *benchmark_issues,
        )
    else:
        benchmark = evaluate(
            resolve_run_path(header["benchmark_directory"], path)
            if header.get("benchmark_directory")
            else path / "eval/001",
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
        and resolve_run_path(header["selected_checkpoint"], path) == Path(cp["path"])
        and independent
        and benchmark["partition"]["passed"] is True
        and benchmark.get("checkpoint_link", {"passed": True})["passed"] is True
    )
    result["C5"] = check(
        c5_ok if selection else None,
        "Verified first consecutive triple, selection, archive update and CSV partitions"
        if c5_ok
        else "Missing/invalid consecutive evaluations, selection or holdout",
    )
    result["C5"].update(first_passing_updates=first, independent_seed_sets=independent)
    if allow_first_full and selection.get("rule") == SINGLE_RULE:
        screening = (
            read_mapping(path / selection["screening"], issues)
            if selection.get("screening")
            else None
        )
        check_single_full(result, screening, events)
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
            f"{path / 'metrics.jsonl'}:selected checkpoint evaluation",
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
                events = [] if reference else read_events(Path(source) / "metrics.jsonl", issues)
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


def recipe(config, *, allow_first_full=False):
    """Select configuration fields that define the three-seed training recipe."""
    result = {
        key: value
        for key, value in config.items()
        if key not in {"seed", "output", "checkpoint", "resume", "replay_path", "mode"}
    }
    if "benchmark" in result:
        result["benchmark"] = {
            key: value for key, value in result["benchmark"].items() if key != "frozen_seed_base"
        }
        if not result["benchmark"]:
            del result["benchmark"]
    if allow_first_full:
        # Historical stopping/serialization rules do not change the optimization recipe.
        for key in ("results", "early_stopping"):
            result.pop(key, None)
        result["learning"] = {
            k: v
            for k, v in result.get("learning", {}).items()
            if k not in {"selection", "required_consecutive_passes"}
        }
    return result


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
                "observation": config.get("observation", {}),
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


def success_rate_interval(successes, episodes):
    """Compute a 95% Wilson interval conditional on one fixed trained policy."""
    if (
        type(successes) is not int
        or type(episodes) is not int
        or episodes <= 0
        or not 0 <= successes <= episodes
    ):
        return None
    estimate = successes / episodes
    z = statistics.NormalDist().inv_cdf(0.975)
    denominator = 1 + z**2 / episodes
    center = (estimate + z**2 / (2 * episodes)) / denominator
    margin = (
        z * math.sqrt(estimate * (1 - estimate) / episodes + z**2 / (4 * episodes**2)) / denominator
    )
    return {
        "estimate": estimate,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
        "confidence_level": 0.95,
        "method": "wilson",
        "successes": successes,
        "episodes": episodes,
    }


def seed_success_rate_spread(values):
    """Describe variability across policies without pooling their episodes."""
    rates = {}
    for value in values.values():
        for scene, interval in value.get("success_rate_intervals", {}).items():
            if interval["value"] is not None:
                rates.setdefault(scene, []).append(interval["value"]["estimate"])
    return {
        scene: {
            "available_training_seeds": len(samples),
            "mean": statistics.mean(samples),
            "min": min(samples),
            "max": max(samples),
            "sample_standard_deviation": statistics.stdev(samples) if len(samples) > 1 else None,
        }
        for scene, samples in rates.items()
    }


def stability_window(run, report_path):
    """Link a completed unchanged continuation to its selected source and actual last model."""
    issues = []
    report = read_mapping(report_path, issues)
    evidence = {"report": str(report_path), "record": report, **check(None, *issues)}
    if not report or report.get("status") == "running":
        return evidence

    def require(valid, reason):
        if not valid:
            raise ValueError(reason)

    try:
        source = run["checkpoint"]
        metadata = source["metadata"]
        provenance = metadata["provenance"]
        config = run["config"]
        batch = config["simulation"]["num_envs"] * config["learning"]["options"]["horizon"]
        steps, interval = math.ceil(900_000 / batch), math.ceil(122_880 / batch)
        origin, interactions = provenance["updates"], provenance["interactions"]
        require(
            report["status"] == "completed"
            and report.get("actor_lr_scale", 1) == 1
            and not report.get("diagnostic", False)
            and report["source_checkpoint"] == source["path"]
            and report["source_run"] == run["path"]
            and report["source_config_sha256_reused"] == provenance["config_sha256"]
            and report["source_sha256_reused"] == provenance["source_sha256"]
            and report["runtime_provenance"]["source_sha256"] == provenance["source_sha256"]
            and report["origin_update"] == origin == run["selection"]["update"]
            and report["origin_interactions"] == interactions
            and config.get("initial_checkpoint") is None
            and config.get("resume") is None,
            "Window must continue the selected scratch model with unchanged code and recipe",
        )
        require(
            report["planned_updates"] == steps
            and report["evaluation_interval_updates"] == interval
            and report["planned_additional_interactions"] == steps * batch
            and report["actual_additional_interactions"] == steps * batch
            and report["last_update"] == origin + steps
            and report["last_interactions"] == interactions + steps * batch,
            "Incomplete registered continuation budget or mismatched actual counters",
        )
        selected = next(e for e in run["checkpoint_evaluations"] if e["update"] == origin)
        rows = report["baseline_cases"]
        cases = {tuple(row[:4]): row[4] for row in rows}
        require(len(cases) == len(rows) == 150, "Baseline must contain 150 distinct cases")
        original = {key for key, event in cases.items() if event == "SUCCESS"}
        geometry = metadata["config"]["training_sampling"]["geometry_sha256"]
        require(
            report["evaluation_seed_base"] == selected["seed_base"]
            and all(event in EVENTS for event in cases.values())
            and set(key[0] for key in cases) == set(NAV_SCENES[:6])
            and all(
                {(int(k[2]), int(k[3])) for k in cases if k[0] == scene}
                == {tuple(k) for k in selected["scenes"][scene]["case_ids"]}
                and sum(k[0] == scene for k in original)
                == selected["scenes"][scene]["successes"]
                >= 23
                for scene in NAV_SCENES[:6]
            )
            and all(k[1] == geometry[k[0]] for k in cases),
            "Window baseline differs from the already verified full validation",
        )
        offsets = [*range(interval, steps, interval), steps]
        checks = report["checks"]
        require(len(checks) == len(offsets), "Missing scheduled stability check")
        passed = []
        for row, offset in zip(checks, offsets, strict=True):
            lost, new = (
                {tuple(k) for k in row["lost_initial_cases"]},
                {tuple(k) for k in row["new_success_cases"]},
            )
            require(
                lost <= original
                and new <= cases.keys() - original
                and len(lost) == len(row["lost_initial_cases"])
                and len(new) == len(row["new_success_cases"]),
                "Invalid retention case identities",
            )
            successful = (original - lost) | new
            counts = {s: sum(k[0] == s for k in successful) for s in NAV_SCENES[:6]}
            retained = len(original - lost)
            ok = min(counts.values()) >= 23 and retained * 100 >= len(original) * 95
            require(
                row["update"] == origin + offset
                and row["additional_interactions"] == offset * batch
                and Path(row["model"]).name == f"model_{origin + offset:06d}"
                and row["per_scene_successes"] == counts
                and row["successes"] == len(successful)
                and row["initial_successes"] == len(original)
                and row["retained_initial_successes"] == retained
                and math.isclose(row["retention_fraction"], retained / len(original))
                and row["passed"] is ok,
                "Check counters, scene counts or retained-case evidence disagree",
            )
            passed.append(ok)
        require(report["passed"] is all(passed), "Window summary disagrees with its checks")
        updates = read_events(report_path.parent / "metrics.jsonl", issues)
        require(
            not issues
            and len(updates) == steps
            and config["learning"]["options"]["lr"] > 0
            and all(
                row["update"] == origin + offset
                and row["interactions"] == interactions + offset * batch
                and number(row.get("actor_grad_norm")) is not None
                and row["actor_grad_norm"] > 0
                and number(row.get("kl_actor_update_skipped")) is not None
                and 0 <= row["kl_actor_update_skipped"] < 1
                for offset, row in enumerate(updates, 1)
            ),
            "Missing actual PPO updates or all Actor updates skipped",
        )
        final = checkpoint(Path(report["final_checkpoint"]))
        evidence["final_checkpoint"] = final
        final_meta = final["metadata"] or {}
        last = final_meta.get("provenance", {})
        require(
            report["final_checkpoint"] == checks[-1]["model"]
            and final["integrity"]["passed"] is True
            and final_meta["config"] == metadata["config"]
            and final_meta["actor"] == metadata["actor"]
            and last["status"] == "completed"
            and last["updates"] == origin + steps
            and last["interactions"] == interactions + steps * batch
            and last["stability_source_checkpoint"] == source["path"]
            and Path(last["stability_window_report"]).resolve() == report_path
            and last["source_sha256"] == provenance["source_sha256"],
            "Final model must be the completed window's actual last state, without rollback",
        )
        evidence.update(check(all(passed), "All scheduled per-scene and retention gates checked"))
    except (KeyError, TypeError, ValueError, StopIteration, ZeroDivisionError) as error:
        evidence.update(check(False, str(error)))
    return evidence


def apply_ppo_stability(run, entry, root, development_seed_bases):
    """Keep selection evidence, grade stability, then use only the final frozen model's quality."""
    if "selection_benchmark" in run:
        run["benchmark"] = run.pop("selection_benchmark")
    for key in ("final_benchmark_link", "final_benchmark_source", "final_stage_costs"):
        run.pop(key, None)
    run["final_benchmark_verified"] = False
    run["ppo_stability"] = check(None, "No completed stability window selected")
    if entry is None:
        return
    report_path = resolve_path(entry["window_report"], root)
    window = stability_window(run, report_path)
    run["ppo_stability"] = window
    if window["passed"] is not True or not entry.get("benchmark_run"):
        return
    directory = resolve_path(entry["benchmark_run"], root)
    issues = []
    config = read_mapping(directory / "config.yaml", issues)
    header = read_mapping(directory / "run.json", issues)
    final = window["final_checkpoint"]
    benchmark = evaluate(
        resolve_run_path(header.get("evaluation_directory", "."), directory),
        run["task"],
        run["seed"],
        "benchmark",
    )
    original = run["config"]
    matched = all(
        config.get(key) == original.get(key)
        for key in ("task", "sensor", "method", "observation", "controller")
    ) and {
        k: v for k, v in config.get("simulation", {}).items() if k not in {"num_envs", "device"}
    } == {
        k: v for k, v in original.get("simulation", {}).items() if k not in {"num_envs", "device"}
    }
    development_seeds = set(run["benchmark"]["seeds"]) | {
        seed for entry in run["checkpoint_evaluations"] for seed in entry["seeds"]
    }
    loaded = header.get("evaluated_checkpoint", {})
    declared = config.get("benchmark", {})
    selected = next(
        e for e in run["checkpoint_evaluations"] if e["update"] == run["selection"]["update"]
    )
    geometry = final["metadata"]["config"]["training_sampling"]["geometry_sha256"]
    task_geometry = all(
        benchmark["scenes"][scene]["claimed"].get("geometry", {}).get("sha256") == geometry[scene]
        and bool(selected["scenes"][scene]["claimed"].get("task_contract"))
        and benchmark["scenes"][scene]["claimed"].get("task_contract")
        == selected["scenes"][scene]["claimed"]["task_contract"]
        for scene in NAV_SCENES[:6]
    )
    linked = (
        not issues
        and header.get("status") == "completed"
        and header.get("layout_version") == 3
        and config.get("mode") == "benchmark"
        and config.get("seed") == run["seed"]
        and config.get("checkpoint") == loaded.get("path") == final["path"]
        and loaded.get("model_metadata") == final["metadata"]
        and declared.get("protocol") == "learning"
        and not declared.get("allow_environment_change", False)
        and number(declared.get("seed_base")) is not None
        and declared["seed_base"] == benchmark["seed_base"] >= 3_000_000
        and declared["seed_base"] not in development_seed_bases
        and not development_seeds.intersection(benchmark["seeds"])
        and benchmark["partition"]["passed"] is True
        and matched
        and task_geometry
    )
    run["final_benchmark_link"] = check(
        linked,
        "Final frozen run must load the actual last model and use an independent cohort",
        *issues,
    )
    run["final_benchmark_source"] = {
        "path": str(directory),
        "config": config,
        "run_record": header,
    }
    if linked:
        run["selection_benchmark"] = run["benchmark"]
        run["benchmark"] = benchmark
        run["final_benchmark_verified"] = True
        run["final_stage_costs"] = {
            "selection_run": run["cost"],
            "continuation_wall_seconds": window["record"].get("total_wall_seconds"),
            "continuation_update_seconds": window["record"].get("training_update_seconds"),
            "frozen_run": cost_evidence(header, [], directory),
            "scope": "Separate recorded stage costs include evaluations; no isolated runtime claim",
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
    if "ppo_stability" in run:
        criteria += [run["ppo_stability"], run.get("final_benchmark_link", check(None))]
        if not run["final_benchmark_verified"]:
            quality = check(None, "Independent frozen test of the actual final model unavailable")
            criteria[0] = quality
    passed = combine(criteria)["passed"]
    terminal_failure = run["recorded_status"] in {"failed", "error"}
    status = (
        "failed"
        if terminal_failure or (passed is False and run["recorded_status"] != "running")
        else (
            "passed"
            if passed is True
            and (
                run["recorded_status"] in {"accepted", "frozen_benchmark_failed"}
                or run.get("selection_audit", {}).get("status") == "accepted"
            )
            else "incomplete"
        )
    )
    return {
        "status": status,
        "run": run["path"],
        "quality": quality,
        "C5": run["C5"],
        "C6": run["C6"],
        "ppo_stability": run.get("ppo_stability"),
        "final_benchmark_link": run.get("final_benchmark_link"),
        "final_model_counters": {
            key: run.get("ppo_stability", {}).get("record", {}).get(key)
            for key in ("last_update", "last_interactions", "actual_additional_interactions")
        }
        if "ppo_stability" in run
        else None,
        "scenes": list(scenes),
        "success_rate_intervals": {
            scene: metric(
                success_rate_interval(
                    run["benchmark"]["scenes"][scene]["successes"],
                    run["benchmark"]["scenes"][scene]["episodes"],
                )
                if run.get("final_benchmark_verified", True)
                else None,
                run["benchmark"]["scenes"][scene]["csv"],
                "Verified episode counts unavailable",
            )
            for scene in scenes
        },
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
    benchmark_runs = manifest.get("benchmark_runs", {})
    if not isinstance(benchmark_runs, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in benchmark_runs.items()
    ):
        raise ValueError("manifest.benchmark_runs must map chosen training paths to frozen runs")
    benchmarks = {resolve_path(k, root): resolve_path(v, root) for k, v in benchmark_runs.items()}
    if len(benchmarks) != len(benchmark_runs) or set(benchmarks) - set(paths):
        raise ValueError("benchmark_runs contains duplicate or unselected training paths")
    stability = manifest.get("ppo_navigation_stability")
    if stability is not None:
        if not isinstance(stability, dict) or any(
            not isinstance(k, str)
            or not k
            or not isinstance(v, dict)
            or set(v) - {"window_report", "benchmark_run"}
            or not isinstance(v.get("window_report"), str)
            or not v["window_report"]
            or (
                "benchmark_run" in v
                and (not isinstance(v["benchmark_run"], str) or not v["benchmark_run"])
            )
            for k, v in stability.items()
        ):
            raise ValueError("ppo_navigation_stability must map runs to window/frozen paths")
        resolved = {resolve_path(k, root): v for k, v in stability.items()}
        if len(resolved) != len(stability) or set(resolved) - set(paths):
            raise ValueError("ppo_navigation_stability contains duplicate or unselected runs")
        stability = resolved
    development_bases = manifest.get("development_seed_bases", [])
    if not isinstance(development_bases, list) or any(
        type(base) is not int or base < 0 for base in development_bases
    ):
        raise ValueError("development_seed_bases must list consumed evaluation seed bases")
    reuse = manifest.get("reuse_reports", {})
    if not isinstance(reuse, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in reuse.items()
    ):
        raise ValueError("reuse_reports must map selected runs to previously verified reports")
    reuse = {resolve_path(k, root): resolve_path(v, root) for k, v in reuse.items()}
    if set(reuse) - set(paths):
        raise ValueError("reuse_reports contains an unselected run")
    snapshots = {p: json.loads(p.read_text()) for p in set(reuse.values())}
    allow_first_full = manifest.get("selection") == SINGLE_RULE
    runs = []
    for path in paths:
        if path not in reuse:
            runs.append(
                collect_run(
                    path,
                    root,
                    benchmark_run=benchmarks.get(path),
                    allow_first_full=allow_first_full,
                )
            )
            continue
        snapshot = snapshots[reuse[path]]
        run = next((r for r in snapshot["runs"] if r["path"] == str(path)), None)
        if run is None:
            raise ValueError(f"No verified evidence for {path} in {reuse[path]}")
        if run["config"] != yaml.safe_load((path / "config.yaml").read_text()) or run[
            "run_record"
        ] != json.loads((path / "run.json").read_text()):
            raise ValueError(
                f"Saved configuration or run record changed after verification: {path}"
            )
        old_benchmarks = {
            resolve_path(k, Path(snapshot["resolved_root"])): resolve_path(
                v, Path(snapshot["resolved_root"])
            )
            for k, v in snapshot["manifest"].get("benchmark_runs", {}).items()
        }
        if old_benchmarks.get(path) != benchmarks.get(path):
            raise ValueError(f"Frozen benchmark selection changed after verification: {path}")
        run["reused_evidence"] = {
            "report": str(reuse[path]),
            "verified_utc": snapshot["generated_utc"],
            "scope": "Reuse recorded episode, selection and model-integrity checks; "
            "configuration and run record remain unchanged. No repeated payload/CSV audit.",
        }
        runs.append(run)
    if stability is not None:
        for run in runs:
            path = Path(run["path"])
            if run["task"] == "navigation" and run["algorithm"] == "ppo":
                if path in reuse and "ppo_stability" in run:
                    snapshot = snapshots[reuse[path]]
                    previous = {
                        resolve_path(k, Path(snapshot["resolved_root"])): v
                        for k, v in snapshot["manifest"].get("ppo_navigation_stability", {}).items()
                    }
                    if (
                        previous.get(path) == stability.get(path)
                        and snapshot["manifest"].get("development_seed_bases", [])
                        == development_bases
                    ):
                        record = run["ppo_stability"]
                        if (
                            record.get("report")
                            and json.loads(Path(record["report"]).read_text()) != record["record"]
                        ):
                            raise ValueError("Stability report changed after verification")
                        frozen = run.get("final_benchmark_source")
                        if frozen and (
                            yaml.safe_load((Path(frozen["path"]) / "config.yaml").read_text())
                            != frozen["config"]
                            or json.loads((Path(frozen["path"]) / "run.json").read_text())
                            != frozen["run_record"]
                        ):
                            raise ValueError("Final frozen run changed after verification")
                        continue
                apply_ppo_stability(run, stability.get(path), root, development_bases)
            elif path in stability:
                raise ValueError("ppo_navigation_stability applies only to PPO Navigation")
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
            recipes = [
                recipe(run["config"], allow_first_full=allow_first_full) for run in selected if run
            ]
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
            # Task success is graded per algorithm; cross-algorithm differences are disclosed.
            comparison = same_recipe["passed"]
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
                    "seed_success_rate_spread": seed_success_rate_spread(values),
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
        "manifest_sha256": None if reuse else digest(manifest_path),
        "resolved_root": str(root),
        "all_passed": all(cell["status"] == "passed" for cell in cells),
        "summary": dict(Counter(cell["status"] for cell in cells)),
        "cells": cells,
        "runs": runs,
        "common_actor_checks": actors,
        "comparison_scope": "Each algorithm is accepted against the task's success criteria. "
        "Cross-algorithm Actor/configuration equality is reported separately; different "
        "training rewards are allowed and do not invalidate individual convergence.",
        "uncertainty_scope": "Per-seed, per-scene 95% Wilson intervals describe finite test "
        "episode sampling conditional on a fixed trained policy and the stated evaluation "
        "distribution. Training-seed means, ranges and sample standard deviations are "
        "descriptive; three seeds do not establish general training reliability. Episodes "
        "from different trained policies are not pooled. Intervals do not change acceptance gates.",
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
        "PPO Navigation stability and final-model counters/costs are separate from selection; "
        "pending final tests cannot use initial-model benchmark quality.",
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
            counts = (
                ", ".join(
                    f"{s}:{scenes[s]['successes']}/{scenes[s]['episodes']}" for s in value["scenes"]
                )
                if run.get("final_benchmark_verified", True)
                else "final frozen test pending"
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
        report["uncertainty_scope"],
        "Per-seed intervals and per-scene seed spread are in the companion JSON.",
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
