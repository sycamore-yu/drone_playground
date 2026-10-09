"""Exercise the public collector CLI with real CSV denominators and provenance files."""

import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "report_acceptance.py"
NAV_SCENES = ["S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06"]


def write_json(path, value):
    """Provide write json for the surrounding execution."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evaluation(path, task="tracking", seed=0, index=0, successes=100, count=100):
    """Provide evaluation for the surrounding execution."""
    mode = "checkpoint_eval" if index else "benchmark"
    base = 1_000_000 if index else 2_000_000
    scenes = NAV_SCENES if task == "navigation" else ["empty" if task == "tracking" else "racing"]
    reports = {}
    all_rows = []
    path.mkdir(parents=True, exist_ok=True)
    for scene_index, scene in enumerate(scenes):
        rows = []
        for world in range(count):
            rows.append(
                {
                    "scene": scene,
                    "event": "SUCCESS" if world < successes else "COLLISION",
                    "initialization_seed": base + 10_000 * seed + 100 * index + scene_index,
                    "world_index": world,
                    "position_rmse": 0.1,
                    "flight_seconds": 20,
                    "gates_passed": 5,
                    "gate_order": "1,2,3,4,2",
                    "min_clearance": 0.2,
                }
            )
        all_rows.extend(rows)
        reports[scene] = {"episodes": count, "successes": min(count, successes), "passed": True}
    with (path / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0])
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(
        path / "report.json",
        {
            "mode": mode,
            "seed_base": base,
            "training_seed": seed,
            "evaluation_index": index,
            "reports": reports,
            "passed": True,
        },
    )


def make_run(
    root, name="run", *, task="tracking", algorithm="ppo", seed=0, passes=(True, True, True)
):
    """Provide make run for the surrounding execution."""
    path = root / name
    path.mkdir()
    sensor = "depth" if task == "navigation" else "state"
    config = {
        "task": {"name": task, "duration": {"tracking": 20, "racing": 60, "navigation": 300}[task]},
        "sensor": {"name": sensor},
        "seed": seed,
        "method": {"name": "policy", "actor": {"kind": sensor}},
        "simulation": {
            "dynamics": "first_principles",
            "drone": "cf21B_500",
            "physics_hz": 500,
            "method_hz": 50,
            "device": "gpu",
        },
        "learning": {
            "algorithm": algorithm,
            "evaluation_interval": 100,
            "required_consecutive_passes": 3,
            "selection": "last_of_first_three_consecutive_passes",
        },
    }
    (path / "config.yaml").write_text(yaml.safe_dump(config))
    update = len(passes) * 100
    cp = path / "checkpoints" / f"step-{update:06d}" / "policy.zip"
    cp.parent.mkdir(parents=True)
    payload = b"opaque-test-parameters"
    with zipfile.ZipFile(cp, "w") as archive:
        archive.writestr("variables.msgpack", payload)
        archive.writestr(
            "metadata.json",
            json.dumps(
                {
                    "format_version": 2,
                    "purpose": "inference",
                    "kind": sensor,
                    "actor": {
                        "kind": sensor,
                        "action_size": 4 if sensor == "state" else 3,
                        "hidden_size": 192,
                    },
                    "config": {"experiment": config},
                    "provenance": {"updates": update},
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ),
        )
    header = {
        "layout_version": 2,
        "status": "accepted",
        "python": "3.12.15",
        "versions": {"jax": "0.11.2"},
        "source_sha256": "same-source",
        "devices": ["cuda:0"],
        "updates": update,
        "interactions": 3000,
        "wall_seconds": 10,
        "training_wall_seconds": 9,
        "interactions_per_second": 300,
        "selected_checkpoint": str(cp.relative_to(path)),
        "benchmark_directory": "eval/001",
        "selection": {
            "checkpoint": str(cp.relative_to(path)),
            "update": update,
            "evaluation_index": len(passes),
            "consecutive_passes": 3,
            "rule": "last_of_first_three_consecutive_passes",
        },
        "config_sha256": hashlib.sha256((path / "config.yaml").read_bytes()).hexdigest(),
    }
    write_json(path / "run.json", header)
    events = [
        {
            "event": "update",
            "update": update,
            "interactions": 3000,
            "gpu_memory_mib": 120,
            "gpu_utilization_percent": 65,
        }
    ]
    count = 25 if task == "navigation" else 100
    for index, passed in enumerate(passes, 1):
        evaluation(
            path / "checkpoints" / f"step-{100 * index:06d}",
            task,
            seed,
            index,
            count if passed else 0,
            count,
        )
        events.append({"event": "checkpoint_evaluation", "update": 100 * index})
    (path / "metrics.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
    evaluation(path / "eval/001", task, seed, successes=count, count=count)
    return path


def initialize_run(path, checkpoint):
    """Provide initialize run for the surrounding execution."""
    config = yaml.safe_load((path / "config.yaml").read_text())
    config["initial_checkpoint"] = str(checkpoint)
    (path / "config.yaml").write_text(yaml.safe_dump(config))
    header = json.loads((path / "run.json").read_text())
    header["config_sha256"] = hashlib.sha256((path / "config.yaml").read_bytes()).hexdigest()
    write_json(path / "run.json", header)
    with zipfile.ZipFile(checkpoint) as archive:
        initial_sha = json.loads(archive.read("metadata.json"))["sha256"]
    target = path / header["selected_checkpoint"]
    with zipfile.ZipFile(target) as archive:
        payload = archive.read("variables.msgpack")
        metadata = json.loads(archive.read("metadata.json"))
    metadata["config"] = {"experiment": config, "initialization": {"sha256": initial_sha}}
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("variables.msgpack", payload)


def run_collector(tmp_path, runs, **extra):
    """Provide run collector for the surrounding execution."""
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"runs": [str(path) for path in runs], **extra})
    prefix = tmp_path / "report"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(manifest), "--output-prefix", str(prefix)],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, json.loads(prefix.with_suffix(".json").read_text()) if (
        completed.returncode == 0
    ) else None


def rewrite_csv(path, change):
    """Provide rewrite csv for the surrounding execution."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    change(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_missing_paths_are_retained_and_all_18_cells_written(tmp_path):
    """Verify missing paths are retained and all 18 cells written."""
    result, report = run_collector(tmp_path, [tmp_path / "missing"])
    assert result.returncode == 0, result.stderr
    assert report["all_passed"] is False
    assert len(report["cells"]) == 18
    assert report["summary"] == {"missing": 18}
    assert report["runs"][0]["path"] == str(tmp_path / "missing")
    wall = report["runs"][0]["cost"]["wall_seconds"]
    assert wall["value"] is None and wall["reason"]
    assert (
        len(
            [
                line
                for line in (tmp_path / "report.md").read_text().splitlines()
                if line.startswith(("| ppo", "| apg", "| shac"))
            ]
        )
        == 18
    )


def test_outdated_run_layout_is_rejected_instead_of_interpreted(tmp_path):
    """The v2 evidence collector must not fall back to legacy selection or scene layouts."""
    path = make_run(tmp_path)
    header = json.loads((path / "run.json").read_text())
    header["layout_version"] = 1
    write_json(path / "run.json", header)
    completed, report = run_collector(tmp_path, [path])
    assert completed.returncode == 0, completed.stderr
    assert report["runs"][0]["identity"]["passed"] is False
    assert "Unsupported run layout" in " ".join(report["runs"][0]["issues"])


@pytest.mark.parametrize("change", [None, "seed", "algorithm"])
def test_resumed_checkpoint_preserves_training_configuration_checks(tmp_path, change):
    """Ignore the resume entrypoint while rejecting changed training seeds or algorithms."""
    path = make_run(tmp_path)
    header = json.loads((path / "run.json").read_text())
    checkpoint = path / header["selected_checkpoint"]
    with zipfile.ZipFile(checkpoint) as archive:
        payload = archive.read("variables.msgpack")
        metadata = json.loads(archive.read("metadata.json"))
    saved = metadata["config"]["experiment"]
    saved["resume"] = str(path / "checkpoints/latest.training.zip")
    if change == "seed":
        saved["seed"] = 1
    elif change == "algorithm":
        saved["learning"]["algorithm"] = "apg"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("variables.msgpack", payload)
    result, report = run_collector(tmp_path, [path])
    assert result.returncode == 0, result.stderr
    assert report["runs"][0]["checkpoint_config"]["passed"] is (change is None)


@pytest.mark.parametrize("mutation", ["short", "failed", "rmse", "duplicate", "holdout"])
def test_csv_overrides_claimed_success(tmp_path, mutation):
    """Verify csv overrides claimed success."""
    path = make_run(tmp_path)
    csv_path = path / "eval/001/episodes.csv"

    def change(rows):
        if mutation == "short":
            del rows[2:]
        elif mutation == "failed":
            for row in rows[:6]:
                row["event"] = "COLLISION"
        elif mutation == "rmse":
            rows[0]["position_rmse"] = "nan"
        elif mutation == "duplicate":
            rows[1]["world_index"] = rows[0]["world_index"]
        elif mutation == "holdout":
            for row in rows:
                row["initialization_seed"] = "1000100"

    rewrite_csv(csv_path, change)
    completed, report = run_collector(tmp_path, [path])
    assert completed.returncode == 0, completed.stderr
    run = report["runs"][0]
    assert report["cells"][0]["seeds"]["0"]["status"] != "passed"
    if mutation == "short":
        assert run["benchmark"]["scenes"]["empty"]["episodes"] == 2
    elif mutation == "failed":
        assert run["benchmark"]["scenes"]["empty"]["outcomes"]["COLLISION"] == 6
    elif mutation == "holdout":
        assert run["C5"]["passed"] is False
        assert run["C5"]["independent_seed_sets"] is False


def test_checkpoint_passes_must_be_consecutive(tmp_path):
    """Verify checkpoint passes must be consecutive."""
    path = make_run(tmp_path, passes=(True, False, True, True))
    _, report = run_collector(tmp_path, [path])
    assert report["runs"][0]["C5"]["passed"] is False
    assert report["runs"][0]["C5"]["first_passing_updates"] is None


def test_three_seeds_and_three_algorithms_with_actual_csv_pass_tracking_only(tmp_path):
    """Verify three seeds and three algorithms with actual csv pass tracking only."""
    runs = [
        make_run(tmp_path, f"{algorithm}-{seed}", algorithm=algorithm, seed=seed)
        for algorithm in ("ppo", "apg", "shac")
        for seed in range(3)
    ]
    completed, report = run_collector(tmp_path, runs)
    assert completed.returncode == 0, completed.stderr
    assert report["summary"] == {"passed": 3, "missing": 15}
    assert not report["all_passed"]
    assert report["runs"][0]["C5"]["first_passing_updates"] == [100, 200, 300]
    assert report["runs"][0]["checkpoint"]["archive_sha256"]


@pytest.mark.parametrize("duplicate", ["path", "seed_group", "seed_list"])
def test_duplicate_input_is_an_error(tmp_path, duplicate):
    """Verify duplicate input is an error."""
    first = make_run(tmp_path, "one")
    paths, extra = [first], {}
    if duplicate == "path":
        paths.append(first)
    elif duplicate == "seed_group":
        paths.append(make_run(tmp_path, "two"))
    else:
        extra["seeds"] = [0, 0, 1]
    result, report = run_collector(tmp_path, paths, **extra)
    assert result.returncode != 0 and report is None
    assert "Duplicate" in result.stderr or "distinct" in result.stderr


def test_mixed_navigation_grades_static_and_dynamic_separately(tmp_path):
    """Verify mixed navigation grades static and dynamic separately."""
    path = make_run(tmp_path, task="navigation")
    benchmark_report = path / "eval/001/report.json"
    recorded = json.loads(benchmark_report.read_text())
    recorded["task_contract"] = {"navigation_goal_radius_m": 0.5}
    write_json(benchmark_report, recorded)
    rewrite_csv(
        path / "eval/001/episodes.csv",
        lambda rows: [
            row.update(event="COLLISION")
            for row in [item for item in rows if item["scene"] == "D01"][:3]
        ],
    )
    header = json.loads((path / "run.json").read_text())
    header["status"] = "frozen_benchmark_failed"
    write_json(path / "run.json", header)
    _, report = run_collector(tmp_path, [path])
    static, dynamic = report["cells"][2:4]
    assert static["seeds"]["0"]["status"] == "passed"
    assert dynamic["seeds"]["0"]["status"] == "failed"
    assert report["runs"][0]["benchmark"]["scenes"]["S06"]["episodes"] == 25
    assert report["runs"][0]["benchmark"]["task_contract"] == recorded["task_contract"]


def test_racing_rejects_illegal_success_gate_order(tmp_path):
    """Verify racing rejects illegal success gate order."""
    path = make_run(tmp_path, task="racing")
    rewrite_csv(
        path / "eval/001/episodes.csv",
        lambda rows: rows[0].update(gate_order="1,2,3,4"),
    )
    _, report = run_collector(tmp_path, [path])
    assert report["cells"][1]["seeds"]["0"]["status"] == "failed"


def test_running_and_failed_records_are_not_dropped(tmp_path):
    """Verify running and failed records are not dropped."""
    paths = [make_run(tmp_path, f"seed-{seed}", seed=seed) for seed in (0, 1)]
    for path, status in zip(paths, ("running", "failed"), strict=True):
        header = json.loads((path / "run.json").read_text())
        header["status"] = status
        write_json(path / "run.json", header)
        header.pop("selection")
        write_json(path / "run.json", header)
    _, report = run_collector(tmp_path, paths)
    assert [r["recorded_status"] for r in report["runs"]] == ["running", "failed"]
    values = report["cells"][0]["seeds"]
    assert [values[str(seed)]["status"] for seed in range(3)] == ["incomplete", "failed", "missing"]


@pytest.mark.parametrize("stage", ["initializing", "benchmark"])
def test_active_run_with_pending_evidence_stays_incomplete(tmp_path, stage):
    """Verify active run with pending evidence stays incomplete."""
    source = make_run(tmp_path, "source")
    path = make_run(tmp_path, "active")
    initialize_run(path, source / "checkpoints/step-000300/policy.zip")
    header = json.loads((path / "run.json").read_text())
    header["status"] = "running"
    header.pop("selected_checkpoint")
    write_json(path / "run.json", header)
    (path / "eval/001/report.json").unlink()
    if stage == "initializing":
        header.pop("selection")
        write_json(path / "run.json", header)
        (path / "metrics.jsonl").write_text("")
    _, report = run_collector(tmp_path, [path])
    assert report["runs"][0]["recorded_status"] == "running"
    assert report["cells"][0]["seeds"]["0"]["status"] == "incomplete"
    assert report["all_passed"] is False


def test_shared_initialization_cost_is_counted_once(tmp_path):
    """Verify shared initialization cost is counted once."""
    source = make_run(tmp_path, "diagnostic", task="racing", algorithm="apg")
    header = json.loads((source / "run.json").read_text())
    header.update(python="3.13.15", versions={"jax": "0.9.2"})
    write_json(source / "run.json", header)
    cp_path = source / "checkpoints/step-000300/policy.zip"
    paths = [make_run(tmp_path, f"formal-{seed}", task="racing", seed=seed) for seed in range(3)]
    for path in paths:
        initialize_run(path, cp_path)
    _, report = run_collector(tmp_path, paths)
    assert len(report["runs"]) == 3
    assert list(report["shared_initialization_costs"]) == [str(source)]
    shared = report["shared_initialization_costs"][str(source)]
    assert shared["cost"]["wall_seconds"]["value"] == 10
    assert len(shared["used_by"]) == 3
    assert shared["run_record"]["python"] == "3.13.15"
    assert all(r["initialization"]["integrity"]["passed"] for r in report["runs"])


def test_transitive_initialization_costs_and_unknown_historical_cost(tmp_path):
    """Verify transitive initialization costs and unknown historical cost."""
    reference = tmp_path / "reference_initialization" / "adapted.policy.zip"
    reference.parent.mkdir()
    payload = b"adapted-historical-parameters"
    provenance = {
        "source_archive": "research/checkpoints/weights/historical.pkl",
        "source_sha256": hashlib.sha256(b"historical-weights").hexdigest(),
        "source_physics": "historical point_mass_lag; not Crazyflow",
        "transfer": "Approximate initialization; current Actor has a bias-free tanh head",
    }
    with zipfile.ZipFile(reference, "w") as archive:
        archive.writestr("variables.msgpack", payload)
        archive.writestr(
            "metadata.json",
            json.dumps(
                {
                    "format_version": 2,
                    "purpose": "inference",
                    "kind": "depth",
                    "actor": {"kind": "depth", "action_size": 3, "hidden_size": 192},
                    "config": {"experiment": {"initial_checkpoint": None}},
                    "provenance": provenance,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ),
        )
    altitude = make_run(tmp_path, "altitude", task="navigation", algorithm="apg")
    mixed = make_run(tmp_path, "mixed", task="navigation", algorithm="apg")
    initialize_run(altitude, reference)
    initialize_run(mixed, altitude / "checkpoints/step-000300/policy.zip")
    # Full source-run cost can exceed the cost at the consumed checkpoint.
    for path, updates, wall in ((altitude, 500, 17.5), (mixed, 700, 41.25)):
        header = json.loads((path / "run.json").read_text())
        header.update(status="interrupted", updates=updates, wall_seconds=wall)
        write_json(path / "run.json", header)
    paths = [
        make_run(
            tmp_path,
            f"formal-{algorithm}-{seed}",
            task="navigation",
            algorithm=algorithm,
            seed=seed,
        )
        for algorithm in ("ppo", "apg", "shac")
        for seed in range(3)
    ]
    for path in paths:
        initialize_run(path, mixed / "checkpoints/step-000300/policy.zip")
    completed, report = run_collector(tmp_path, paths)
    assert completed.returncode == 0, completed.stderr
    assert len(report["runs"]) == 9
    shared = report["shared_initialization_costs"]
    lineage = list(map(str, (mixed, altitude, reference)))
    assert list(shared) == lineage
    assert [shared[str(p)]["cost"]["updates"]["value"] for p in (mixed, altitude)] == [700, 500]
    assert sum(shared[str(p)]["cost"]["wall_seconds"]["value"] for p in (mixed, altitude)) == 58.75
    for entry in shared.values():
        assert entry["used_by"] == list(map(str, paths))
        assert not entry["issues"]
    for run in report["runs"]:
        assert run["initialization"]["cost_lineage"] == lineage
        assert run["initialization"]["shared_cost_id"] == str(mixed)
    for child, parent in ((mixed, altitude), (altitude, reference)):
        initial = shared[str(child)]["initialization"]
        assert initial["shared_cost_id"] == str(parent)
        assert initial["evidence"]["integrity"]["passed"] is True
    historical = shared[str(reference)]
    assert historical["kind"] == "reference_archive"
    assert historical["reference_archive"]["metadata"]["provenance"] == provenance
    assert (
        historical["reference_archive"]["archive_sha256"]
        == hashlib.sha256(reference.read_bytes()).hexdigest()
    )
    assert historical["initialization"]["shared_cost_id"] is None
    for key in ("updates", "interactions", "wall_seconds", "training_wall_seconds"):
        value = historical["cost"][key]
        assert value["value"] is None
        assert "Historical training cost unavailable" in value["reason"]
        assert value["source"] == f"{reference}:metadata.provenance"
    markdown = (tmp_path / "report.md").read_text()
    assert all(f"`{path}`" in markdown for path in (mixed, altitude, reference))
    assert "historical training cost **unavailable**" in markdown
    assert "used by 9 formal runs through initialization ancestry" in markdown
    assert all(value in markdown for value in provenance.values())


def test_cyclic_initialization_is_reported_without_double_counting(tmp_path):
    """Verify cyclic initialization is reported without double counting."""
    source = make_run(tmp_path, "source", task="racing", algorithm="apg")
    checkpoint = source / "checkpoints/step-000300/policy.zip"
    initialize_run(source, checkpoint)
    formal = make_run(tmp_path, "formal", task="racing")
    initialize_run(formal, checkpoint)
    completed, report = run_collector(tmp_path, [formal])
    assert completed.returncode == 0, completed.stderr
    assert report["runs"][0]["initialization"]["cost_lineage"] == [str(source)]
    shared = report["shared_initialization_costs"][str(source)]
    assert shared["used_by"] == [str(formal)]
    assert "Cyclic initialization lineage" in shared["issues"][0]


def test_saved_physics_change_cannot_pass(tmp_path):
    """Verify saved physics change cannot pass."""
    path = make_run(tmp_path)
    config = yaml.safe_load((path / "config.yaml").read_text())
    config["simulation"]["physics_hz"] = 100
    (path / "config.yaml").write_text(yaml.safe_dump(config))
    _, report = run_collector(tmp_path, [path])
    assert report["runs"][0]["physics_config"]["passed"] is False
    assert report["cells"][0]["seeds"]["0"]["status"] == "failed"
