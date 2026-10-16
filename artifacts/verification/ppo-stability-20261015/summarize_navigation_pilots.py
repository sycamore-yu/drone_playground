"""Recompute complete pilot screens and case retention from preserved raw CSVs."""

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCENES = ("S01", "S02", "S03", "D01", "D02", "D03")
CASE_FIELDS = ("scene", "geometry_sha256", "initialization_seed", "world_index")


def digest(path):
    """Hash the raw artifact whose contents support a reported result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(run):
    """Ignore incomplete evaluations and retain every observed successful case."""
    history, probes, skipped = [], [], []
    previous_successes, ever_successes = set(), set()
    cohort = None
    for directory in sorted((run / "checkpoints").glob("model_*")):
        metadata = json.loads((directory / "metadata.json").read_text())
        update = metadata["provenance"]["updates"]
        interactions = metadata["provenance"]["interactions"]
        for kind in ("screening", "training_distribution"):
            report_path = directory / kind / "report.json"
            csv_path = directory / kind / "episodes.csv"
            if not report_path.exists():
                if csv_path.exists():
                    skipped.append(str(csv_path.relative_to(ROOT)))
                continue
            report = json.loads(report_path.read_text())
            if not report.get("complete"):
                skipped.append(str(report_path.relative_to(ROOT)))
                continue
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            cases = {tuple(row[k] for k in CASE_FIELDS): row for row in rows}
            assert len(cases) == len(rows), csv_path
            scenes = SCENES if kind == "screening" else ("train/procedural",)
            assert set(report["reports"]) == set(scenes), report_path
            by_scene = {}
            for scene in scenes:
                selected = [r for r in rows if r["scene"] == scene]
                successes = sum(r["event"] == "SUCCESS" for r in selected)
                assert len(selected) == report["reports"][scene]["episodes"] == 8
                assert successes == report["reports"][scene]["successes"]
                by_scene[scene] = {
                    "successes": successes,
                    "episodes": len(selected),
                    "events": dict(Counter(r["event"] for r in selected)),
                    "mean_best_goal_progress": statistics.mean(
                        float(r["normalized_best_goal_progress"]) for r in selected
                    ),
                    "mean_final_goal_progress": statistics.mean(
                        float(r["normalized_final_goal_progress"]) for r in selected
                    ),
                }
            successes = {key for key, row in cases.items() if row["event"] == "SUCCESS"}
            entry = {
                "update": update,
                "interactions": interactions,
                "successes": len(successes),
                "episodes": len(rows),
                "per_scene": by_scene,
                "csv": str(csv_path.relative_to(ROOT)),
                "csv_sha256": digest(csv_path),
                "report_sha256": digest(report_path),
                "training_state_retained": (directory / "training/commit_success.txt").is_file(),
            }
            if kind == "training_distribution":
                probes.append(entry)
                continue
            assert len(rows) == 48
            assert cohort is None or set(cases) == cohort, csv_path
            cohort = set(cases)
            entry.update(
                retained_previous_successes=len(successes & previous_successes),
                previous_successes=len(previous_successes),
                lost_previous_cases=sorted(previous_successes - successes),
                new_success_cases=sorted(successes - ever_successes),
                historical_successes=len(ever_successes),
                retained_historical_successes=len(successes & ever_successes),
                historical_retention=(
                    len(successes & ever_successes) / len(ever_successes)
                    if ever_successes
                    else None
                ),
            )
            previous_successes = successes
            ever_successes |= successes
            history.append(entry)
    return {
        "config_sha256": digest(run / "config.yaml"),
        "complete_screens": history,
        "training_distribution_probes": probes,
        "incomplete_evidence_skipped": skipped,
    }


def summarize_branch(directory):
    """Compare complete fixed-case checks against the actual fork's source successes."""
    experiment = json.loads((directory / "experiment.json").read_text())
    source = Path(experiment["checkpoint"]) / "screening/episodes.csv"
    with source.open(newline="") as stream:
        baseline = {tuple(row[k] for k in CASE_FIELDS): row for row in csv.DictReader(stream)}
    assert len(baseline) == 48
    original_successes = {key for key, row in baseline.items() if row["event"] == "SUCCESS"}
    assert len(original_successes) == 21
    origin = experiment["source_provenance"]["updates"]
    branch = experiment["branch_config"]
    batch = branch["simulation"]["num_envs"] * branch["learning"]["options"]["horizon"]
    checks = []
    for report_path in sorted(directory.glob("update-*/report.json")):
        report = json.loads(report_path.read_text())
        if not report.get("complete"):
            continue
        csv_path = report_path.parent / "episodes.csv"
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        cases = {tuple(row[k] for k in CASE_FIELDS): row for row in rows}
        assert len(rows) == 48 and cases.keys() == baseline.keys()
        successes = {key for key, row in cases.items() if row["event"] == "SUCCESS"}
        per_scene = {scene: sum(key[0] == scene for key in successes) for scene in SCENES}
        for scene in SCENES:
            assert report["reports"][scene]["episodes"] == 8
            assert report["reports"][scene]["successes"] == per_scene[scene]
        update = int(report_path.parent.name.split("-")[1])
        if update == origin:
            assert {k: r["event"] for k, r in cases.items()} == {
                k: r["event"] for k, r in baseline.items()
            }
        checks.append(
            {
                "update": update,
                "additional_interactions": (update - origin) * batch,
                "successes": len(successes),
                "retained_source_successes": len(successes & original_successes),
                "retention_fraction": len(successes & original_successes) / len(original_successes),
                "new_successes": len(successes - original_successes),
                "per_scene_successes": per_scene,
                "events": dict(Counter(row["event"] for row in rows)),
                "lost_cases": sorted(original_successes - successes),
                "csv_sha256": digest(csv_path),
                "report_sha256": digest(report_path),
            }
        )
    return {
        "experiment": experiment,
        "source_csv_sha256": digest(source),
        "complete_checks": checks,
    }


if __name__ == "__main__":
    runs = [
        *(ROOT / "results/goal_reaching_pilot").glob("*_s0"),
        *(ROOT / "results/goal_reaching_batch64").glob("*_s0"),
        *(ROOT / "results/goal_reaching_differentiable").glob("*_s0"),
    ]
    result = {
        "scope": "Development only. Historical success sets include retained CSVs only; "
        "early pilot CSVs deleted before the retention fix are unavailable. "
        "No missing case outcomes are reconstructed from aggregate logs.",
        "runs": {str(run.relative_to(ROOT)): summarize(run) for run in sorted(runs)},
        "branches": {
            name: summarize_branch(Path(__file__).parent / name)
            for name in (
                "depth-clearance-control",
                "depth-clearance-minimum",
                "depth-deterministic-control",
                "depth-deterministic-update",
                "depth-rehearsal",
                "depth-h512-lambda095",
                "depth-h512-lambda1",
            )
            if (Path(__file__).parent / name / "experiment.json").exists()
        },
    }
    assert len({b["experiment"]["restored_state_sha256"] for b in result["branches"].values()}) <= 1
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    for name, run in result["runs"].items():
        print(
            name,
            [
                (
                    x["interactions"],
                    x["successes"],
                    x["retained_previous_successes"],
                    x["previous_successes"],
                )
                for x in run["complete_screens"]
            ],
        )
    for name, branch in result["branches"].items():
        print(
            name,
            [
                (row["update"], row["successes"], row["retained_source_successes"])
                for row in branch["complete_checks"]
            ],
        )
