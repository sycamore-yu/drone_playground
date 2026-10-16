"""Cache new scratch screens and compare preserved case outcomes by interactions."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "results/fixed_scenes_parallel_worlds256/depth_ppo_s0"
OUTPUT = HERE / "depth-fixed-scenes-worlds256-scratch-progress.json"
SCENES = ("S01", "S02", "S03", "D01", "D02", "D03")
CASE_FIELDS = ("scene", "geometry_sha256", "initialization_seed", "world_index")


def successes(events):
    """Return complete case identities for successful episodes."""
    return {tuple(row[:4]) for row in events if row[4] == "SUCCESS"}


def main():
    """Read each new screen once; old control data come from its existing cache."""
    control = json.loads((HERE / "depth-fixed-scenes-parallel-progress.json").read_text())
    controls = {row["interactions"]: row for row in control["checks"]}
    result = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {"checks": []}
    cached = {row["update"] for row in result["checks"]}
    previous = successes(result["checks"][-1]["case_events"]) if cached else set()
    historical = set().union(*(successes(row["case_events"]) for row in result["checks"]))
    for directory in sorted((RUN / "checkpoints").glob("model_*")):
        update = int(directory.name.split("_")[1])
        report_path = directory / "screening/report.json"
        if update in cached or not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        if not report.get("complete"):
            continue
        assert set(report["reports"]) == set(SCENES)
        with (report_path.parent / "episodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        cases = {tuple(row[key] for key in CASE_FIELDS): row["event"] for row in rows}
        assert len(cases) == len(rows) == 48
        assert set(cases) == {tuple(row[:4]) for row in control["checks"][0]["case_events"]}
        events = [[*key, value] for key, value in sorted(cases.items())]
        current = successes(events)
        metadata = json.loads((directory / "metadata.json").read_text())["provenance"]
        assert metadata["updates"] == update
        assert metadata["interactions"] == update * 256 * 512
        per_scene = {}
        for scene in SCENES:
            selected = [row for row in rows if row["scene"] == scene]
            count = sum(row["event"] == "SUCCESS" for row in selected)
            assert len(selected) == report["reports"][scene]["episodes"] == 8
            assert count == report["reports"][scene]["successes"]
            per_scene[scene] = {
                "successes": count,
                "episodes": 8,
                "events": dict(Counter(row["event"] for row in selected)),
                "median_flight_seconds": statistics.median(
                    float(row["flight_seconds"]) for row in selected
                ),
            }
        matched = controls.get(metadata["interactions"])
        comparison = None
        if matched:
            assert set(cases) == {tuple(row[:4]) for row in matched["case_events"]}
            baseline = successes(matched["case_events"])
            assert len(baseline) == matched["successes"]
            comparison = {
                "control_update": matched["update"],
                "control_successes": len(baseline),
                "retained_control_successes": len(current & baseline),
                "lost_control_cases": sorted(baseline - current),
                "new_vs_control_cases": sorted(current - baseline),
            }
        result["checks"].append(
            {
                "update": update,
                "interactions": metadata["interactions"],
                "successes": len(current),
                "per_scene": per_scene,
                "previous_successes": len(previous),
                "retained_previous_successes": len(current & previous),
                "lost_previous_cases": sorted(previous - current),
                "historical_successes": len(historical),
                "retained_historical_successes": len(current & historical),
                "new_success_cases": sorted(current - historical),
                "case_events": events,
                "matched_control": comparison,
                "report": str(report_path.relative_to(ROOT)),
                "training_state_retained": (directory / "training/commit_success.txt").is_file(),
                "session_wall_seconds": metadata["session_wall_seconds"],
            }
        )
        previous = current
        historical |= current
    run = json.loads((RUN / "run.json").read_text())
    result.update(
        scope="Development screening only. Native CSVs lack goal-progress fields; "
        "no missing progress is inferred. Earlier complete screens use their cached case records.",
        run=str(RUN),
        runtime_commit=run["git_commit"],
        source_sha256=run["source_sha256"],
        config_sha256=run["config_sha256"],
        run_status={key: run.get(key) for key in ("status", "last_update", "last_interactions")},
    )
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["checks"]:
        print(
            row["update"],
            row["interactions"],
            row["successes"],
            [row["per_scene"][scene]["successes"] for scene in SCENES],
            "retained",
            row["retained_previous_successes"],
            "/",
            row["previous_successes"],
        )


if __name__ == "__main__":
    main()
