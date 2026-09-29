"""Validate the explicit 30-cell acceptance selection against immutable run evidence.

Run from the repository root. This reads local trusted experiment artifacts and
writes only the explicitly selected summary file, never a source run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def episodes(report: dict) -> list[dict]:
    if "episodes" in report:
        return report["episodes"]
    return [row for cell in report.get("cells", {}).values() for row in cell["episodes"]]


def validated_outcomes(
    result: dict, report: dict, *, native: bool = False, scene_prefix: str | None = None
) -> dict:
    if result.get("status") != "completed":
        raise ValueError("Evaluation has no completed result")
    rows = episodes(report)
    if len(rows) != report["num_trials"] or not rows:
        raise ValueError("Missing or inconsistent evaluation denominator")
    navigation = "arrived" in report
    keys = (
        ("arrived", "collision", "out_of_bounds", "numerical_failure", "timeout")
        if navigation
        else ("completed",)
    )

    def counts(items):
        return {
            key: sum(
                row.get("outcome") == key if navigation and "outcome" in row else bool(row.get(key))
                for row in items
            )
            for key in keys
        }

    totals = counts(rows)
    if navigation and sum(totals.values()) != len(rows):
        raise ValueError("Every navigation trial must have one terminal outcome")
    for key, count in totals.items():
        if key in report and report[key] != count:
            raise ValueError(f"Aggregate {key} differs from raw episodes")
    if scene_prefix:
        rows = [row for row in rows if row["scene_id"].startswith(scene_prefix)]
        if not rows:
            raise ValueError("Requested scene family has no episodes")
        totals = counts(rows)
    executed = True
    if native:
        diagnostics = report.get("diagnostics", [])
        executed = len(diagnostics) == report["num_trials"] and all(
            row.get("commands", 0) > 0 and row.get("trajectories", 0) > 0 for row in diagnostics
        )
    quality = report.get("quality_passed")
    return dict(
        num_trials=len(rows),
        **totals,
        execution_passed=executed,
        quality_passed=quality if executed else False,
        rmse_all_mean_m=report.get("rmse_all_mean"),
        gates_passed_mean=report.get("gates_passed_mean"),
    )


def collect(root: Path, selections: list[dict]) -> dict:
    wanted = {
        (method, task)
        for method in ("ppo", "bptt", "shac", "pointcloud", "super", "ego_planner")
        for task in ("hovering", "tracking", "racing", "navigation/static", "navigation/dynamic")
    }
    actual = [(item["method"], item["task"]) for item in selections]
    if set(actual) != wanted or len(actual) != len(wanted):
        raise ValueError("Acceptance requires exactly the 30 distinct registered cells")
    collected = []
    for item in selections:
        run = root / "experiments" / item["run_id"]
        report, result, config = (
            read_json(run / path)
            for path in ("eval/report.json", "result.json", "resolved-config.json")
        )
        native = item["method"] in ("super", "ego_planner")
        outcome = validated_outcomes(
            result, report, native=native, scene_prefix=item.get("scene_prefix")
        )
        evidence = {}
        for name in (
            "eval/report.json",
            "result.json",
            "resolved-config.json",
            "manifest.json",
            "code.patch",
            "command.txt",
        ):
            path = run / name
            evidence[str(path.relative_to(root))] = sha256(path)
        manifest = read_json(run / "manifest.json")
        if manifest["code"]["patch_sha256"] != sha256(run / "code.patch"):
            raise ValueError(f"Changed source patch in {run}")
        entry = dict(
            item,
            **outcome,
            evidence=evidence,
            command=(run / "command.txt").read_text().strip(),
            task_config=config["env"]["task"],
            execution=config["env"]["execution"],
            runtime=config["runtime"],
            network=config.get("network"),
            sensor=config["env"]["sensor"],
            protocol=manifest.get("benchmark_protocol"),
        )
        checkpoint_name = report.get("checkpoint")
        if not native:
            if not report.get("parameters_frozen") or not checkpoint_name:
                raise ValueError(f"Missing frozen-policy evidence in {run}")
            checkpoint = Path(checkpoint_name)
            if not checkpoint.is_absolute():
                checkpoint = root / checkpoint
            metadata = read_json(checkpoint.with_suffix(".json"))
            digest = sha256(checkpoint)
            if digest != metadata["sha256"]:
                raise ValueError(f"Checkpoint authentication failed: {checkpoint}")
            step = metadata.get("step", metadata.get("updates"))
            if step is None or step <= 0:
                raise ValueError("Acceptance must use a trained checkpoint, including warm starts")
            entry["checkpoint"] = dict(
                path=str(checkpoint.relative_to(root)),
                sha256=digest,
                selected_step_or_updates=step,
                parameter_sha256=report.get("parameter_sha256"),
            )
        source = item.get("training_run")
        if source:
            train = root / "experiments" / source
            trained, train_cfg = (
                read_json(train / "result.json"),
                read_json(train / "resolved-config.json"),
            )
            if trained["status"] != "completed" or trained.get("actor_parameter_delta_l2", 0) <= 0:
                raise ValueError(f"No authenticated real training update in {source}")
            if not checkpoint.is_relative_to(train):
                raise ValueError("Selected checkpoint and training-run identities disagree")
            entry["training"] = dict(
                run_id=source,
                actual_steps=trained["actual_steps"],
                actor_parameter_delta_l2=trained["actor_parameter_delta_l2"],
                elapsed_s=trained["elapsed_s"],
                trainer_metrics=trained.get("trainer_metrics"),
                warm_start=train_cfg["training"].get("warm_start"),
                command=(train / "command.txt").read_text().strip(),
                runtime=train_cfg["runtime"],
                algorithm=train_cfg["algorithm"],
                network=train_cfg["network"],
                result_sha256=sha256(train / "result.json"),
            )
        files = sorted((run / "rollouts").rglob("*.mj_unroll"))
        if not files:
            raise ValueError(f"Missing replay for {run}")
        entry["replays"] = [
            dict(path=str(path.relative_to(root)), sha256=sha256(path)) for path in files
        ]
        entry["replay_coverage"] = (
            "all" if len(files) == report["num_trials"] else "fixed cases plus worst case"
        )
        for file in files:
            for name in ("scene.xml", "rscope_meta.pkl"):
                path = file.parent / name
                if not path.exists():
                    raise ValueError(f"Missing replay companion {path}")
        collected.append(entry)
    return dict(
        schema_version=1,
        cells=collected,
        cell_count=len(collected),
        all_results_present=True,
        all_execution_passed=all(row["execution_passed"] for row in collected),
        control_learning_quality_passed=all(
            row["quality_passed"]
            for row in collected
            if row["method"] not in ("super", "ego_planner")
            and not row["task"].startswith("navigation/")
        ),
        notes=[
            "Quality and executed interfaces are independently reported.",
            "Control transfer and original point-cloud reconstruction are distinct recipes.",
            "SHAC racing uses BPTT warm start followed by real SHAC parameter updates.",
            "Original point-cloud reconstruction is independent of this baseline matrix.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.is_relative_to(root / "experiments") or output == args.selection.resolve():
        raise ValueError("Summary output must be separate from source experiments and selection")
    summary = collect(root, read_json(args.selection)["cells"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "cells"}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
