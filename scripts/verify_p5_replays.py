#!/usr/bin/env python3
"""Read one representative replay in every completed P5 cell with the shipped viewer."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="v2")
    parser.add_argument("--reader", type=Path,
                        default=ROOT.parents[2] / "opensource_projects/rscope-vscode/python/bridge.py")
    args = parser.parse_args()
    target = ROOT / "docs/verification" / ("p5-results-" + args.revision)
    summary = json.loads((target / "summary.json").read_text())
    command = [sys.executable, str(ROOT / "scripts/verify_phase_replays.py"),
               "--reader", str(args.reader), "--output", str(target / "replays.json")]
    selected = []
    archive_episodes = archive_frames = 0
    outcome_names = {0: "timeout", 1: "arrived", 2: "collision", 3: "out_of_bounds",
                     4: "numerical_failure", 5: "timeout"}
    for cell in summary["cells"]:
        if cell["status"] != "completed":
            continue
        run = ROOT / "experiments" / cell["run_id"]
        report = json.loads((run / "eval/report.json").read_text())
        episodes = report["cells"][cell["difficulty"]]["episodes"]
        if cell.get("all_case_archive_verified"):
            with np.load(ROOT / cell["archive"], allow_pickle=False) as archive:
                offsets = archive["offsets"]
                if offsets[0] != 0 or len(offsets) != len(episodes) + 1:
                    raise RuntimeError(f"Archive offset/count mismatch: {cell['archive']}")
                for episode in episodes:
                    case_id = episode["case"]
                    start, stop = offsets[case_id:case_id + 2]
                    if (archive["case_ids"][case_id] != case_id
                            or archive["scenario_ids"][case_id] != episode["scenario_id"]
                            or stop - start != episode["steps"]
                            or outcome_names[int(archive["outcome"][stop - 1])] != episode["outcome"]):
                        raise RuntimeError(f"Archive episode identity/event mismatch: {cell['archive']}/{case_id}")
                    np.testing.assert_allclose(archive["metric_goal_distance"][stop - 1],
                                               episode["final_goal_distance_m"], atol=1e-6)
                    np.testing.assert_allclose(archive["metric_clearance"][start:stop].min(),
                                               episode["min_clearance_m"], atol=1e-6)
                archive_episodes += len(episodes)
                archive_frames += int(offsets[-1])
        case = next((x for x in episodes if not x["arrived"]), episodes[0])
        directory = run / "rollouts" / cell["difficulty"] / f"case-{case['case']:03d}"
        command += ["--directory", str(directory)]
        selected.append(dict(run_id=cell["run_id"], task=cell["task"], split=cell["split"],
                             difficulty=cell["difficulty"], case=case["case"],
                             scenario_id=case["scenario_id"], outcome=case["outcome"],
                             episode_steps=case["steps"],
                             directory=str(directory.relative_to(ROOT))))
    if not selected:
        raise SystemExit("No completed cell replays yet")
    subprocess.run(command, check=True)
    result = json.loads((target / "replays.json").read_text())
    for cell in selected:
        records = [x for x in result["records"] if str(Path(x["file"]).parent) == cell["directory"]]
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one replay for {cell['directory']}")
        cell["padding_frames"] = records[0]["frames"] - cell["episode_steps"]
        if cell["padding_frames"] < 0 or (cell["task"] == "dynamic" and cell["padding_frames"]):
            raise RuntimeError(f"Replay terminal time mismatch: {cell}")
    result.update(matrix_complete=summary["matrix_complete"], selected_cells=selected,
                  all_case_archive_episodes=archive_episodes, all_case_archive_frames=archive_frames,
                  coverage="First failure in each cell, or first successful episode if all arrived")
    if summary["matrix_complete"] and archive_episodes != 5760:
        raise RuntimeError("Complete matrix requires all 5760 archived evaluation trajectories")
    (target / "replays.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(verified_cells=len(selected), matrix_complete=summary["matrix_complete"])))


if __name__ == "__main__":
    main()
