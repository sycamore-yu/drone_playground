#!/usr/bin/env python3
"""Read one representative replay in every completed P5 cell with the shipped viewer."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
    for cell in summary["cells"]:
        if cell["status"] != "completed":
            continue
        run = ROOT / "experiments" / cell["run_id"]
        report = json.loads((run / "eval/report.json").read_text())
        episodes = report["cells"][cell["difficulty"]]["episodes"]
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
                  coverage="First failure in each cell, or first successful episode if all arrived")
    (target / "replays.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(verified_cells=len(selected), matrix_complete=summary["matrix_complete"])))


if __name__ == "__main__":
    main()
