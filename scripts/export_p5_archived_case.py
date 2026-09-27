#!/usr/bin/env python3
"""Open any archived evaluation case in the existing RScope format, without reflight."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import numpy as np

from drone_playground.composition import build_environment
from drone_playground.evaluation.trace_archive import load_navigation_case
from drone_playground.runs.navigation_scene import (
    active_indices,
    create_replay_model,
    obstacle_track,
)
from drone_playground.runs.rscope_io import export_rollout

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("case", type=int)
    args = parser.parse_args()
    run = ROOT / "experiments" / args.run_id
    config = json.loads((run / "manifest.json").read_text())["config"]
    trace, scenario = load_navigation_case(run / "traces", args.difficulty, args.case)
    env = build_environment(config, "cpu", config["evaluation"]["split"],
                            int(config["evaluation"]["episodes"]))
    try:
        active = active_indices(env.bank, scenario)
        trace["obstacle_pos"] = obstacle_track(env.bank, scenario, np.asarray(trace["time"])[:, 0])[
            :, active][:, None]
        target = run / "rollouts-from-archive" / args.difficulty / f"case-{args.case:03d}"
        path = export_rollout(create_replay_model(env, scenario), target, trace)
        print(path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
