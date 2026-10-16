"""Recompute paired diagnostic outcomes and the recorded reward-order counterexample."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episodes(path):
    rows = list(csv.DictReader(path.open()))
    keys = ("scene", "geometry_sha256", "initialization_seed", "world_index")
    result = {tuple(row[key] for key in keys): row for row in rows}
    assert len(result) == len(rows) == 32
    return result


root = Path(__file__).resolve().parent
paired = {}
origin_hashes = set()
for name in ("lidar-control", "lidar-memory", "lidar-critic-frozen", "lidar-quarter-lr"):
    directory = root / name
    experiment = json.loads((directory / "experiment.json").read_text())
    assert experiment["status"] == "completed" and experiment["final_update"] == 2880
    origin_hashes.add(experiment["restored_state_sha256"])
    baseline = episodes(directory / "update-002640/episodes.csv")
    old_successes = {key for key, row in baseline.items() if row["event"] == "SUCCESS"}
    assert len(old_successes) == 29
    checkpoints = []
    for update in (2640, 2700, 2760, 2820, 2880):
        path = directory / f"update-{update:06d}/episodes.csv"
        rows = episodes(path)
        assert rows.keys() == baseline.keys()
        successes = {key for key, row in rows.items() if row["event"] == "SUCCESS"}
        checkpoints.append({
            "update": update,
            "additional_interactions": (update - 2640) * 512,
            "successes": len(successes),
            "retained_source_successes": len(successes & old_successes),
            "retention_fraction": len(successes & old_successes) / len(old_successes),
            "new_successes": len(successes - old_successes),
            "events": dict(Counter(row["event"] for row in rows.values())),
            "per_scene_successes": {
                scene: sum(key[0] == scene for key in successes)
                for scene in ("S01", "D01", "D02", "D03")
            },
            "lost_cases": [list(key) for key in sorted(old_successes - successes)],
            "csv_sha256": digest(path),
        })
    paired[name] = {"arguments": experiment["arguments"], "evaluations": checkpoints}
assert len(origin_hashes) == 1

calibration = Path(
    "/home/tong/tongworkspace/simulation_dev/new_drone_playground/results/"
    "navigation_stability_fixes_v01/critic-calibration/lidar_distance_curriculum_s0"
)
metadata = json.loads((calibration / "D01.json").read_text())
assert metadata[0]["initial_state_sha256"] == metadata[1]["initial_state_sha256"]
ranking = []
for update in (2640, 4320):
    path = calibration / f"D01_{update:06d}.npz"
    with np.load(path) as data:
        active = data["active"][:, 4].astype(bool)
        rewards = data["reward"][active, 4]
        row = next(row for row in metadata if row["update"] == update)["episodes"][4]
        actual = float(np.dot(0.995 ** np.arange(len(rewards)), rewards))
        np.testing.assert_allclose(actual, row["initial_realized_return"], atol=1e-8)
        ranking.append({
            "update": update, "event": row["event"], "seconds": row["seconds"],
            "gamma": 0.995, "original_discounted_return": actual,
            "remaining_metres": row["remaining"], "raw_sha256": digest(path),
        })
assert ranking[0]["event"] == "SUCCESS" and ranking[1]["event"] == "COLLISION"
assert ranking[0]["original_discounted_return"] < ranking[1]["original_discounted_return"]
result = {
    "scope": "development diagnostics; not convergence or formal acceptance",
    "paired_restored_state_sha256": next(iter(origin_hashes)),
    "paired_continuations": paired,
    "reward_order_counterexample": {
        "scene": "D01", "seed": 1000003, "world_index": 4,
        "initial_state_sha256": metadata[0]["initial_state_sha256"],
        "records": ranking,
        "limitation": "two realized stochastic trajectories; not expected policy returns or a unique cause",
        "proposed_goal_reward_success_minus_failure_lower_bound": (
            2.0 + (ranking[1]["remaining_metres"] - ranking[0]["remaining_metres"]) / 20.0 - 0.5
        ),
        "bound_scope": "analytic full-episode undiscounted bound, not a replay or training result",
    },
}
(root / "paired-diagnostic-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({name: [row["successes"] for row in run["evaluations"]]
                  for name, run in paired.items()}))
print(json.dumps(result["reward_order_counterexample"]))
