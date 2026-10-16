"""Measure the existing altitude controller with a frozen PPO horizontal policy."""

import csv
import json
from collections import Counter
from pathlib import Path

import orbax.checkpoint as ocp
from omegaconf import OmegaConf

from drone_playground.simulation.evaluation import evaluate
from drone_playground.simulation.networks import Actor
from drone_playground.simulation.policy import read_checkpoint
from drone_playground.simulation.records import atomic_json


class HorizontalActor:
    """Retain learned horizontal requests and the full original recurrent network."""

    def __init__(self, actor):
        """Keep the frozen parameter layout and observation contract."""
        self.actor = actor

    def initialize_memory(self, batch):
        """Use the same zero episode-start state."""
        return self.actor.initialize_memory(batch)

    def apply(self, parameters, observation, memory):
        """The existing Action computes vertical PD control from true goal height."""
        raw, memory, auxiliary = self.actor.apply(parameters, observation, memory)
        assert raw.shape[-1] == 3
        return raw[..., :2], memory, auxiliary


def main():
    """Reuse the complete control and evaluate one intentional control change."""
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    source = root / "results/fixed_scenes_long_return/depth_ppo_s0/checkpoints/model_000032"
    output = base / "depth-altitude-counterfactual"
    cached = json.loads((base / "fixed-scenes-long-return-progress.json").read_text())
    control = next(row for row in cached["checks"] if row["update"] == 32)
    expected = {tuple(row[:4]): row[-1] for row in control["case_events"]}
    assert len(expected) == 48 and control["successes"] == 0
    _, metadata = read_checkpoint(source, "training")
    with ocp.StandardCheckpointer() as checkpointer:
        parameters = checkpointer.restore(source / "training")["params"]
    actor = Actor(**metadata["actor"])
    config = OmegaConf.create(metadata["config"]["experiment"])
    assert metadata["provenance"]["config_sha256"] == cached["config_sha256"]
    assert config.method.action.level == "acceleration"
    config.method.action.level = "horizontal_acceleration"
    config.mode = "checkpoint_eval"
    config.benchmark.protocol = "diagnostic"
    config.benchmark.episodes = 8
    config.benchmark.scenes = ["S01", "S02", "S03", "D01", "D02", "D03"]
    config.benchmark.seed_base = 1_000_000
    config.benchmark.record_replay = False
    config.benchmark.record_trajectories = True
    assert OmegaConf.to_container(config.task) == metadata["config"]["experiment"]["task"]
    assert config.task.duration == 300
    output.mkdir(exist_ok=False)
    report = evaluate(
        config, output / "altitude_pd", actor=HorizontalActor(actor), parameters=parameters
    )
    with (output / "altitude_pd/episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fields = ("scene", "geometry_sha256", "initialization_seed", "world_index")
    actual = {tuple(row[key] for key in fields): row for row in rows}
    assert report["complete"] and len(actual) == len(rows) == 48
    assert actual.keys() == expected.keys()
    assert all(float(row["flight_seconds"]) <= 300.002 for row in rows)
    counts = {scene: row["successes"] for scene, row in report["reports"].items()}
    for scene in config.benchmark.scenes:
        assert counts[scene] == sum(
            row["scene"] == scene and row["event"] == "SUCCESS" for row in rows
        )
    result = {
        "status": "completed",
        "source": str(source),
        "config_sha256_reused": cached["config_sha256"],
        "source_sha256_reused": cached["source_sha256"],
        "control_evidence": str(base / "fixed-scenes-long-return-progress.json"),
        "control_successes": 0,
        "per_scene_successes": counts,
        "events": dict(Counter(row["event"] for row in rows)),
        "supports_vertical_control_intervention": counts["S01"] >= 6 and counts["D01"] >= 6,
        "all_primary_screens_pass": all(count >= 6 for count in counts.values()),
        "intervention": "Only vertical control is replaced by existing Action altitude PD: "
        "az=clip(2*(goal_z-z)-1.5*vz,-3,3). Horizontal learned actions, true time input, "
        "physics, goal radius and deadline are unchanged.",
        "limits": "Frozen development diagnostic, not scratch training or acceptance. "
        "A retained recipe must disclose its changed action/control interface and train "
        "new two-action policies from random initialization. No general 3D maneuvering claim.",
    }
    atomic_json(output / "comparison.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
