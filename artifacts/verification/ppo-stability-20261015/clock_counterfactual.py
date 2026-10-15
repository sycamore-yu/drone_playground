"""Test a frozen policy's unseen remaining-time inputs without changing the task."""

import csv
import json
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from omegaconf import OmegaConf

from drone_playground.simulation.evaluation import evaluate
from drone_playground.simulation.networks import Actor
from drone_playground.simulation.policy import read_checkpoint
from drone_playground.simulation.records import atomic_json


def clamp_clock(observation, floor):
    """Change only the last state feature, the finite-horizon remaining time."""
    state = observation["state"]
    assert state.ndim == 2 and state.shape[-1] == 22
    return {**observation, "state": state.at[:, -1].set(jnp.maximum(state[:, -1], floor))}


class ClampedClock:
    """Use the existing frozen Actor with one diagnostic observation intervention."""

    def __init__(self, actor, floor):
        """Retain the unchanged Actor and its original parameter layout."""
        self.actor, self.floor = actor, floor

    def initialize_memory(self, batch):
        """Use the original episode-start recurrent state."""
        return self.actor.initialize_memory(batch)

    def apply(self, parameters, observation, memory):
        """Clamp perceived time while physical time and all other inputs remain real."""
        return self.actor.apply(parameters, clamp_clock(observation, self.floor), memory)


def main():
    """Reuse the completed control and measure only the new counterfactual cases."""
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    source = root / "results/fixed_scenes_long_return/depth_ppo_s0/checkpoints/model_000032"
    output = base / "depth-clock-counterfactual"
    output.mkdir(exist_ok=False)
    cached = json.loads((base / "fixed-scenes-long-return-progress.json").read_text())
    control = next(row for row in cached["checks"] if row["update"] == 32)
    scenes = ("S01", "D01")
    expected = {tuple(row[:4]): row[-1] for row in control["case_events"] if row[0] in scenes}
    assert len(expected) == 16 and set(expected.values()) == {"TIMEOUT"}
    _, metadata = read_checkpoint(source, "training")
    with ocp.StandardCheckpointer() as checkpointer:
        parameters = checkpointer.restore(source / "training")["params"]
    actor = Actor(**metadata["actor"])
    config = OmegaConf.create(metadata["config"]["experiment"])
    assert config.task.options.finite_horizon and config.task.duration == 300
    assert config.observation.navigation_relative_goal
    assert config.observation.navigation_world_rotation
    assert config.simulation.navigation_goal_observation
    assert metadata["provenance"]["config_sha256"] == cached["config_sha256"]
    block_steps = config.learning.scene_updates * config.learning.options.horizon
    training_max_actor_time = (block_steps - 1) / config.simulation.method_hz
    floor = 1.0 - training_max_actor_time / config.task.duration
    assert block_steps == 1024 and np.isclose(floor, 0.659)

    # Check feature isolation and preservation throughout the training-time interval.
    state = jnp.arange(4 * 22, dtype=jnp.float32).reshape(4, 22)
    state = state.at[:, -1].set(jnp.array([1.0, 0.8, 0.5, 0.0]))
    observation = {"state": state, "depth": jnp.ones((4, 2, 2))}
    clamped = clamp_clock(observation, floor)
    np.testing.assert_array_equal(clamped["state"][:, :-1], state[:, :-1])
    np.testing.assert_allclose(clamped["state"][:, -1], [1.0, 0.8, floor, floor])
    np.testing.assert_array_equal(clamped["state"][:2], state[:2])
    assert clamped["depth"] is observation["depth"]

    config.mode = "checkpoint_eval"
    config.benchmark.protocol = "diagnostic"
    config.benchmark.episodes = 8
    config.benchmark.scenes = list(scenes)
    config.benchmark.seed_base = 1_000_000
    config.benchmark.record_replay = False
    config.benchmark.record_trajectories = True
    report = evaluate(
        config, output / "clamped", actor=ClampedClock(actor, floor), parameters=parameters
    )
    with (output / "clamped/episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fields = ("scene", "geometry_sha256", "initialization_seed", "world_index")
    actual = {tuple(row[key] for key in fields): row for row in rows}
    assert report["complete"] and len(actual) == len(rows) == 16
    assert actual.keys() == expected.keys()
    assert all(float(row["flight_seconds"]) <= 300.002 for row in rows)
    counts = {scene: row["successes"] for scene, row in report["reports"].items()}
    for scene in scenes:
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
        "training_max_actor_time_seconds": training_max_actor_time,
        "remaining_time_floor": floor,
        "task_duration_seconds_unchanged": config.task.duration,
        "per_scene_successes": counts,
        "events": dict(Counter(row["event"] for row in rows)),
        "supports_clock_coverage_intervention": all(counts[scene] >= 6 for scene in scenes),
        "limits": "Development-only frozen observation counterfactual, not training or acceptance. "
        "A positive result implicates extrapolation in this input, not a unique collapse cause. "
        "Training continues concurrently; this diagnostic's wall time is not a cost comparison.",
    }
    atomic_json(output / "comparison.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
