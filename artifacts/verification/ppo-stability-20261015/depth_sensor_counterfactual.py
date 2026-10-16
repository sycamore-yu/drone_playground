"""Replay one retained peak policy with actual versus unavailable Depth input."""

import csv
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import serialization
from hydra.utils import instantiate
from omegaconf import OmegaConf

from drone_playground.simulation.evaluation import evaluate
from drone_playground.simulation.records import atomic_json


class WithoutDepth:
    """Mask only perception, retaining the exact frozen state/GRU policy."""

    def __init__(self, actor):
        """Retain the original actor and parameter layout."""
        self.actor = actor

    def initialize_memory(self, batch):
        """Use the same episode-start memory as the original policy."""
        return self.actor.initialize_memory(batch)

    def apply(self, parameters, observation, memory):
        """Supply an unavailable Depth frame to the unchanged frozen network."""
        masked = {
            **observation,
            "depth": jnp.zeros_like(observation["depth"]),
            "depth_mask": jnp.zeros_like(observation["depth_mask"]),
        }
        return self.actor.apply(parameters, masked, memory)


def episodes(path):
    """Use physical scene and initialization identity, independent of row order."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fields = ("scene", "geometry_sha256", "initialization_seed", "world_index")
    result = {tuple(row[k] for k in fields): row for row in rows}
    assert len(result) == len(rows)
    return result


root = Path(__file__).resolve().parent
source = root / "depth-batch64-retention-pair/model_000300"
output = root / "depth-sensor-counterfactual"
output.mkdir(exist_ok=False)
metadata = json.loads((source / "metadata.json").read_text())
config = OmegaConf.create(metadata["config"]["experiment"])
assert config.method.actor.kind == "depth" and config.learning.algorithm == "ppo"
with ocp.StandardCheckpointer() as checkpointer:
    state = checkpointer.restore(source / "training")
parameters = state["params"]
parameter_hash = hashlib.sha256(serialization.to_bytes(parameters)).hexdigest()
actor = instantiate(config.method.actor, action_size=3)
assert actor.specification == metadata["actor"]
config.mode = "checkpoint_eval"
config.benchmark.protocol = "diagnostic"
config.benchmark.episodes = 8
config.benchmark.scenes = ["S01", "D01", "D03"]
config.benchmark.seed_base = 1_000_000
config.benchmark.record_replay = False
config.benchmark.record_trajectories = True
expected = {
    k: v
    for k, v in episodes(source / "screening/episodes.csv").items()
    if k[0] in config.benchmark.scenes
}
assert len(expected) == 24 and sum(r["event"] == "SUCCESS" for r in expected.values()) == 20
results = {}
for name, policy in [("actual", actor), ("without_depth", WithoutDepth(actor))]:
    report = evaluate(config, output / name, actor=policy, parameters=parameters)
    rows = episodes(output / name / "episodes.csv")
    assert rows.keys() == expected.keys()
    if name == "actual":
        assert {k: v["event"] for k, v in rows.items()} == {
            k: v["event"] for k, v in expected.items()
        }
    source_successes = {k for k, v in expected.items() if v["event"] == "SUCCESS"}
    successes = {k for k, v in rows.items() if v["event"] == "SUCCESS"}
    results[name] = {
        "per_scene": {s: r["successes"] for s, r in report["reports"].items()},
        "successes": len(successes),
        "retained_source_successes": len(successes & source_successes),
        "lost_cases": sorted(source_successes - successes),
        "new_successes": sorted(successes - source_successes),
        "csv_sha256": hashlib.sha256((output / name / "episodes.csv").read_bytes()).hexdigest(),
    }
    assert hashlib.sha256(serialization.to_bytes(parameters)).hexdigest() == parameter_hash
    atomic_json(
        output / "comparison.json",
        {
            "source": str(source),
            "actor_sha256": parameter_hash,
            "source_metadata_sha256": hashlib.sha256(
                (source / "metadata.json").read_bytes()
            ).hexdigest(),
            "intervention": (
                "Only depth and depth_mask become all-zero, the encoder's no-valid-return input. "
                "Geometry, acquisition, state observations, recurrent reset and weights are fixed."
            ),
            "limitation": (
                "Development counterfactual. Performance loss shows sensitivity to missing Depth, "
                "not competent geometric avoidance; missing data can be out of distribution."
            ),
            "results": results,
        },
    )
    print(name, json.dumps(results[name]), flush=True)
