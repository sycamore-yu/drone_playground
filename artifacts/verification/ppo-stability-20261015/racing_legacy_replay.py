"""Replay the original successful Racing PPO checkpoint on its consumed cases."""

import csv
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from drone_playground.simulation.evaluation import create_environment, evaluate
from drone_playground.simulation.policy import load_policy
from drone_playground.simulation.records import atomic_json, source_identity

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
from ppo_branch_diagnostic import tree_hash  # noqa: E402

source = Path(
    "/home/tong/tongworkspace/simulation_dev/new_drone_playground/"
    "results/procedural_v01/racing_ppo_s0/checkpoints/model_000200"
)
output = Path(__file__).with_name("racing_legacy_replay_matched")
if output.exists():
    raise FileExistsError(output)
actor, parameters, metadata = load_policy(source)
parameter_hash = tree_hash(parameters)
original = json.loads((source / "report.json").read_text())
config = OmegaConf.create(metadata["config"]["experiment"])
config.mode = "checkpoint_eval"
config.output = str(output)
config.checkpoint = str(source)
config.resume = None
config.benchmark.seed_base = original["seed_base"]
config.benchmark.record_trajectories = False
config.benchmark.record_replay = False
assert config.sensor.name == "state" and actor.kind == "state"
env = create_environment(config, num_envs=100, scene="racing")
assert env.sensor is None
output.mkdir()
OmegaConf.save(config, output / "config.yaml")
try:
    # The old archive lacks a sensor contract: direct forward diagnosis only.
    evaluate(
        config,
        output,
        actor=actor,
        parameters=parameters,
        evaluation_index=original["evaluation_index"],
        environments={
            ("racing", 100): env,
        },
    )
finally:
    env.sim.close()
with (source / "episodes.csv").open() as stream:
    before = list(csv.DictReader(stream))
with (output / "episodes.csv").open() as stream:
    after = list(csv.DictReader(stream))
fields = ("scene", "geometry_sha256", "initialization_seed", "world_index")
old = {tuple(row[k] for k in fields): row for row in before}
new = {tuple(row[k] for k in fields): row for row in after}
assert len(old) == len(new) == 100 and old.keys() == new.keys()
assert tree_hash(parameters) == parameter_hash
result = {
    "runtime_provenance": source_identity(),
    "source": str(source),
    "actor_parameter_sha256": parameter_hash,
    "original_report_sha256": hashlib.sha256((source / "report.json").read_bytes()).hexdigest(),
    "seed_base": original["seed_base"],
    "evaluation_index": original["evaluation_index"],
    "successes": sum(row["event"] == "SUCCESS" for row in after),
    "episodes": len(after),
    "matching_event_count": sum(old[key]["event"] == new[key]["event"] for key in old),
    "formal_loader": "Rejected missing legacy sensor contract; failure retained separately",
    "scope": "Frozen forward compatibility on consumed cases; not scratch convergence or C5",
}
atomic_json(output / "comparison.json", result)
print(json.dumps(result), flush=True)
