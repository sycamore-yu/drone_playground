"""Record the missing S03 paths for the completed full-batch PPO comparison."""

import csv
import inspect
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from drone_playground.simulation.evaluation import evaluate
from drone_playground.simulation.policy import load_policy
from drone_playground.simulation.records import atomic_json

OUTPUT = Path(__file__).resolve().parent
BASE = OUTPUT.parent
FIELDS = ("scene", "geometry_sha256", "initialization_seed", "world_index")


def main():
    """Reuse frozen weights and check all case outcomes against the original tables."""
    design = json.loads(
        (BASE / "candidate-configs/depth-s03-full-batch-trajectories-design.json").read_text()
    )
    runtime = Path(inspect.getfile(evaluate)).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True)
    assert revision.strip() == design["runtime_commit"]
    result = {"status": "running", "scope": design["scope"], "runs": {}}
    actor, environments = None, {}
    started = time.monotonic()
    try:
        for label, source in design["sources"].items():
            directory = OUTPUT / label
            directory.mkdir(exist_ok=False)
            actor, parameters, metadata = load_policy(source, actor=actor)
            config = OmegaConf.create(metadata["config"]["experiment"])
            assert config.seed == 0 and config.benchmark.fixed_checkpoint_seeds
            config.mode = "checkpoint_eval"
            config.benchmark.protocol = "learning"
            config.benchmark.scenes = ["S03"]
            config.benchmark.episodes = 25
            config.benchmark.seed_base = 1_500_000
            config.benchmark.record_trajectories = True
            config.benchmark.record_replay = False
            before = time.monotonic()
            report = evaluate(
                config, directory, actor=actor, parameters=parameters,
                environments=environments, checkpoint_config=metadata["config"],
            )
            assert report["complete"] and report["reports"]["S03"]["episodes"] == 25
            tables = []
            for path in (Path(source) / "episodes.csv", directory / "episodes.csv"):
                with path.open(newline="") as stream:
                    rows = [r for r in csv.DictReader(stream) if r["scene"] == "S03"]
                tables.append({tuple(r[k] for k in FIELDS): r["event"] for r in rows})
            assert len(tables[0]) == len(tables[1]) == 25
            assert tables[0].keys() == tables[1].keys()
            env = environments[("S03", 25)]
            result["runs"][label] = {
                "source": source, "wall_seconds": time.monotonic() - before,
                "successes": report["reports"]["S03"]["successes"],
                "case_events_match": tables[0] == tables[1],
                "changed_events": [
                    {"case": key, "original": tables[0][key], "recorded": event}
                    for key, event in tables[1].items() if tables[0][key] != event
                ],
                "goal_metres": np.asarray(env.task.goal).tolist(),
                "bounds_metres": [np.asarray(env.task.low).tolist(),
                                  np.asarray(env.task.high).tolist()],
            }
            atomic_json(OUTPUT / "comparison.json", result)
            print(label, result["runs"][label], flush=True)
    finally:
        for env in environments.values():
            if hasattr(env, "sim"):
                env.sim.close()
    result.update(status="completed", wall_seconds=time.monotonic() - started)
    atomic_json(OUTPUT / "comparison.json", result)


if __name__ == "__main__":
    main()
