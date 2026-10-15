"""Localize a recorded policy regression with two frozen parameter-block exchanges."""

import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def replace_action_head(representation, head):
    """Reuse immutable arrays, replacing only the action projection without mutation."""
    assert representation.keys() == head.keys() == {"params"}
    assert representation["params"].keys() == head["params"].keys()
    assert "gru" in representation["params"] and "action_head" in head["params"]
    return {
        "params": {
            **representation["params"],
            "action_head": head["params"]["action_head"],
        }
    }


def main():
    """Evaluate only the new hybrid policies; reuse both existing model outcomes."""
    import inspect

    import jax
    from omegaconf import OmegaConf

    from drone_playground.simulation.evaluation import evaluate
    from drone_playground.simulation.policy import load_policy
    from drone_playground.simulation.records import atomic_json

    sys.path.insert(0, str(ROOT / "tools"))
    from ppo_stability_window import SCENES, quality, read_cases

    design_path = BASE / "candidate-configs/depth-parallel-components-design.json"
    design = json.loads(design_path.read_text())
    control = json.loads((BASE / "depth-fixed-scenes-parallel-stability-report.json").read_text())
    regressed = next(row for row in control["checks"] if row["update"] == 184)
    baseline = {tuple(row[:4]): row[4] for row in control["baseline_cases"]}
    lost = {tuple(row) for row in regressed["lost_initial_cases"]}
    assert control["status"] == "completed" and regressed["successes"] == 124
    assert quality(baseline, baseline)["successes"] == 150
    runtime = Path(inspect.getfile(evaluate)).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True).strip()
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "src"],
        cwd=runtime,
        text=True,
    )
    assert revision == control["runtime_provenance"]["git_commit"] and not changes
    source = Path(control["source_checkpoint"])
    after_path = Path(regressed["model"])
    actor, before, before_meta = load_policy(source)
    _, after, after_meta = load_policy(after_path, actor=actor)
    assert before_meta["config"] == after_meta["config"]
    assert before_meta["actor"]["kind"] == "depth"
    assert jax.tree.structure(before) == jax.tree.structure(after)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
    config = OmegaConf.create(before_meta["config"]["experiment"])
    assert config.seed == 0 and config.benchmark.fixed_checkpoint_seeds
    config.mode = "checkpoint_eval"
    config.benchmark.protocol = "diagnostic"
    config.benchmark.scenes = list(SCENES)
    config.benchmark.episodes = 25
    config.benchmark.seed_base = control["evaluation_seed_base"]
    config.benchmark.record_replay = False
    config.benchmark.record_trajectories = False
    output = Path(design["output"])
    output.mkdir(exist_ok=False)
    result = {
        "status": "running",
        "scope": design["scope"],
        "design": str(design_path),
        "source_checkpoint": str(source),
        "regressed_checkpoint": str(after_path),
        "runtime_provenance_reused": control["runtime_provenance"],
        "runner_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "evaluation_seed_base": control["evaluation_seed_base"],
        "baseline_successes": 150,
        "regressed_successes": 124,
        "regressed_lost_cases": regressed["lost_initial_cases"],
        "hybrids": {},
    }
    atomic_json(output / "comparison.json", result)
    environments = {}
    try:
        for name, representation, head in (
            ("before_representation_after_head", before, after),
            ("after_representation_before_head", after, before),
        ):
            parameters = replace_action_head(representation, head)
            started = time.monotonic()
            evaluate(
                config,
                output / name,
                actor=actor,
                parameters=parameters,
                environments=environments,
                checkpoint_config=before_meta["config"],
            )
            cases, report = read_cases(output / name)
            successes = {key for key, event in cases.items() if event == "SUCCESS"}
            score = quality(cases, baseline)
            result["hybrids"][name] = {
                **score,
                "recovered_regressed_cases": sorted(successes & lost),
                "lost_previously_retained_cases": sorted(set(baseline) - lost - successes),
                "per_scene_events": {s: r["outcomes"] for s, r in report["reports"].items()},
                "wall_seconds": time.monotonic() - started,
            }
            atomic_json(output / "comparison.json", result)
            print(name, score["successes"], score["per_scene_successes"], flush=True)
        result["status"] = "completed"
        atomic_json(output / "comparison.json", result)
    finally:
        for env in environments.values():
            env.sim.close()


if __name__ == "__main__":
    main()
