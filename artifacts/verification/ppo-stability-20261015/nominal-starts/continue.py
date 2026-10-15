"""Compare reset distribution or rollout length from intact parallel Depth PPO update48."""

import argparse
import copy
import csv
import json
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning.checkpoint import load_model_state, save_model
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method, evaluate
from drone_playground.simulation.records import atomic_json

ROOT = Path(__file__).resolve().parents[4]
SCENES = ["S01", "S02", "S03", "D01", "D02", "D03"]


def main():
    """Retain the full source state and compare at equal additional interactions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=512, choices=(512, 2048))
    parser.add_argument("--randomize-navigation-start", action="store_true")
    args = parser.parse_args()
    if args.horizon == 2048 and not args.randomize_navigation_start:
        raise ValueError("Long-rollout comparison must retain the original random resets")
    if args.horizon == 512 and args.randomize_navigation_start:
        raise ValueError("Unchanged control already exists; reuse its completed checks")
    interactions_per_update = 64 * args.horizon
    planned_updates = 1572864 // interactions_per_update
    interval = 524288 // interactions_per_update
    assert planned_updates * interactions_per_update == 1572864
    assert interval * interactions_per_update == 524288
    output = args.output.resolve()
    source = ROOT / "results/fixed_scenes_parallel/depth_ppo_s0/checkpoints/model_000048"
    metadata = json.loads((source / "metadata.json").read_text())
    saved = copy.deepcopy(metadata["config"])
    assert saved["learning"]["randomize_navigation_start"] is True
    assert saved["experiment"]["learning"]["scene_sampling"] == "parallel"
    for options in (saved["learning"], saved["experiment"]["learning"]["options"]):
        options["randomize_navigation_start"] = args.randomize_navigation_start
        options["horizon"] = args.horizon
    saved["experiment"]["output"] = str(output)
    changes = subprocess.run(
        ["git", "diff", "811eabe", "--", "src", ":(exclude)src/drone_playground/cli.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert not changes, "Native learning or environment source changed since matched control"
    config = OmegaConf.create(saved["experiment"])
    trainer = Trainer(
        create_environment(config, scene=list(config.learning.scenes)),
        kind=config.method.actor.kind,
        algorithm="ppo",
        seed=config.seed,
        config=OmegaConf.to_container(config.learning.options, resolve=True),
        actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
        method=create_method(config),
        loss=config.learning.loss,
        backward_model=config.learning.backward_model,
        backward_options=OmegaConf.to_container(config.learning.backward_options),
    )
    assert trainer.resolved_config == saved["learning"]
    assert {scene.name: scene.geometry_hash for scene in trainer.env.scene.scenes} == (
        saved["training_sampling"]["geometry_sha256"]
    )
    state, _ = load_model_state(source, trainer.initialize())
    assert int(state.updates) == 48 and int(state.interactions) == 1572864
    cache = json.loads(
        (
            ROOT
            / "artifacts/verification/ppo-stability-20261015"
            / "depth-fixed-scenes-parallel-progress.json"
        ).read_text()
    )
    baseline = next(c for c in cache["checks"] if c["update"] == 48)
    cases = {tuple(row[:4]): row[4] for row in baseline["case_events"]}
    initial_successes = {k for k, event in cases.items() if event == "SUCCESS"}
    assert len(cases) == 48 and len(initial_successes) == 38
    output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    report = {
        "status": "running",
        "diagnostic": True,
        "source_checkpoint": str(source),
        "source_provenance_reused": metadata["provenance"],
        "source_config": metadata["config"],
        "continuation_config": saved,
        "runtime_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "source_runtime_difference": "Native src differs from811eabe only by CLI frozen-model "
        "metadata recording; learner/environment/network source unchanged",
        "planned_updates": planned_updates,
        "planned_additional_interactions": 1572864,
        "interactions_per_update": interactions_per_update,
        "screening_interval_updates": interval,
        "initial_successes": 38,
        "checks": [],
        "training_update_seconds": 0,
        "scope": "Matched reset-distribution or rollout-length diagnostic, "
        "not a new scratch seed or formal stability pass",
    }
    atomic_json(output / "report.json", report)
    evaluation = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    evaluation.mode = "checkpoint_eval"
    evaluation.benchmark.protocol = "diagnostic"
    evaluation.benchmark.scenes = SCENES
    evaluation.benchmark.episodes = 8
    evaluation.benchmark.seed_base = 1_000_000
    evaluation.benchmark.record_trajectories = False
    evaluation.benchmark.record_replay = False
    environments = {}

    def save():
        directory = output / f"model_{int(state.updates):06d}"
        save_model(
            directory,
            state,
            config=saved,
            kind=trainer.kind,
            actor_spec=trainer.actor.specification,
            export_policy=True,
            provenance={
                "updates": int(state.updates),
                "interactions": int(state.interactions),
                "status": report["status"],
                "diagnostic": True,
                "source_checkpoint": str(source),
                "runtime_commit": report["runtime_commit"],
                "diagnostic_report": str(output / "report.json"),
            },
        )
        return directory

    try:
        for offset in range(1, planned_updates + 1):
            before = time.monotonic()
            state, metrics = trainer.update(state)
            values = {k: float(v) for k, v in jax.device_get(metrics).items()}
            assert np.isfinite(list(values.values())).all()
            assert int(state.updates) == 48 + offset
            assert int(state.interactions) == 1572864 + offset * interactions_per_update
            values.update(update=int(state.updates), update_seconds=time.monotonic() - before)
            report["training_update_seconds"] += values["update_seconds"]
            report.update(last_update=int(state.updates), last_interactions=int(state.interactions))
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(values) + "\n")
            if offset % interval == 0:
                directory = save()
                evaluate(
                    evaluation,
                    directory,
                    actor=trainer.actor,
                    parameters=state.params,
                    evaluation_index=int(state.interactions) // 524288,
                    environments=environments,
                )
                with (directory / "episodes.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                current = {
                    tuple(
                        r[k]
                        for k in ("scene", "geometry_sha256", "initialization_seed", "world_index")
                    ): r["event"]
                    for r in rows
                }
                assert len(rows) == len(current) == 48 and current.keys() == cases.keys()
                successes = {k for k, event in current.items() if event == "SUCCESS"}
                counts = {s: sum(k[0] == s for k in successes) for s in SCENES}
                retained = len(successes & initial_successes)
                result = {
                    "update": int(state.updates),
                    "additional_interactions": offset * interactions_per_update,
                    "successes": len(successes),
                    "per_scene_successes": counts,
                    "retained_initial_successes": retained,
                    "initial_successes": 38,
                    "case_events": [[*k, event] for k, event in sorted(current.items())],
                    "lost_initial_cases": sorted(initial_successes - successes),
                    "new_success_cases": sorted(successes - initial_successes),
                    "pilot_gate": counts["S03"] >= 6
                    and retained >= 37
                    and all(counts[s] >= 7 for s in SCENES if s != "S03"),
                }
                report["checks"].append(result)
                print(json.dumps(result), flush=True)
            atomic_json(output / "report.json", report)
        report.update(
            status="completed",
            actual_additional_interactions=int(state.interactions) - 1572864,
            pilot_passed=report["checks"][-1]["pilot_gate"],
            total_wall_seconds=time.monotonic() - started,
        )
        report["final_checkpoint"] = str(save())
        atomic_json(output / "report.json", report)
    except BaseException as error:
        report.update(status="interrupted", error=repr(error))
        save()
        atomic_json(output / "report.json", report)
        raise
    finally:
        trainer.env.sim.close()
        for item in environments.values():
            if hasattr(item, "sim"):
                item.sim.close()


if __name__ == "__main__":
    main()
