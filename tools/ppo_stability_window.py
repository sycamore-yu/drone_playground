"""Continue a selected fixed-scene PPO learner through the agreed stability window."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

SCENES = ("S01", "S02", "S03", "D01", "D02", "D03")
CASE_FIELDS = ("scene", "geometry_sha256", "initialization_seed", "world_index")


def window_steps(batch):
    """Round the registered interaction budget and cadence up to whole updates."""
    if type(batch) is not int or batch <= 0:
        raise ValueError("Rollout batch must be a positive integer")
    return math.ceil(900_000 / batch), math.ceil(122_880 / batch)


def quality(cases, baseline):
    """Require every primary scene and retention of the original successful cases."""
    if cases.keys() != baseline.keys():
        raise ValueError("Stability checks must use the exact original validation cases")
    if set(key[0] for key in cases) != set(SCENES) or any(
        sum(key[0] == scene for key in cases) != 25 for scene in SCENES
    ):
        raise ValueError("Stability requires 25 cases in each of six primary scenes")
    successes = {key for key, event in cases.items() if event == "SUCCESS"}
    original = {key for key, event in baseline.items() if event == "SUCCESS"}
    counts = {scene: sum(key[0] == scene for key in successes) for scene in SCENES}
    retained = len(successes & original)
    return {
        "per_scene_successes": counts,
        "successes": len(successes),
        "initial_successes": len(original),
        "retained_initial_successes": retained,
        "retention_fraction": retained / len(original) if original else 0.0,
        "lost_initial_cases": sorted(original - successes),
        "new_success_cases": sorted(successes - original),
        "passed": all(value >= 23 for value in counts.values())
        and bool(original)
        and retained * 100 >= len(original) * 95,
    }


def read_cases(directory):
    """Validate a newly completed table once; never replay an existing control."""
    report = json.loads((directory / "report.json").read_text())
    if not report.get("complete"):
        raise ValueError("Incomplete stability evaluation")
    with (directory / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = {tuple(row[key] for key in CASE_FIELDS): row["event"] for row in rows}
    if len(cases) != len(rows):
        raise ValueError("Duplicate stability cases")
    for row in rows:
        seconds = float(row["flight_seconds"])
        if not math.isfinite(seconds) or not 0 < seconds <= 300.002:
            raise ValueError("Invalid physical completion time")
        if row["event"] == "SUCCESS" and (
            row["terminated"] != "True" or row["truncated"] != "False"
        ):
            raise ValueError("Successful arrival must be a true termination")
    score = quality(cases, cases)
    if set(report["reports"]) != set(SCENES) or any(
        report["reports"][scene]["episodes"] != 25
        or report["reports"][scene]["successes"] != score["per_scene_successes"][scene]
        for scene in SCENES
    ):
        raise ValueError("Stability report disagrees with its episode table")
    return cases, report


def main():
    """Keep the original optimizer, scene schedule, rewards and random-state lineage."""
    import jax
    import numpy as np
    from omegaconf import OmegaConf

    from drone_playground.learning import checkpoint
    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.evaluation import create_environment, create_method, evaluate
    from drone_playground.simulation.records import atomic_json, source_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.checkpoint.resolve(), args.output.resolve()
    metadata = json.loads((source / "metadata.json").read_text())
    config = OmegaConf.create(metadata["config"]["experiment"])
    original_run = json.loads((source.parents[1] / "run.json").read_text())
    selection = original_run["selection"]
    assert config.learning.algorithm == "ppo" and config.task.name == "navigation"
    assert config.initial_checkpoint is None and config.resume is None
    assert config.learning.selection == selection["rule"] == "first_full_validation_pass"
    assert selection["checkpoint"] == str(Path("checkpoints") / source.name)
    assert metadata["provenance"]["consecutive_passes"] == 1
    assert config.benchmark.fixed_checkpoint_seeds and config.task.duration == 300
    assert not config.simulation.get("procedural")
    baseline, original_report = read_cases(source)
    assert original_report["passed"] and quality(baseline, baseline)["passed"]
    scenes = list(config.learning.scenes)
    scene_updates = config.learning.scene_updates
    progress = metadata["provenance"]["training_sampling"]
    scene_index, updates_in_scene = progress["scene_index"], progress["updates_in_scene"]
    output.mkdir(exist_ok=False)
    trainers, environments = {}, {}
    started = time.monotonic()

    def trainer_for(index):
        if index not in trainers:
            trainers[index] = Trainer(
                create_environment(config, scene=scenes[index]),
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
            assert trainers[index].resolved_config == metadata["config"]["learning"]
            geometry = metadata["config"]["training_sampling"]["geometry_sha256"]
            assert trainers[index].env.scene.geometry_hash == geometry[scenes[index]]
        return trainers[index]

    trainer = trainer_for(scene_index)
    state, _ = checkpoint.load_model_state(source, trainer.initialize())
    origin_update, origin_interactions = int(state.updates), int(state.interactions)
    assert selection["update"] == metadata["provenance"]["updates"] == origin_update
    assert origin_interactions == metadata["provenance"]["interactions"]
    batch = config.simulation.num_envs * config.learning.options.horizon
    updates, interval = window_steps(batch)
    evaluation = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    evaluation.mode = "checkpoint_eval"
    evaluation.benchmark.protocol = "learning"
    evaluation.benchmark.scenes = list(SCENES)
    evaluation.benchmark.episodes = 25
    evaluation.benchmark.seed_base = original_report["seed_base"]
    evaluation.benchmark.record_replay = False
    evaluation.benchmark.record_trajectories = False
    result = {
        "status": "running",
        "passed": False,
        "source_checkpoint": str(source),
        "source_run": str(source.parents[1]),
        "source_config_sha256_reused": metadata["provenance"]["config_sha256"],
        "source_sha256_reused": metadata["provenance"]["source_sha256"],
        "runtime_provenance": source_identity(),
        "origin_update": origin_update,
        "origin_interactions": origin_interactions,
        "planned_updates": updates,
        "planned_additional_interactions": updates * batch,
        "evaluation_interval_updates": interval,
        "evaluation_seed_base": original_report["seed_base"],
        "baseline_cases": [[*key, event] for key, event in sorted(baseline.items())],
        "checks": [],
        "scene_switches": [],
        "training_update_seconds": 0.0,
        "setup_seconds": time.monotonic() - started,
        "scope": "Actual unchanged PPO continuation; original validation reused, no early stop "
        "on failure, no rollback, actual final model retained. "
        "Independent final test remains separate.",
    }
    assert result["runtime_provenance"]["source_sha256"] == result["source_sha256_reused"]
    atomic_json(output / "report.json", result)

    def save_model(directory):
        provenance = {
            **metadata["provenance"],
            **result["runtime_provenance"],
            "updates": int(state.updates),
            "interactions": int(state.interactions),
            "last_update": int(state.updates),
            "last_interactions": int(state.interactions),
            "status": result["status"],
            "consecutive_passes": int(result["checks"][-1]["passed"]) if result["checks"] else 1,
            "last_evaluation": result["checks"][-1] if result["checks"] else original_report,
            "training_sampling": {
                "scene_index": scene_index,
                "updates_in_scene": updates_in_scene,
            },
            "stability_source_checkpoint": str(source),
            "stability_window_report": str(output / "report.json"),
            "evaluation_index": original_report["evaluation_index"] + len(result["checks"]),
            "training_wall_seconds": metadata["provenance"]["training_wall_seconds"]
            + time.monotonic()
            - started,
        }
        checkpoint.save_model(
            directory,
            state,
            config=metadata["config"],
            provenance=provenance,
            kind=trainer.kind,
            actor_spec=trainer.actor.specification,
            export_policy=True,
        )

    try:
        for offset in range(1, updates + 1):
            assert (int(state.updates) - updates_in_scene) % (scene_updates * len(scenes)) == (
                scene_index * scene_updates
            )
            if updates_in_scene == scene_updates:
                scene_index = (scene_index + 1) % len(scenes)
                trainer = trainer_for(scene_index)
                truncated = int(jax.device_get((~state.env_state.done).sum()))
                state = trainer.reset_episodes(state)
                updates_in_scene = 0
                result["scene_switches"].append(
                    {
                        "before_update": int(state.updates),
                        "scene": scenes[scene_index],
                        "truncated_worlds": truncated,
                    }
                )
            before = time.monotonic()
            candidate, metrics = trainer.update(state)
            measured = {key: float(value) for key, value in jax.device_get(metrics).items()}
            assert np.isfinite(list(measured.values())).all(), measured
            state = candidate
            updates_in_scene += 1
            assert int(state.updates) == origin_update + offset
            assert int(state.interactions) == origin_interactions + offset * batch
            measured.update(
                update=int(state.updates),
                scene=scenes[scene_index],
                update_seconds=time.monotonic() - before,
            )
            result["training_update_seconds"] += measured["update_seconds"]
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(measured) + "\n")
            result.update(last_update=int(state.updates), last_interactions=int(state.interactions))
            if offset % interval == 0 or offset == updates:
                directory = output / f"model_{int(state.updates):06d}"
                save_model(directory)
                before = time.monotonic()
                evaluate(
                    evaluation,
                    directory,
                    actor=trainer.actor,
                    parameters=state.params,
                    evaluation_index=original_report["evaluation_index"]
                    + len(result["checks"])
                    + 1,
                    environments=environments,
                )
                cases, _ = read_cases(directory)
                score = quality(cases, baseline)
                result["checks"].append(
                    {
                        "update": int(state.updates),
                        "additional_interactions": offset * batch,
                        "model": str(directory),
                        "evaluation_seconds": time.monotonic() - before,
                        **score,
                    }
                )
                save_model(directory)
                print(json.dumps(result["checks"][-1]), flush=True)
            atomic_json(output / "report.json", result)
        result.update(
            status="completed",
            passed=all(row["passed"] for row in result["checks"]),
            actual_additional_interactions=int(state.interactions) - origin_interactions,
            final_checkpoint=result["checks"][-1]["model"],
            total_wall_seconds=time.monotonic() - started,
        )
        assert len(result["checks"]) == math.ceil(updates / interval)
        save_model(Path(result["final_checkpoint"]))
        atomic_json(output / "report.json", result)
    except BaseException as error:
        save_model(output / f"model_{int(state.updates):06d}")
        result.update(status="interrupted", error=repr(error), last_update=int(state.updates))
        atomic_json(output / "report.json", result)
        raise
    finally:
        for item in [*trainers.values(), *environments.values()]:
            env = getattr(item, "env", item)
            if hasattr(env, "sim"):
                env.sim.close()


if __name__ == "__main__":
    main()
