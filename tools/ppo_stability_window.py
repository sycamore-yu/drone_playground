"""Measure fixed-scene PPO continuation with an optional single-setting diagnostic."""

import argparse
import copy
import csv
import inspect
import json
import math
import subprocess
import time
from pathlib import Path

SCENES = ("S01", "S02", "S03", "D01", "D02", "D03")
CASE_FIELDS = ("scene", "geometry_sha256", "initialization_seed", "world_index")


def window_steps(batch, diagnostic_updates=None):
    """Round the registered interaction budget and cadence up to whole updates."""
    if type(batch) is not int or batch <= 0:
        raise ValueError("Rollout batch must be a positive integer")
    updates = math.ceil(900_000 / batch)
    if diagnostic_updates is not None:
        if type(diagnostic_updates) is not int or not 0 < diagnostic_updates <= updates:
            raise ValueError("Diagnostic updates must be a positive integer within the full window")
        updates = diagnostic_updates
    return updates, math.ceil(122_880 / batch)


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


def continuation_config(config, scale, disable_entropy=False, full_batch=False):
    """Change one optimizer setting in both saved representations."""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Actor LR scale must be finite and in (0,1]")
    if sum((scale != 1, disable_entropy, full_batch)) > 1:
        raise ValueError("Change only one learning setting per matched diagnostic")
    result = copy.deepcopy(config)
    rate = result["learning"]["lr"] * scale
    result["learning"]["lr"] = rate
    result["experiment"]["learning"]["options"]["lr"] = rate
    if disable_entropy:
        weight = result["learning"]["entropy_weight"]
        if (
            not weight > 0
            or weight != result["experiment"]["learning"]["options"]["entropy_weight"]
        ):
            raise ValueError("Entropy removal requires a consistent positive source coefficient")
        result["learning"]["entropy_weight"] = 0.0
        result["experiment"]["learning"]["options"]["entropy_weight"] = 0.0
    if full_batch:
        batches = result["learning"]["minibatches"]
        if (
            type(batches) is not int
            or batches <= 1
            or batches != result["experiment"]["learning"]["options"]["minibatches"]
        ):
            raise ValueError(
                "Full-batch diagnostic requires consistent multiple source minibatches"
            )
        result["learning"]["minibatches"] = 1
        result["experiment"]["learning"]["options"]["minibatches"] = 1
    return result


def reuse_baseline(control, source, provenance):
    """Reuse the completed matched control's original case records."""
    if (
        control["status"] != "completed"
        or control.get("actor_lr_scale", 1.0) != 1.0
        or control.get("diagnostic", False)
        or control["source_checkpoint"] != str(source)
        or control["source_config_sha256_reused"] != provenance["config_sha256"]
        or control["source_sha256_reused"] != provenance["source_sha256"]
        or control["origin_update"] != provenance["updates"]
        or control["origin_interactions"] != provenance["interactions"]
    ):
        raise ValueError("Reference window is not the completed matched control")
    rows = control["baseline_cases"]
    cases = {tuple(row[:4]): row[4] for row in rows}
    if len(cases) != len(rows) or not quality(cases, cases)["passed"]:
        raise ValueError("Reference window lacks a valid first-pass baseline")
    return cases


def main():
    """Continue complete learner state with an explicitly recorded optional intervention."""
    import jax
    import numpy as np
    import optax
    from omegaconf import OmegaConf

    from drone_playground.learning import checkpoint
    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.evaluation import create_environment, create_method, evaluate
    from drone_playground.simulation.records import atomic_json, source_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-lr-scale", type=float, default=1.0)
    parser.add_argument("--disable-entropy-bonus", action="store_true")
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--diagnostic-updates", type=int)
    parser.add_argument("--reuse-window", type=Path)
    args = parser.parse_args()
    source, output = args.checkpoint.resolve(), args.output.resolve()
    metadata = json.loads((source / "metadata.json").read_text())
    diagnostic = args.actor_lr_scale != 1.0 or args.disable_entropy_bonus or args.full_batch
    if args.diagnostic_updates is not None and not diagnostic:
        raise ValueError("Only a diagnostic may use a shorter pilot window")
    if diagnostic and args.reuse_window is None:
        raise ValueError("Diagnostic requires an existing matched control window")
    metadata["config"] = continuation_config(
        metadata["config"], args.actor_lr_scale, args.disable_entropy_bonus, args.full_batch
    )
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
    control = None
    if args.reuse_window is None:
        baseline, original_report = read_cases(source)
        runtime_provenance = source_identity()
    else:
        control = json.loads(args.reuse_window.read_text())
        baseline = reuse_baseline(control, source, metadata["provenance"])
        original_report = json.loads((source / "report.json").read_text())
        assert control["evaluation_seed_base"] == original_report["seed_base"]
        checkout = Path(inspect.getfile(Trainer)).resolve().parents[3]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "src"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        runtime_provenance = control["runtime_provenance"]
        assert revision == runtime_provenance["git_commit"] and not changes
    assert original_report["passed"] and quality(baseline, baseline)["passed"]
    scenes = list(config.learning.scenes)
    parallel = config.learning.get("scene_sampling", "sequential") == "parallel"
    scene_updates = config.learning.get("scene_updates", 20)
    scene_index, updates_in_scene = 0, 0
    if not parallel:
        progress = metadata["provenance"]["training_sampling"]
        scene_index, updates_in_scene = progress["scene_index"], progress["updates_in_scene"]
    output.mkdir(exist_ok=False)
    trainers, environments = {}, {}
    started = time.monotonic()

    def trainer_for(index):
        if index not in trainers:
            trainers[index] = Trainer(
                create_environment(config, scene=scenes if parallel else scenes[index]),
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
            if parallel:
                batch_scene = trainers[index].env.scene
                assert {scene.name: scene.geometry_hash for scene in batch_scene.scenes} == geometry
                assert (
                    batch_scene.worlds_per_scene
                    == metadata["config"]["training_sampling"]["worlds_per_scene"]
                )
            else:
                assert trainers[index].env.scene.geometry_hash == geometry[scenes[index]]
        return trainers[index]

    trainer = trainer_for(scene_index)
    state, _ = checkpoint.load_model_state(source, trainer.initialize())
    origin_update, origin_interactions = int(state.updates), int(state.interactions)
    assert selection["update"] == metadata["provenance"]["updates"] == origin_update
    assert origin_interactions == metadata["provenance"]["interactions"]
    batch = config.simulation.num_envs * config.learning.options.horizon
    updates, interval = window_steps(batch, args.diagnostic_updates)
    if control is not None:
        reference_updates, _ = window_steps(batch)
        assert reference_updates == control["planned_updates"]
        assert interval == control["evaluation_interval_updates"]
        assert reference_updates * batch == control["actual_additional_interactions"]
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
        "runtime_provenance": runtime_provenance,
        "runner_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "origin_update": origin_update,
        "origin_interactions": origin_interactions,
        "planned_updates": updates,
        "planned_additional_interactions": updates * batch,
        "evaluation_interval_updates": interval,
        "evaluation_seed_base": original_report["seed_base"],
        "baseline_cases": [[*key, event] for key, event in sorted(baseline.items())],
        "checks": [],
        "actor_lr_scale": args.actor_lr_scale,
        "actor_lr": config.learning.options.lr,
        "entropy_bonus_disabled": args.disable_entropy_bonus,
        "entropy_weight": config.learning.options.entropy_weight,
        "full_batch": args.full_batch,
        "optimizer_steps_per_update": config.learning.options.ppo_epochs
        * config.learning.options.minibatches,
        "diagnostic_updates": args.diagnostic_updates,
        "diagnostic": diagnostic,
        "reference_window": str(args.reuse_window.resolve()) if args.reuse_window else None,
        "scene_switches": [],
        "training_update_seconds": 0.0,
        "setup_seconds": time.monotonic() - started,
        "scope": "Actual unchanged PPO continuation; original validation reused, no early stop "
        "on failure, no rollback, actual final model retained. "
        "Independent final test remains separate.",
    }
    if args.actor_lr_scale != 1.0:
        result["scope"] = (
            "Matched Actor-LR continuation diagnostic; all other learning settings and "
            "complete source state are retained. This is not unchanged-recipe acceptance."
        )
    elif args.disable_entropy_bonus:
        result["scope"] = (
            "Matched entropy-bonus removal diagnostic; stochastic sampling, trainable log_std, "
            "all other learning settings and complete source state are retained. "
            "This is not unchanged-recipe acceptance."
        )
    elif args.full_batch:
        result["scope"] = (
            "Full-batch PPO optimizer diagnostic. Rollout size and learning rates are unchanged; "
            "fewer Actor and Critic Adam steps per rollout accompany larger gradient batches. "
            "This is not unchanged-recipe acceptance."
        )
    if args.diagnostic_updates is not None:
        result["scope"] += " Short pilot, not the complete registered stability window."
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
            "stability_source_checkpoint": str(source),
            "stability_window_report": str(output / "report.json"),
            "continuation_actor_lr_scale": args.actor_lr_scale,
            "continuation_entropy_bonus_disabled": args.disable_entropy_bonus,
            "continuation_full_batch": args.full_batch,
            "evaluation_index": original_report["evaluation_index"] + len(result["checks"]),
            "training_wall_seconds": metadata["provenance"]["training_wall_seconds"]
            + time.monotonic()
            - started,
        }
        if not parallel:
            provenance["training_sampling"] = {
                "scene_index": scene_index,
                "updates_in_scene": updates_in_scene,
            }
        if diagnostic:
            provenance["source_config_sha256"] = provenance.pop("config_sha256")
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
            if not parallel:
                assert (int(state.updates) - updates_in_scene) % (scene_updates * len(scenes)) == (
                    scene_index * scene_updates
                )
            if not parallel and updates_in_scene == scene_updates:
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
            if diagnostic:
                delta = optax.tree.norm(
                    jax.tree.map(lambda a, b: a - b, candidate.params, state.params)
                )
                measured["actor_parameter_delta_norm"] = float(delta)
                assert np.isfinite(measured["actor_parameter_delta_norm"]) and delta > 0
            state = candidate
            if not parallel:
                updates_in_scene += 1
            assert int(state.updates) == origin_update + offset
            assert int(state.interactions) == origin_interactions + offset * batch
            measured.update(
                update=int(state.updates),
                update_seconds=time.monotonic() - before,
            )
            measured.update({"scenes": scenes} if parallel else {"scene": scenes[scene_index]})
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
