"""Run native Brax PPO/APG and the local SHAC extension on Crazyflow tasks.

PPO has native snapshot callbacks. Brax 0.14.2 APG supplies progress metrics
but no parameter callback: initial/final snapshots are saved without modifying
its optimizer loop. This distinction is recorded in each run configuration.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import crazyflow
import jax
import jax.numpy as jnp
import numpy as np
from brax.training import networks
from brax.training.acme import running_statistics, specs
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.apg import train as apg_train
from brax.training.agents.ppo import train as ppo_train

from drone_playground.artifacts.reporting import save_report
from drone_playground.benchmarks import (
    apply_quality,
    checkpoint_eval_score,
    checkpoint_eval_seeds,
    pilot_scene_rates,
)
from drone_playground.environments.factory import build_environment
from drone_playground.evaluation.run import make_evaluator
from drone_playground.evaluation.tracking.metrics import select_replays
from drone_playground.learning.brax_configuration import native_training_config
from drone_playground.learning.checkpointing import require_matching_physical_decoder, save_policy
from drone_playground.learning.inference import load_policy
from drone_playground.learning.wrappers import wrap_for_training
from drone_playground.networks.factory import network_factory


def evaluation_scalars(task: str, report: dict) -> dict:
    """Map a task report onto the scalar TensorBoard surface."""
    if task == "navigation":
        return {
            "eval/success_rate": report["success_rate"],
            "eval/collision_rate": report["collision_rate"],
            "eval/constrained_time_s": report["constrained_time_mean_s"],
            "eval/timeout_rate": report["timeout_rate"],
            "eval/return": report["return_mean"],
        }
    return {
        "eval/completion_rate": report["completion_rate"],
        "eval/rmse_all_m": report["rmse_all_mean"],
        "eval/return": report["return_mean"],
    }


def inherit_dva_selection(state_path: Path, destination: Path, task: str):
    """Continue checkpoint_eval selection without importing evaluations after the saved state."""
    meta = json.loads(state_path.with_suffix(".json").read_text())
    saved = meta["config"]
    cutoff = int(meta["updates"]) * int(saved["num_envs"]) * int(saved["horizon_length"])
    source = state_path.resolve().parent.parent
    candidates = []
    for path in sorted((source / "eval").glob("step-*.json")):
        report = json.loads(path.read_text())
        if report.get("role") == "eval" and report["step"] <= cutoff:
            candidates.append(report)
    if not candidates:
        raise ValueError("D.VA experiment resume requires its earlier checkpoint_eval reports")
    best = max(candidates, key=lambda report: checkpoint_eval_score(task, report))
    policy = (source / best["checkpoint"]).resolve()
    if not policy.is_relative_to(source):
        raise ValueError("Inherited policy must belong to the source run")
    _, _, identity = load_policy(policy)
    if identity["step"] != best["step"]:
        raise ValueError("Inherited report and checkpoint step disagree")
    for path in (policy, policy.with_suffix(".json")):
        shutil.copyfile(path, destination / "checkpoints" / path.name)
    score = checkpoint_eval_score(task, best)
    save_report(
        destination / "checkpoints" / "best.json",
        dict(
            path=policy.name,
            step=best["step"],
            score=list(score),
            selection_role="eval",
        ),
    )
    save_report(
        destination / "resume-selection.json",
        dict(
            source_state=str(state_path),
            source_state_sha256=meta["sha256"],
            eligible_steps=[r["step"] for r in candidates],
            cutoff_step=cutoff,
            selected_step=best["step"],
            future_evaluations_excluded=True,
        ),
    )
    save_report(destination / "eval" / f"step-{best['step']:010d}.json", best)
    return score, best


def export_run_replays(env, trace, report, directory):
    """Export replays through the contract the selected task actually owns."""
    if env.task.name == "navigation":
        from drone_playground.evaluation.navigation.policy import (
            export_navigation_replays,
            select_episodes,
        )

        return export_navigation_replays(
            env,
            trace,
            directory,
            case_indices=select_episodes(report),
            scenario_groups=report["scenario_groups"],
        )
    from drone_playground.visualization.rscope_io import export_rollout as write_rollout

    return write_rollout(env.sim, directory, select_replays(trace, report))


def train(
    config: dict,
    root: Path,
    run_id: str,
    device: str = "gpu",
    warm_start: Path | None = None,
) -> dict:
    """Train the declared complete budget and record real snapshots and evaluations."""
    from drone_playground.artifacts.console import capture_console
    from drone_playground.artifacts.record import RunRecorder
    from drone_playground.visualization.rscope_publish import publish_snapshot

    config = dict(config)
    algorithm = config["algorithm"]
    if algorithm not in {"ppo", "apg", "bptt", "shac", "dva"}:
        raise ValueError("Supported algorithms are PPO, APG, SHAC and D.VA")
    config.update(
        device=device,
        snapshot_schedule="initial-and-final" if algorithm == "apg" else "per-epoch",
        trainer=(
            f"drone_playground.learning.algorithms.{algorithm}"
            if algorithm in {"bptt", "shac", "dva"}
            else "brax.training.agents." + algorithm
        ),
        reset_contract="fresh-same-step-with-terminal-observation",
        timeout_contract=config["components"]["env"]["task"]["time_limit_kind"],
        task_protocol=config["components"]["env"]["task"],
    )
    from drone_playground.artifacts.provenance import package_provenance

    config["crazyflow_code"], dependency_patch = package_provenance(crazyflow)
    config["actual_devices"] = [str(device) for device in jax.devices()]
    task_id = "07" if config["task"] == "racing" else ("04" if algorithm == "shac" else "05")
    resolved = dict(config["components"])
    resolved["provenance"] = {
        "crazyflow_code": config["crazyflow_code"],
        "trainer": config["trainer"],
        "actual_devices": config["actual_devices"],
    }
    rec = RunRecorder(root, run_id, resolved, task_id=task_id)
    (rec.path / "crazyflow.patch").write_bytes(dependency_patch)
    console = capture_console(rec.path / "console.log")
    console.__enter__()
    env = evaluation_env = None
    evaluator = None
    best_score = (-1, -float("inf"))
    best = None
    snapshot_count = 0
    live_publications = []
    start = time.monotonic()
    initial_params = None
    try:
        rec.phase("initializing", step=0)
        if config.get("warm_start"):
            metadata = json.loads(Path(config["warm_start"]).with_suffix(".json").read_text())
            require_matching_physical_decoder(metadata, config)
        if algorithm == "dva" and config.get("resume"):
            best_score, best = inherit_dva_selection(
                Path(config["resume"]), rec.path, config["task"]
            )
        env = build_environment(
            config["components"], device, role="train", count=config["num_envs"]
        )
        rec.record_environment(env)
        if config["task"] == "racing":
            save_report(
                rec.path / "native-task-config.json",
                json.loads(env.config.to_json()),
            )
        checkpoint_eval_count = int(config.get("checkpoint_eval_episodes", 32))
        if checkpoint_eval_count < 1:
            raise ValueError("CheckpointEval evaluation requires at least one episode")
        evaluation_env = build_environment(
            config["components"], device, "eval", checkpoint_eval_count
        )
        evaluation_seeds = checkpoint_eval_seeds(config, checkpoint_eval_count)
        if env.task.name == "navigation":
            checkpoint_eval_case = {
                "role": "eval",
                "reset_seeds": evaluation_seeds,
                "configured_repeats": checkpoint_eval_count,
                "bank_digest": evaluation_env.bank.digest(),
                "selection_rule": config.get("checkpoint_eval_metric"),
                "acceptance": config["components"]["evaluation"].get("acceptance"),
            }
            save_report(
                rec.path / "eval" / "checkpoint-eval-cases.json",
                checkpoint_eval_case,
            )
        else:
            save_report(
                rec.path / "eval" / "checkpoint-eval-cases.json",
                {
                    "role": "eval",
                    "reset_seeds": evaluation_seeds,
                    "reference_seeds": (
                        list(
                            range(
                                evaluation_env.reference_seed,
                                evaluation_env.reference_seed + checkpoint_eval_count,
                            )
                        )
                        if getattr(env, "reference_kind", None) == "random"
                        else [evaluation_env.reference_seed]
                    ),
                    "selection_rule": config.get("checkpoint_eval_metric"),
                    "acceptance": config["components"]["evaluation"].get("acceptance"),
                },
            )

        def snapshot(step, make_policy, params):
            nonlocal evaluator, best_score, best, snapshot_count, initial_params
            step = int(step)
            if initial_params is None:
                initial_params = jax.tree.map(lambda x: np.array(x, copy=True), params[1])
            if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
                raise FloatingPointError("Non-finite learned parameters")
            rec.phase("evaluation", step=step)
            checkpoint = save_policy(rec.path / "checkpoints", params, config, step)
            if evaluator is None:
                evaluator = make_evaluator(evaluation_env, make_policy, evaluation_seeds)
            t = time.monotonic()
            report, trace = evaluator.run(params)
            if snapshot_count == 0 and env.task.name == "navigation":
                checkpoint_eval_case.update(
                    scenario_groups=report["scenario_groups"],
                    initial_conditions=report["initial_conditions"],
                )
                save_report(
                    rec.path / "eval" / "checkpoint-eval-cases.json",
                    checkpoint_eval_case,
                )
            apply_quality(report, config)
            report.update(
                step=step,
                checkpoint=str(checkpoint.relative_to(rec.path)),
                role="eval",
            )
            if config.get("checkpoint_eval_metric") == "release-pilot-v1":
                report["selection_rule"] = "release-pilot-v1"
                report["scene_success_rates"] = pilot_scene_rates(env.task.name, report)
                report["pilot_objective"] = min(report["scene_success_rates"].values())
                report["quality_passed"] = None
                report["quality_rule"] = "CheckpointEval exploration; no formal release claim"
            if config.get("checkpoint_eval_metric") in (
                "navigation-convergence-v1",
                "navigation-convergence-v2",
                "navigation-convergence-v3",
            ):
                report["selection_rule"] = config["checkpoint_eval_metric"]
                scene_outcomes = {}
                for cell in report["cells"].values():
                    for row in cell["episodes"]:
                        scene_outcomes.setdefault(row["subtype"], []).append(row["arrived"])
                report["scene_success_rates"] = {
                    name: float(np.mean(values)) for name, values in scene_outcomes.items()
                }
                apply_quality(report, config)
            save_report(rec.path / "eval" / f"step-{step:010d}.json", report)
            scalars = evaluation_scalars(env.task.name, report)
            scalars["eval/seconds"] = time.monotonic() - t
            if report.get("quality_passed") is not None:
                scalars["eval/quality_passed"] = float(report["quality_passed"])
            rec.log(step, scalars)
            if env.task.name == "racing":
                rec.log(
                    step,
                    {
                        "eval/gates_passed_mean": report.get("gates_passed_mean", 0.0),
                        "eval/collision_rate": report.get("collision_rate", 0.0),
                    },
                )
            replay_directory = rec.path / "rollouts" / f"step-{step:010d}"
            if config["components"]["evaluation"].get("record_replays", False) or config.get(
                "publish_live", False
            ):
                t = time.monotonic()
                export_run_replays(evaluation_env, trace, report, replay_directory)
                rec.log(step, {"record/export_seconds": time.monotonic() - t})
            score = checkpoint_eval_score(
                env.task.name, report, config.get("checkpoint_eval_metric")
            )
            if score > best_score:
                best_score, best = score, report
                save_report(
                    rec.path / "checkpoints" / "best.json",
                    {
                        "path": checkpoint.name,
                        "step": step,
                        "score": list(score),
                        "selection_role": "eval",
                    },
                )
            if config.get("publish_live", False):
                publication = publish_snapshot(
                    replay_directory, rec.path, first=snapshot_count == 0
                )
                live_publications.append({"step": step, **publication})
                save_report(
                    rec.path / "live-publications.json",
                    {"snapshots": live_publications},
                )
                rec.log(
                    step,
                    {"record/live_publication_error": float(publication["status"] == "error")},
                )
                if publication["status"] != "published":
                    print(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "step": step,
                                "optional_live_publication": publication,
                            }
                        ),
                        flush=True,
                    )
            snapshot_count += 1
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "step": step,
                        **scalars,
                        "quality_passed": report.get("quality_passed"),
                    }
                ),
                flush=True,
            )
            rec.phase("compiling" if step == 0 else "training", step=step)
            if time.monotonic() - start > config.get("max_wall_seconds", 3600):
                raise TimeoutError("Declared wall-clock budget reached; snapshots preserved")

        def progress(step, metrics):
            actual_step = int(step)
            if algorithm == "apg":
                epochs = max(config.get("num_evals", 9) - 1, 1)
                actual_step = (
                    int(step)
                    * config["policy_updates"]
                    // epochs
                    * config["num_envs"]
                    * config["horizon_length"]
                )
            clean = {k: float(v) for k, v in metrics.items() if np.asarray(v).size == 1}
            if clean:
                rec.log(actual_step, clean)
            rec.phase("training", step=actual_step)
            print(
                json.dumps({"run_id": run_id, "step": actual_step, "metrics": clean}),
                flush=True,
            )
            if time.monotonic() - start > config.get("max_wall_seconds", 3600):
                raise TimeoutError("Declared wall-clock budget reached at an epoch boundary")

        factory = network_factory(config)
        restore = None
        if warm_start is not None and algorithm == "ppo":
            _, restore, previous = load_policy(warm_start)
            previous_native = native_training_config(previous["config"])
            for key in (
                "algorithm",
                "task",
                "dynamics",
                "hidden_sizes",
                "normalize_observations",
                "distribution_type",
            ):
                if config.get(key) != previous_native.get(key):
                    raise ValueError(f"Warm-start configuration differs: {key}")

        rec.phase("compiling", step=0)
        if algorithm == "ppo":
            maker, params, metrics = ppo_train.train(
                environment=env,
                num_timesteps=config["num_timesteps"],
                num_envs=config["num_envs"],
                episode_length=env.episode_length,
                wrap_env_fn=wrap_for_training,
                action_repeat=1,
                unroll_length=config.get("unroll_length", 16),
                batch_size=config.get("batch_size", 64),
                num_minibatches=config.get("num_minibatches", 16),
                num_updates_per_batch=config.get("num_updates_per_batch", 8),
                learning_rate=config.get("learning_rate", 0.0008),
                entropy_cost=config.get("entropy_cost", 0.007),
                discounting=config.get("discounting", 0.92),
                gae_lambda=config.get("gae_lambda", 0.94),
                clipping_epsilon=config.get("clipping_epsilon", 0.2),
                max_grad_norm=config.get("max_grad_norm", 1.0),
                normalize_observations=config.get("normalize_observations", False),
                num_evals=config.get("num_evals", 9),
                num_eval_envs=4,
                run_evals=False,
                seed=config.get("seed", 0),
                network_factory=factory,
                policy_params_fn=snapshot,
                progress_fn=progress,
                restore_params=restore,
                bootstrap_on_timeout=env.time_limit_kind == "truncation",
                log_training_metrics=False,
            )
            actual_steps = max(
                int(p.stem.split("-")[-1]) for p in (rec.path / "checkpoints").glob("step-*.pkl")
            )
        elif algorithm == "apg":
            epochs = max(config.get("num_evals", 9) - 1, 1)
            if config["policy_updates"] % epochs:
                raise ValueError("APG policy_updates must be divisible by evaluation epochs")

            def capture_factory(*args, **kwargs):
                net = factory(*args, **kwargs)
                init = net.policy_network.init

                def capture(key):
                    p = init(key)
                    normalizer = running_statistics.init_state(
                        specs.Array((env.observation_size,), jnp.float32)
                    )
                    snapshot(0, apg_networks.make_inference_fn(net), (normalizer, p))
                    return p

                return net.replace(
                    policy_network=networks.FeedForwardNetwork(capture, net.policy_network.apply)
                )

            maker, params, metrics = apg_train.train(
                environment=env,
                episode_length=env.episode_length,
                policy_updates=config["policy_updates"],
                horizon_length=config["horizon_length"],
                num_envs=config["num_envs"],
                wrap_env_fn=wrap_for_training,
                num_evals=config.get("num_evals", 9),
                num_eval_envs=4,
                learning_rate=config.get("learning_rate", 0.005),
                normalize_observations=config.get("normalize_observations", False),
                max_gradient_norm=config.get("max_grad_norm", 1.0),
                use_schedule=config.get("use_schedule", True),
                schedule_decay=config.get("schedule_decay", 0.997),
                deterministic_eval=True,
                seed=config.get("seed", 0),
                network_factory=capture_factory,
                progress_fn=progress,
            )
            actual_steps = config["policy_updates"] * config["num_envs"] * config["horizon_length"]
            snapshot(actual_steps, maker, params)

        else:
            trainer = __import__(
                f"drone_playground.learning.algorithms.{algorithm}",
                fromlist=["train"],
            )
            maker, params, metrics = trainer.train(
                env,
                config,
                policy_params_fn=snapshot,
                progress_fn=progress,
                state_directory=rec.path / "training-state",
                restore_state=Path(config["resume"]) if config.get("resume") else None,
            )
            actual_steps = metrics["actual_steps"]

        delta = float(
            np.sqrt(
                sum(
                    float(np.sum((np.asarray(a) - np.asarray(b)) ** 2))
                    for a, b in zip(
                        jax.tree.leaves(initial_params), jax.tree.leaves(params[1]), strict=True
                    )
                )
            )
        )
        if not np.isfinite(delta) or delta == 0:
            raise AssertionError("Training did not produce a finite, nonzero policy update")
        result = {
            "algorithm": algorithm,
            "actual_steps": actual_steps,
            "actor_parameter_delta_l2": delta,
            "snapshots": snapshot_count,
            "best_checkpoint_eval_result": best,
            "elapsed_seconds": time.monotonic() - start,
            "quality_passed": best.get("quality_passed") if best else None,
            "engineer_passed": True,
            "full_budget_completed": True,
            "live_publications": live_publications,
            "trainer_metrics": {
                k: float(np.asarray(v))
                for k, v in metrics.items()
                if np.asarray(v).size == 1
                and np.issubdtype(np.asarray(v).dtype, np.number)
                and np.isfinite(np.asarray(v)).all()
            },
        }
        rec.finish("completed", **result)
        return result
    except BaseException as error:
        traceback.print_exc()
        rec.finish(
            "failed",
            error=repr(error),
            elapsed_seconds=time.monotonic() - start,
        )
        raise
    finally:
        if env is not None:
            env.close()
        if evaluation_env is not None:
            evaluation_env.close()
        console.__exit__(*sys.exc_info())


def train_experiment(config, root, run_id):
    """Select the requested learning algorithm and execute its training run."""
    return train(
        native_training_config(config),
        root,
        run_id,
        config["runtime"]["device"],
        config["training"].get("warm_start"),
    )
