"""Run native Brax PPO/APG and the local SHAC extension on Crazyflow tasks.

PPO has native snapshot callbacks. Brax 0.14.2 APG supplies progress metrics
but no parameter callback: initial/final snapshots are saved without modifying
its optimizer loop. This distinction is recorded in each run configuration.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training import networks, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.apg import train as apg_train
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from flax import linen

from drone_playground.evaluation.tracking import (
    PolicyEvaluator,
    save_report,
    select_replays,
    tree_digest,
)
from drone_playground.tasks.tracking import TrackingEnv, wrap_for_training


def network_factory(config: dict):
    sizes = tuple(config.get("hidden_sizes", [64, 64]))
    if config["algorithm"] == "ppo":
        return functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=sizes,
            value_hidden_layer_sizes=sizes,
            activation=linen.elu,
            init_noise_std=config.get("init_noise_std", 0.367879),
            distribution_type=config.get("distribution_type", "tanh_normal"),
            noise_std_type="log",
            state_dependent_std=False,
            mean_kernel_init_fn=jax.nn.initializers.orthogonal,
            mean_kernel_init_kwargs={"scale": 0.01},
        )
    if config["algorithm"] in {"apg", "shac"}:
        return functools.partial(
            apg_networks.make_apg_networks,
            hidden_layer_sizes=sizes,
            activation=linen.elu,
            layer_norm=config.get("layer_norm", True),
        )
    raise ValueError(f"Unsupported algorithm: {config['algorithm']}")


def save_policy(directory: Path, params, config: dict, step: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{int(step):010d}.pkl"
    temp = path.with_suffix(".tmp")
    model.save_params(str(temp), jax.tree.map(np.asarray, params))
    temp.replace(path)
    metadata = {
        "step": int(step),
        "config": config,
        "observation_size": config.get("observation_size", 43),
        "action_size": 4,
        "checkpoint_kind": "inference-parameters",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameter_sha256": tree_digest(params),
        "continuation": (
            "PPO warm start only; optimizer/RNG are reinitialized"
            if config["algorithm"] == "ppo"
            else (
                "SHAC inference; full continuation is stored in training-state/"
                if config["algorithm"] == "shac"
                else "APG inference only; native trainer has no restore hook"
            )
        ),
    }
    save_report(path.with_suffix(".json"), metadata)
    return path


def load_policy(path: str | Path):
    """Load a locally trusted Brax checkpoint and verify its sidecar hash."""
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Checkpoint digest does not match its metadata")
    config = metadata["config"]
    preprocess = (
        running_statistics.normalize
        if config.get("normalize_observations", False)
        else types.identity_observation_preprocessor
    )
    net = network_factory(config)(
        metadata["observation_size"], metadata["action_size"], preprocess_observations_fn=preprocess
    )
    maker = (
        ppo_networks.make_inference_fn
        if config["algorithm"] == "ppo"
        else apg_networks.make_inference_fn
    )
    # Preserve dtype metadata such as Brax's float64 NumPy std_eps scalar.
    # JAX places the arrays when the policy is compiled; eagerly casting every
    # leaf would change the checkpoint's content hash under default float32 mode.
    params = model.load_params(str(path))
    if tree_digest(params) != metadata["parameter_sha256"]:
        raise ValueError("Loaded parameter content changed")
    return maker(net), params, metadata


def make_task(config: dict, device: str, split: str = "train", count: int = 32) -> TrackingEnv:
    seeds = {"train": 10000, "dev": 20000, "heldout": 30000}
    if split not in seeds:
        raise ValueError(f"Unknown dataset split: {split}")
    task_class = TrackingEnv
    task_options = {"numerical_guard": config.get("numerical_guard", False)}
    if config.get("task") == "racing":
        from drone_playground.tasks.racing import RacingEnv

        task_class = RacingEnv
        task_options = {}
    return task_class(
        task=config.get("task", "figure8"),
        dynamics=config.get("dynamics", "so_rpy"),
        drone=config.get("drone"),
        freq=config.get("freq", 50),
        device=device,
        reference_seed=seeds[split],
        reference_count=config.get("reference_count", 256) if split == "train" else count,
        **task_options,
    )


def development_score(task: str, report: dict) -> tuple:
    """Select only on development data; race progress breaks incomplete ties."""
    if task == "racing":
        return (report["completed"], report["gates_passed_mean"], -report["rmse_all_mean"])
    return (report["completed"], -report["rmse_all_mean"])


def make_evaluator(env, make_policy, seeds):
    if env.task == "racing":
        from drone_playground.evaluation.racing import RaceEvaluator

        return RaceEvaluator(env, make_policy, seeds)
    return PolicyEvaluator(env, make_policy, seeds)


def train(
    config: dict, root: Path, run_id: str, device: str = "gpu", warm_start: Path | None = None
) -> dict:
    """Train the declared complete budget and record real snapshots and evaluations."""
    from drone_playground.runs.console import capture_console
    from drone_playground.runs.record import RunRecorder
    from drone_playground.runs.rscope_io import export_rollout, publish_snapshot

    config = dict(config)
    algorithm = config["algorithm"]
    if algorithm not in {"ppo", "apg", "shac"}:
        raise ValueError("Supported algorithms are PPO, APG and SHAC")
    config.update(
        device=device,
        snapshot_schedule="initial-and-final" if algorithm == "apg" else "per-epoch",
        trainer=(
            "drone_playground.learning.shac"
            if algorithm == "shac"
            else "brax.training.agents." + algorithm
        ),
        reset_contract="fresh-same-step-with-terminal-observation",
        timeout_contract=(
            "SHAC pre-reset terminal-value bootstrap; true termination masks value"
            if algorithm == "shac"
            else "native Brax GAE masks truncated transition; time_out bootstrap disabled"
        ),
        task_protocol={
            "figure8": "crazyflow-figure8-v1",
            "random": "lsy-random-spline-v1",
            "racing": "lsy-level0-reference-tracking-v1",
        }[config["task"]],
    )
    if config.get("numerical_guard", False):
        config["task_protocol"] += "+finite-square-v1"
    source_root = Path(crazyflow.__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dependency_patch = subprocess.check_output(
        ["git", "-C", str(source_root), "diff", "HEAD", "--"]
    )
    config["crazyflow_code"] = {
        "commit": revision,
        "working_tree_patch_sha256": hashlib.sha256(dependency_patch).hexdigest(),
    }
    config["actual_devices"] = [str(device) for device in jax.devices()]
    task_id = "07" if config["task"] == "racing" else ("04" if algorithm == "shac" else "05")
    rec = RunRecorder(root, run_id, config, task_id=task_id)
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
        env = make_task(config, device)
        if config["task"] == "racing":
            save_report(rec.path / "native-task-config.json", env.config.to_dict())
        evaluation_env = make_task(config, device, "dev", 32)
        dev_seeds = list(range(20000, 20032))
        save_report(
            rec.path / "eval" / "dev-cases.json",
            {
                "split": "dev",
                "reset_seeds": dev_seeds,
                "reference_seeds": dev_seeds if env.task == "random" else [20000],
                "quality_rule": (
                    "at least29/32 complete all native gate passes"
                    if env.task == "racing"
                    else "at least29/32 complete, completed mean RMSE <=0.25m"
                ),
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
                evaluator = make_evaluator(evaluation_env, make_policy, dev_seeds)
            t = time.monotonic()
            report, trace = evaluator.run(params)
            report.update(step=step, checkpoint=str(checkpoint.relative_to(rec.path)), split="dev")
            save_report(rec.path / "eval" / f"step-{step:010d}.json", report)
            rec.log(
                step,
                {
                    "eval/completion_rate": report["completion_rate"],
                    "eval/rmse_all_m": report["rmse_all_mean"],
                    "eval/return": report["return_mean"],
                    "eval/seconds": time.monotonic() - t,
                    "eval/quality_passed": float(report["quality_passed"]),
                },
            )
            if env.task == "racing":
                rec.log(
                    step,
                    {
                        "eval/gates_passed_mean": report.get("gates_passed_mean", 0.0),
                        "eval/collision_rate": report.get("collision_rate", 0.0),
                    },
                )
            t = time.monotonic()
            replay = select_replays(trace, report)
            replay["metrics"]["training_step"] = np.full_like(replay["reward"], step)
            replay_directory = rec.path / "rollouts" / f"step-{step:010d}"
            export_rollout(evaluation_env.sim, replay_directory, replay)
            rec.log(step, {"record/export_seconds": time.monotonic() - t})
            score = development_score(env.task, report)
            if score > best_score:
                best_score, best = score, report
                save_report(
                    rec.path / "checkpoints" / "best.json",
                    {
                        "path": checkpoint.name,
                        "step": step,
                        "score": list(score),
                        "selection_split": "dev",
                    },
                )
            if config.get("publish_live", False):
                publication = publish_snapshot(
                    replay_directory, rec.path, first=snapshot_count == 0
                )
                live_publications.append({"step": step, **publication})
                save_report(rec.path / "live-publications.json", {"snapshots": live_publications})
                rec.log(
                    step, {"record/live_publication_error": float(publication["status"] == "error")}
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
                        "completed": report["completed"],
                        "rmse": report["rmse_all_mean"],
                        "quality_passed": report["quality_passed"],
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
            print(json.dumps({"run_id": run_id, "step": actual_step, "metrics": clean}), flush=True)
            if time.monotonic() - start > config.get("max_wall_seconds", 3600):
                raise TimeoutError("Declared wall-clock budget reached at an epoch boundary")

        factory = network_factory(config)
        restore = None
        if warm_start is not None:
            if algorithm != "ppo":
                raise ValueError("Native APG does not expose a restore hook")
            _, restore, previous = load_policy(warm_start)
            for key in (
                "algorithm",
                "task",
                "dynamics",
                "hidden_sizes",
                "normalize_observations",
                "distribution_type",
            ):
                if config.get(key) != previous["config"].get(key):
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
                bootstrap_on_timeout=False,
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
            from drone_playground.learning import shac

            maker, params, metrics = shac.train(
                env,
                config,
                policy_params_fn=snapshot,
                progress_fn=progress,
                state_directory=rec.path / "training-state",
            )
            actual_steps = metrics["actual_steps"]

        delta = float(
            np.sqrt(
                sum(
                    float(np.sum((np.asarray(a) - np.asarray(b)) ** 2))
                    for a, b in zip(jax.tree.leaves(initial_params), jax.tree.leaves(params[1]))
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
            "best_development_result": best,
            "elapsed_seconds": time.monotonic() - start,
            "quality_passed": bool(best and best["quality_passed"]),
            "engineer_passed": True,
            "full_budget_completed": True,
            "live_publications": live_publications,
            "trainer_metrics": {
                k: float(v)
                for k, v in metrics.items()
                if np.asarray(v).size == 1 and np.isfinite(float(v))
            },
        }
        rec.finish("completed", **result)
        return result
    except BaseException as error:
        traceback.print_exc()
        rec.finish("failed", error=repr(error), elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        if env is not None:
            env.close()
        if evaluation_env is not None:
            evaluation_env.close()
        console.__exit__(*sys.exc_info())
