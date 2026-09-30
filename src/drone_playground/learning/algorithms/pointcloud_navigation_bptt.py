"""Direct recurrent navigation training with explicit domain-adaptation provenance."""

import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from drone_playground.environments.tasks.pointcloud_control import delayed_step
from drone_playground.evaluation.tracking import save_report, tree_digest
from drone_playground.learning.algorithms.pointcloud_bptt import continuation_contract, initialize
from drone_playground.runs.pointcloud import load_training_state, save_training_state
from drone_playground.runtime.jax_runner import recurrent_scan


def augment_point_measurements(points, valid, key, standard_deviation_m):
    """Add seeded measurement noise to valid points; zero preserves frozen recipes."""
    if not math.isfinite(standard_deviation_m) or standard_deviation_m < 0:
        raise ValueError("Point noise standard deviation must be finite and nonnegative")
    if standard_deviation_m == 0:
        return points
    noise = jax.random.normal(key, points.shape, dtype=points.dtype) * standard_deviation_m
    return points + jnp.where(valid[..., None], noise, 0.0)


def rollout_loss(task, network, parameters, key, count, horizon):
    bank, physical, clocks, speeds, delays = task.training_initial(key, count)
    initial_clearance = task.clearance(bank, physical, clocks)
    hidden = jnp.zeros((count, network.hidden_size))
    previous = jnp.zeros((count, 3))

    def step(carry, index):
        current, memory, last = carry
        now = clocks + index * task.dt
        points, valid, proprio, _ = task.observation(bank, current, now, speeds)
        auxiliary = jnp.float32(0)
        if task.config.get("method", {}).get("implementation") == "depth_recurrent":
            prediction, memory = network.apply(
                parameters, points, valid, proprio, memory, method=network.predict
            )
            action = prediction[..., :3] - prediction[..., 3:]
            auxiliary = jnp.mean(
                (prediction[..., 3:] - jax.lax.stop_gradient(proprio[..., :3])) ** 2
            )
            auxiliary *= task.config["algorithm"]["velocity_prediction_weight"]
        else:
            points = augment_point_measurements(
                points,
                valid,
                jax.random.fold_in(key, index),
                float(task.config["training"].get("point_noise_std_m", 0.0)),
            )
            action, memory = network.apply(parameters, points, valid, proprio, memory)
        command = task.command(action, current)
        nxt, _ = delayed_step(
            task.model, current, command, last, delays, task.physics_dt, task.substeps
        )
        clearance = task.clearance(bank, nxt, now + task.dt)
        loss, parts = task.objective(
            current, nxt, bank.goal, speeds, clearance, command, last, task.dt
        )
        loss = loss + auxiliary
        parts.update(
            velocity_prediction_loss=auxiliary,
            loss=loss,
            endpoint_collision_fraction=jnp.mean((clearance < 0).astype(jnp.float32)),
            mean_speed=jnp.mean(jnp.linalg.norm(nxt.vel, axis=-1)),
        )
        return (nxt, memory, command), parts

    _, metrics = recurrent_scan(step, (physical, hidden, previous), horizon, rematerialize=True)
    metrics = {name: jnp.mean(value) for name, value in metrics.items()}
    metrics["initial_min_clearance"] = jnp.min(initial_clearance)
    return metrics["loss"], metrics


def adaptation_contract(config):
    value = continuation_contract(config)
    value["training"].pop("warm_start", None)
    # A missing field in legacy states and an explicit null have the same recipe.
    if value["training"].get("scene") is None:
        value["training"].pop("scene", None)
    return value


def train(config, root, run_id):
    from drone_playground.composition import build_environment
    from drone_playground.evaluation.pointcloud_navigation import (
        PointCloudNavigationEvaluator,
        navigation_development_selection,
    )
    from drone_playground.runs.console import capture_console
    from drone_playground.runs.record import RunRecorder

    settings, algo = config["training"], config["algorithm"]
    count, horizon, target = (
        int(settings["num_envs"]),
        int(algo["horizon_length"]),
        int(settings["policy_updates"]),
    )
    stop = int(settings.get("stop_after_updates") or target)
    if not 1 <= stop <= target:
        raise ValueError("Stage end must lie within the declared update budget")
    task = build_environment(config, config["runtime"]["device"])
    state, network, optimizer = initialize(task, config)
    optimizer = optax.chain(optax.clip_by_global_norm(float(algo["max_grad_norm"])), optimizer)
    state = state.replace(opt_state=optimizer.init(state.params))
    provenance = None
    if settings.get("warm_start"):
        source, metadata = load_training_state(settings["warm_start"])
        if metadata["config"]["network"] != config["network"]:
            raise ValueError(
                "Parameter warm start cannot silently change point-cloud input conditioning"
            )
        for field in ("sensor", "observation", "execution"):
            if metadata["config"]["env"][field] != config["env"][field]:
                raise ValueError(f"Warm-start sensor/control/plant contract differs: {field}")
        if metadata["config"]["env"]["task"]["freq"] != task.freq:
            raise ValueError("Warm start must preserve the recurrent policy clock")
        state = state.replace(params=source.params, opt_state=optimizer.init(source.params))
        provenance = dict(
            checkpoint=str(Path(settings["warm_start"]).resolve()),
            source_updates=int(source.updates),
            parameter_sha256=metadata["parameter_sha256"],
            optimizer_and_rng="reinitialized",
            navigation_visible_to_adaptation=True,
            transport_and_integration="new explicit 25–50ms, 500Hz adaptation recipe",
        )
    if settings.get("resume"):
        restored, metadata = load_training_state(settings["resume"])
        if adaptation_contract(metadata["config"]) != adaptation_contract(config):
            raise ValueError("Exact continuation changes the adaptation contract")
        state = restored
        provenance = dict(
            exact_resume=str(Path(settings["resume"]).resolve()),
            preceding_parameter_origin=metadata["config"]["training"].get("warm_start"),
        )
    first_update = int(state.updates)
    if first_update >= stop:
        raise ValueError("This continuation must execute new updates")
    initial_parameters = jax.tree.map(np.asarray, state.params)
    evaluator = PointCloudNavigationEvaluator(
        task,
        network,
        int(settings.get("development_seed_start", 20000)),
        int(settings["development_episodes"]),
        config["evaluation"]["commanded_speed"],
        settings.get("development_initial_conditions"),
    )
    milestones = {
        int(value) for value in np.linspace(first_update, stop, max(2, settings["num_evals"]))
    }

    @jax.jit
    def update(current):
        key, sample = jax.random.split(current.key)
        (_, metrics), gradient = jax.value_and_grad(
            lambda parameters: rollout_loss(task, network, parameters, sample, count, horizon),
            has_aux=True,
        )(current.params)
        change, optimizer_state = optimizer.update(gradient, current.opt_state, current.params)
        return current.replace(
            params=optax.apply_updates(current.params, change),
            opt_state=optimizer_state,
            key=key,
            updates=current.updates + 1,
        ), {**metrics, "gradient_norm": optax.global_norm(gradient)}

    with RunRecorder(
        root, run_id, config, task_id="navigation-convergence/pointcloud-training"
    ) as rec:
        with capture_console(rec.path / "console.log"):
            save_report(rec.path / "scene-manifest.json", task.manifest)
            save_report(rec.path / "training-scene-manifest.json", task.training_manifest)
            save_report(rec.path / "components.json", task.component_identity)
            save_report(
                rec.path / "training-identity.json",
                dict(
                    adapter=task.settings["adapter_identity"],
                    warm_start=provenance,
                    source_paper_reproduction=False,
                    policy_hz=task.freq,
                    physics_hz=task.physics_freq,
                    initialization="safe departure/near-goal/global course states with random dynamic phases",
                    training_bank_digest=task.training_bank.digest(),
                    evaluation_bank_digest=task.bank.digest(),
                    training_geometry_separate=task.training_bank.digest() != task.bank.digest(),
                    sensor_points=task.sensor.points_per_frame,
                    training_termination="continuous soft collision objective, no event truncation",
                    evaluation_termination="first physical collision/arrival/boundary/nonfinite/300s event",
                    training_clearance_sampling_hz=task.freq,
                    randomized_delay_ms=config["runtime"]["action_delay_ms"],
                ),
            )
            best = best_report = None

            def snapshot(current):
                nonlocal best, best_report
                updates = int(current.updates)
                path = rec.path / "training-state" / f"update-{updates:07d}.pkl"
                save_training_state(path, current, config)
                rec.phase("development", step=updates * count * horizon)
                report, trace = evaluator.run(current.params)
                report.update(
                    updates=updates,
                    step=updates * count * horizon,
                    checkpoint=str(path),
                    split="dev",
                )
                if settings.get("development_metric") == "release-pilot-v1":
                    report["pilot_objective"] = min(report["scene_success_rates"].values())
                    report["selection_rule"] = "release-pilot-v1"
                if settings.get("development_metric") in ("navigation-development-v2", "navigation-development-primary-v1"):
                    report.update(navigation_development_selection(report, settings['development_metric']))
                save_report(rec.path / "eval" / f"update-{updates:07d}.json", report)
                remaining = float(np.mean([r["final_goal_distance_m"] for r in report["episodes"]]))
                score = (
                    report["success_rate"],
                    -report["failure_rate"],
                    -remaining,
                    -report["constrained_time_mean_s"],
                )
                if settings.get("development_metric") == "release-pilot-v1":
                    score = (report["pilot_objective"],)
                if settings.get("development_metric") in ("navigation-development-v2", "navigation-development-primary-v1"):
                    score = tuple(report["score"])
                if best is None or score > tuple(best["score"]):
                    best = dict(
                        checkpoint=str(path.resolve()),
                        updates=updates,
                        score=list(score),
                        parameter_sha256=tree_digest(current.params),
                        selection_split="dev",
                    )
                    best_report = report
                    save_report(rec.path / "best.json", best)
                evaluator.export(trace, rec.path / "rollouts" / f"update-{updates:07d}")
                rec.log(
                    updates * count * horizon,
                    dict(
                        dev_arrived=report["arrived"],
                        dev_trials=report["num_trials"],
                        dev_failure_rate=report["failure_rate"],
                        dev_remaining_m=remaining,
                        dev_quality_passed=float(report["quality_passed"]),
                    ),
                )
                return updates

            started = time.monotonic()
            last_snapshot = snapshot(state)
            first_seconds = net_seconds = 0.0
            metrics = {}
            for iteration in range(first_update + 1, stop + 1):
                rec.phase("training", step=int(state.updates) * count * horizon)
                tick = time.monotonic()
                state, raw = update(state)
                metrics = {name: float(value) for name, value in raw.items()}
                elapsed = time.monotonic() - tick
                if iteration == first_update + 1:
                    first_seconds = elapsed
                else:
                    net_seconds += elapsed
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"Non-finite adaptation update {iteration}: {metrics}")
                if metrics["initial_min_clearance"] < 0:
                    raise RuntimeError("A training rollout was initialized inside an obstacle")
                if iteration % 10 == 0 or iteration in milestones:
                    rec.log(iteration * count * horizon, {**metrics, "updates": iteration})
                if iteration in milestones:
                    last_snapshot = snapshot(state)
                if time.monotonic() - started >= settings["max_wall_seconds"]:
                    break
            if last_snapshot != int(state.updates):
                snapshot(state)
            delta = float(
                np.sqrt(
                    sum(
                        np.square(np.asarray(a) - b).sum()
                        for a, b in zip(
                            jax.tree.leaves(state.params),
                            jax.tree.leaves(initial_parameters),
                            strict=True,
                        )
                    )
                )
            )
            if not np.isfinite(delta) or delta <= 0:
                raise RuntimeError("Adaptation did not make a finite policy update")
            result = dict(
                actual_updates=int(state.updates),
                target_updates=target,
                actual_steps=int(state.updates) * count * horizon,
                session_steps=(int(state.updates) - first_update) * count * horizon,
                full_budget_completed=int(state.updates) == target,
                requested_stage_completed=int(state.updates) == stop,
                actor_parameter_delta_l2=delta,
                selected=best,
                best_development_result=best_report,
                selected_checkpoint_has_new_updates=best["updates"] > first_update,
                quality_passed=best_report["quality_passed"],
                warm_start=provenance,
                wall_seconds=time.monotonic() - started,
                trainer_metrics={
                    **metrics,
                    "compile_and_first_update_seconds": first_seconds,
                    "net_update_seconds": net_seconds,
                },
            )
            rec.finish("completed" if result["full_budget_completed"] else "paused", **result)
    task.close()
    return result
