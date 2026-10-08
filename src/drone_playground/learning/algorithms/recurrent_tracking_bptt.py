"""Explicit PointNet/GRU control-task transfer, using trajectory gradients.

This is not the paper navigation objective: the canonical task reference supplies
position/velocity targets, while the paper encoder, recurrent policy and lag
model remain intact. Commands arrive after a seeded 25--50ms actuator delay.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from drone_playground.control.delay import delayed_step
from drone_playground.learning.checkpointing import (
    restore_recurrent_state,
    run_recurrent_updates,
    save_recurrent_snapshot,
)
from drone_playground.runtime.jax_runner import recurrent_scan


def loss_function(task, network, params, key, count, horizon):
    """Compute recurrent trajectory-tracking loss across an episode batch."""
    physical, times, delay = task.training_initial(key, count, horizon)
    hidden = jnp.zeros((count, network.hidden_size))
    previous = jnp.zeros((count, 3))
    settings = task.loss

    def step(carry, index):
        state, memory, last = carry
        timestamp = times + index * task.dt
        points, valid, proprio = task.measure(state, timestamp)
        action, memory = network.apply(params, points, valid, proprio, memory)
        command = task.command(action, state)
        nxt, positions = delayed_step(
            task.dynamics,
            state,
            command,
            last,
            delay,
            task.physics_dt,
            task.substeps,
        )
        target_position, target_velocity = task.reference(timestamp + task.dt)
        clearance = jax.vmap(task.clearance)(positions)
        parts = dict(
            position=jnp.mean(jnp.sum((nxt.pos - target_position) ** 2, axis=-1)),
            velocity=jnp.mean(jnp.sum((nxt.vel - target_velocity) ** 2, axis=-1)),
            acceleration=jnp.mean(jnp.sum(command**2, axis=-1)),
            jerk=jnp.mean(jnp.sum(((command - last) / task.dt) ** 2, axis=-1)),
            collision=jnp.mean(jnp.square(jax.nn.relu(settings["collision_margin_m"] - clearance))),
        )
        return (nxt, memory, command), parts

    _, parts = recurrent_scan(step, (physical, hidden, previous), horizon, rematerialize=True)
    metrics = {name: jnp.mean(value) for name, value in parts.items()}
    loss = sum(settings[name + "_loss"] * value for name, value in metrics.items())
    return loss, {**metrics, "loss": loss}


def train(config, root, run_id):
    """Train a declared control transfer; select only on fixed checkpoint_eval seeds."""
    from pathlib import Path

    from drone_playground.artifacts.console import capture_console
    from drone_playground.artifacts.record import RunRecorder
    from drone_playground.artifacts.reporting import save_report
    from drone_playground.artifacts.training_state import load_training_state
    from drone_playground.environments.factory import build_environment
    from drone_playground.evaluation.tracking.acceleration import ControlEvaluator, export_replays
    from drone_playground.learning.algorithms.recurrent_bptt import (
        initialize,
    )

    settings, algorithm = config["training"], config["algorithm"]
    count, horizon = int(settings["num_envs"]), int(algorithm["horizon_length"])
    target = int(settings["policy_updates"])
    stop = int(settings.get("stop_after_updates") or target)
    milestones = {int(value) for value in np.linspace(0, target, max(2, settings["num_evals"]))}
    milestones.add(stop)
    task = None
    evaluation_env = None
    with (
        RunRecorder(
            root,
            run_id,
            config,
            task_id="final-acceptance/pointcloud-control-training",
        ) as rec,
        capture_console(rec.path / "console.log"),
    ):
        try:
            rec.phase("initializing")
            env = build_environment(config, config["runtime"]["device"], "train", count)
            task = env.task
            rec.record_environment(env)
            state, network, optimizer = initialize(task, config)
            clip = algorithm.get("max_grad_norm")
            if clip is not None:
                optimizer = optax.chain(optax.clip_by_global_norm(float(clip)), optimizer)
                state = state.replace(opt_state=optimizer.init(state.params))
            warm_provenance = None
            if settings.get("warm_start"):
                source, metadata = load_training_state(settings["warm_start"])
                for field in ("method", "network"):
                    if metadata["config"][field] != config[field]:
                        raise ValueError(f"Point-cloud warm-start contract differs: {field}")
                for field in ("sensor", "controller", "dynamics"):
                    if metadata["config"]["env"][field] != config["env"][field]:
                        raise ValueError(f"Point-cloud warm-start input/plant differs: {field}")
                state = state.replace(
                    params=source.params,
                    opt_state=optimizer.init(source.params),
                )
                warm_provenance = dict(
                    checkpoint=str(Path(settings["warm_start"]).resolve()),
                    source_updates=int(source.updates),
                    source_parameter_sha256=metadata["parameter_sha256"],
                    optimizer_and_rng="reinitialized",
                )
            state, restored = restore_recurrent_state(state, config)
            start_updates = int(state.updates)
            if start_updates >= stop:
                raise ValueError("Continuation must perform at least one additional update")
            initial = jax.tree.map(np.asarray, state.params)
            checkpoint_eval_count = int(settings["checkpoint_eval_episodes"])
            from drone_playground.benchmarks import checkpoint_eval_seeds

            evaluation_env = build_environment(
                config,
                config["runtime"]["device"],
                "eval",
                checkpoint_eval_count,
            )
            evaluation_task = evaluation_env.task
            evaluator = ControlEvaluator(
                evaluation_task,
                network,
                checkpoint_eval_seeds(settings, checkpoint_eval_count),
            )
            save_report(rec.path / "components.json", env.component_identity)
            save_report(rec.path / "geometry.json", task.geometry_identity)
            save_report(
                rec.path / "sensor-calibration.json",
                task.sensor_calibration,
            )
            save_report(
                rec.path / "training-identity.json",
                dict(
                    provenance=task.settings["provenance"],
                    warm_start=warm_provenance,
                    policy_hz=task.freq,
                    physics_hz=task.physics_freq,
                    randomized_delay_ms=config["runtime"]["action_delay_ms"],
                    initialization="random reference phases and perturbed task initial states",
                    gradient="BPTT through delayed commands and declared lag dynamics",
                    sensor_state_gradient="detached",
                    source_objective_used=False,
                ),
            )

            @jax.jit
            def update(current):
                key, sample = jax.random.split(current.key)
                (loss, parts), grad = jax.value_and_grad(loss_function, argnums=2, has_aux=True)(
                    task, network, current.params, sample, count, horizon
                )
                change, optimizer_state = optimizer.update(grad, current.opt_state, current.params)
                return current.replace(
                    params=optax.apply_updates(current.params, change),
                    opt_state=optimizer_state,
                    key=key,
                    updates=current.updates + 1,
                ), {
                    **parts,
                    "gradient_norm": optax.global_norm(grad),
                    "loss": loss,
                }

            best, best_report = restored.get("selection"), restored.get("selection_report")

            def snapshot(current):
                nonlocal best, best_report
                updates = int(current.updates)
                step = updates * count * horizon
                checkpoint = rec.path / "training-state" / f"update-{updates:07d}.pkl"
                rec.phase("checkpoint_eval", step, updates=updates)
                report, trace = evaluator.run(current.params)
                from drone_playground.benchmarks import apply_quality

                apply_quality(report, config)
                report.update(
                    role="eval",
                    step=step,
                    updates=updates,
                    checkpoint=str(checkpoint),
                )
                score = (
                    report["completed"],
                    report.get("gates_passed_mean", 0.0),
                    -report["rmse_all_mean"],
                )
                best, best_report = save_recurrent_snapshot(
                    checkpoint,
                    current,
                    config,
                    report,
                    score,
                    best,
                    best_report,
                    report_path=rec.path / "eval" / f"step-{step:010d}.json",
                    fields={"step": step},
                )
                if config["evaluation"].get("record_replays", False):
                    export_replays(
                        task,
                        trace,
                        report,
                        rec.path / "rollouts" / f"step-{step:010d}",
                    )
                print(
                    {
                        "update": updates,
                        "checkpoint_eval_completed": report["completed"],
                        "checkpoint_eval_rmse": report["rmse_all_mean"],
                        "quality": report["quality_passed"],
                    },
                    flush=True,
                )

            state, latest_metrics, execution = run_recurrent_updates(
                state,
                update,
                snapshot,
                config,
                rec,
                steps_per_update=count * horizon,
                milestones=milestones,
            )
            first_seconds = execution["durations"][0]
            net_seconds = sum(execution["durations"][1:])
            delta = float(
                np.sqrt(
                    sum(
                        np.square(np.asarray(a) - b).sum()
                        for a, b in zip(
                            jax.tree.leaves(state.params),
                            jax.tree.leaves(initial),
                            strict=True,
                        )
                    )
                )
            )
            if not np.isfinite(delta) or delta <= 0:
                raise RuntimeError("Control training did not change finite actor parameters")
            full = int(state.updates) == target
            result = dict(
                actual_updates=int(state.updates),
                target_updates=target,
                actual_steps=int(state.updates) * count * horizon,
                session_steps=(int(state.updates) - start_updates) * count * horizon,
                target_steps=target * count * horizon,
                full_budget_completed=full,
                actor_parameter_delta_l2=delta,
                selected=best,
                best_checkpoint_eval_result=best_report,
                selected_checkpoint_has_new_updates=best["updates"] > start_updates,
                quality_passed=best_report["quality_passed"],
                engineer_passed=True,
                warm_start=warm_provenance,
                trainer_metrics=dict(
                    **latest_metrics,
                    compile_and_first_update_seconds=first_seconds,
                    net_update_seconds=net_seconds,
                ),
                recipe_identity=task.settings["provenance"],
            )
            rec.finish("completed" if full else "paused", **result)
            return result
        finally:
            if evaluation_env is not None:
                evaluation_env.close()
            if task is not None:
                env.close()
