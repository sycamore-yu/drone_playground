"""Recurrent, rematerialized direct policy training for the point-cloud paper.

This trainer uses the existing JAX/Flax/Optax/Brax stack and run recorder. Its
complete recipe and reconstruction assumptions live in the experiment config.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from drone_playground.artifacts.console import capture_console
from drone_playground.artifacts.record import RunRecorder
from drone_playground.artifacts.reporting import save_report
from drone_playground.environments.scenes.procedural_navigation import make_bank
from drone_playground.learning.checkpointing import (
    restore_recurrent_state,
    run_recurrent_updates,
    save_recurrent_snapshot,
)
from drone_playground.networks.factory import build_network
from drone_playground.numerics import euclidean_norm
from drone_playground.runtime.jax_runner import recurrent_scan


@struct.dataclass
class TrainingState:
    params: object
    opt_state: object
    key: jax.Array
    updates: jax.Array


def initialize(task, config):
    """Initialize recurrent learner parameters and episode state."""
    network = build_network(config["network"])
    key, init_key = jax.random.split(jax.random.PRNGKey(config["training"]["seed"]))
    image_shape = getattr(network, "input_shape", None)
    params = network.init(
        init_key,
        jnp.zeros((1, *image_shape) if image_shape else (1, 1, 3)),
        jnp.ones((1, *image_shape) if image_shape else (1, 1), bool),
        jnp.zeros((1, 10)),
        jnp.zeros((1, network.hidden_size)),
    )
    algo = config["algorithm"]
    optimizer = optax.adamw(
        algo["learning_rate"],
        b1=algo["betas"][0],
        b2=algo["betas"][1],
        eps=algo["epsilon"],
        weight_decay=algo["weight_decay"],
    )
    return (
        TrainingState(params, optimizer.init(params), key, jnp.int32(0)),
        network,
        optimizer,
    )


def rollout_objective(task, network, params, bank, speeds, horizon, reset_key=None):
    """Compute the differentiable recurrent rollout loss over a fixed horizon."""
    state = task.initial_state(bank, reset_key)
    memory = jnp.zeros((bank.num_instances, network.hidden_size))

    def step(carry, index):
        physical, hidden = carry
        timestamp = index * task.dt
        points, valid, proprio, target = task.measure(
            bank, physical, jnp.full((bank.num_instances,), timestamp), speeds
        )
        body_action, hidden = network.apply(params, points, valid, proprio, hidden)
        command = task.command(body_action, physical)
        nxt = task.dynamics.step(physical, task.controller.apply(physical, command), task.dt)
        clearance, approaching = task.proximity(bank, nxt, timestamp + task.dt)
        row = dict(
            velocity=nxt.vel,
            target_velocity=target,
            acceleration=command,
            clearance=clearance,
            approaching_speed=approaching,
        )
        return (nxt, hidden), row

    _, trace = recurrent_scan(step, (state, memory), horizon, rematerialize=True)
    loss, parts = task.loss(trace, task.dt)
    metrics = {
        **parts,
        "loss": loss,
        "mean_speed": jnp.mean(euclidean_norm(trace["velocity"])),
        "min_clearance": jnp.min(trace["clearance"]),
        "step_collision_fraction": jnp.mean((trace["clearance"] < 0).astype(jnp.float32)),
    }
    return loss, metrics


def make_update(task, network, optimizer, config):
    """Build the compiled gradient update for a recurrent policy."""
    count = config["training"]["num_envs"]
    horizon = config["algorithm"]["horizon_length"]
    command_distribution = (
        config["training"].get("command_distribution")
        or config["env"]["task"]["command_distribution"]
    )
    lower, upper = command_distribution.get(
        "speed_range_mps",
        config["env"]["task"]["command_distribution"]["speed_range_mps"],
    )
    distribution = config["training"].get("scene_distribution") or {"type": "procedural"}
    fixed_bank = None
    if distribution["type"] != "procedural":
        if hasattr(task.scene, "build"):
            fixed_bank, _ = make_bank(task.scene, config["training"]["seed"], count)
        elif distribution["type"] == "generated":
            fixed_bank = task.scene.sample(jax.random.PRNGKey(config["training"]["seed"]), count)
        else:
            raise ValueError(
                "Fixed training distribution requires a scene bank; use generated or "
                "procedural for primitive sampling"
            )

    @jax.jit
    def update(state):
        key, scene_key, speed_key, reset_key = jax.random.split(state.key, 4)
        bank = (
            task.scene.sample(scene_key, count)
            if fixed_bank is None
            else fixed_bank.select(
                jax.random.randint(scene_key, (count,), 0, fixed_bank.num_instances)
            )
        )
        if command_distribution["kind"] == "position":
            from drone_playground.environments.randomization import sample_command

            bank = bank.replace(
                goal=jax.vmap(lambda goal, rng: sample_command(goal, rng, command_distribution))(
                    bank.goal, jax.random.split(speed_key, count)
                )
            )
        speeds = jax.random.uniform(speed_key, (count,), minval=lower, maxval=upper)

        def objective(params):
            return rollout_objective(task, network, params, bank, speeds, horizon, reset_key)

        (_, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(state.params)
        delta, opt_state = optimizer.update(gradient, state.opt_state, state.params)
        params = optax.apply_updates(state.params, delta)
        return TrainingState(params, opt_state, key, state.updates + 1), {
            **metrics,
            "gradient_norm": optax.tree.norm(gradient),
            "parameter_norm": optax.tree.norm(params),
        }

    return update


def make_checkpoint_eval_evaluator(task, network, config):
    """Build frozen-seed evaluation for recurrent learner checkpoints."""
    from drone_playground.environments.factory import build_environment

    count = config["training"]["checkpoint_eval_envs"]
    evaluation_env = build_environment(config, config["runtime"]["device"], "eval", count)
    evaluation_task = evaluation_env.task
    seed = config["training"]["checkpoint_eval_seed_start"]
    bank = evaluation_task.scene.sample(jax.random.PRNGKey(seed), count)
    speeds = jnp.linspace(*config["env"]["task"]["command_distribution"]["speed_range_mps"], count)
    horizon = config["algorithm"]["horizon_length"]
    evaluation_env.close()
    return (
        jax.jit(
            lambda params: rollout_objective(
                evaluation_task,
                network,
                params,
                bank,
                speeds,
                horizon,
                jax.random.PRNGKey(seed),
            )[1]
        ),
        bank,
    )


def train(config, root: Path, run_id: str):
    """Train and save a recurrent policy using backpropagation through time."""
    from drone_playground.environments.factory import build_environment

    env = build_environment(
        config, config["runtime"]["device"], role="train", count=config["training"]["num_envs"]
    )
    try:
        task = env.task
        state, network, optimizer = initialize(task, config)
        state, restored = restore_recurrent_state(state, config)
        selected = restored.get("selection")
        selected_report = restored.get("selection_report")
        updates = config["training"]["policy_updates"]
        stop = config["training"].get("stop_after_updates")
        stop = updates if stop is None else min(updates, int(stop))
        if int(state.updates) >= stop:
            raise ValueError("Requested training stage is already complete")
        update = make_update(task, network, optimizer, config)
        checkpoint_eval, checkpoint_eval_bank = make_checkpoint_eval_evaluator(
            task, network, config
        )
        count = config["training"]["num_envs"]
        per_update = count * config["algorithm"]["horizon_length"]
        milestone = max(1, updates // max(1, config["training"]["num_evals"] - 1))
        start_update = int(state.updates)
        start_params = state.params
        with RunRecorder(root, run_id, config, task_id="DP-005-pointcloud-paper") as rec:
            rec.record_environment(env)
            with capture_console(rec.path / "console.log"):
                save_report(rec.path / "components.json", env.component_identity)
                save_report(rec.path / "sensor-calibration.json", task.sensor_calibration)
                save_report(
                    rec.path / "checkpoint-eval-scene.json",
                    dict(
                        seed=config["training"]["checkpoint_eval_seed_start"],
                        bank_digest=checkpoint_eval_bank.digest(),
                        num_instances=checkpoint_eval_bank.num_instances,
                        navigation_benchmark_visible_to_training=False,
                    ),
                )
                rec.phase("compiling", step=start_update * per_update)

                def snapshot(current):
                    nonlocal selected, selected_report
                    metrics = {k: float(v) for k, v in checkpoint_eval(current.params).items()}
                    if not all(np.isfinite(list(metrics.values()))):
                        raise FloatingPointError("Nonfinite checkpoint_eval evaluation")
                    path = rec.path / "training-state" / f"update-{int(current.updates):07d}.pkl"
                    selected, selected_report = save_recurrent_snapshot(
                        path,
                        current,
                        config,
                        metrics,
                        (-metrics["loss"],),
                        selected,
                        selected_report,
                        report_path=rec.path
                        / "eval"
                        / f"checkpoint_eval-{int(current.updates):07d}.json",
                        fields={"checkpoint_eval_loss": metrics["loss"]},
                    )
                    rec.log(
                        int(current.updates) * per_update,
                        {f"checkpoint_eval/{k}": v for k, v in metrics.items()},
                    )
                    return path

                state, _, execution = run_recurrent_updates(
                    state,
                    update,
                    snapshot,
                    config,
                    rec,
                    steps_per_update=per_update,
                    milestones=set(range(milestone, stop + 1, milestone)),
                    metric_prefix="train/",
                )
                durations = execution["durations"]
                last_path = execution["last_snapshot"]
                actual = int(state.updates)
                parameter_delta = float(
                    jnp.sqrt(
                        sum(
                            jnp.sum((a - b) ** 2)
                            for a, b in zip(
                                jax.tree.leaves(start_params),
                                jax.tree.leaves(state.params),
                                strict=True,
                            )
                        )
                    )
                )
                result = dict(
                    actual_updates=actual,
                    target_updates=updates,
                    actual_steps=actual * per_update,
                    session_steps=(actual - start_update) * per_update,
                    target_steps=updates * per_update,
                    full_budget_completed=actual == updates,
                    checkpoint=str(last_path.resolve()),
                    selected=selected,
                    parameter_delta_l2=parameter_delta,
                    first_update_seconds=durations[0],
                    steady_update_seconds=float(np.mean(durations[1:] or durations)),
                    training_scene="independent static primitives",
                    navigation_benchmark_evaluated=False,
                )
                rec.finish("completed" if actual == updates else "paused", **result)
                return result
    finally:
        env.close()
