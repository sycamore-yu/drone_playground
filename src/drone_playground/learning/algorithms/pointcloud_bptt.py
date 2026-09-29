"""Recurrent, rematerialized direct policy training for the point-cloud paper.

This trainer uses the existing JAX/Flax/Optax/Brax stack and run recorder. Its
complete recipe and reconstruction assumptions live in the experiment config.
"""

import copy
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from hydra.utils import instantiate

from drone_playground.environments.scenes.navigation import euclidean_norm
from drone_playground.evaluation.tracking import save_report, tree_digest
from drone_playground.runs.console import capture_console
from drone_playground.runs.pointcloud import load_training_state, save_training_state
from drone_playground.runs.record import RunRecorder
from drone_playground.runtime.jax_runner import recurrent_scan


@struct.dataclass
class TrainingState:
    params: object
    opt_state: object
    key: jax.Array
    updates: jax.Array


def initialize(task, config):
    network = instantiate(config["network"], _convert_="all")
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
    return TrainingState(params, optimizer.init(params), key, jnp.int32(0)), network, optimizer


def rollout_objective(task, network, params, bank, speeds, horizon):
    state = task.initial_state(bank)
    memory = jnp.zeros((bank.num_instances, network.hidden_size))

    def step(carry, index):
        physical, hidden = carry
        timestamp = index * task.dt
        points, valid, proprio, target = task.observation(bank, physical, timestamp, speeds)
        body_action, hidden = network.apply(params, points, valid, proprio, hidden)
        command = task.command(body_action, physical)
        nxt = task.model.step(physical, command, task.dt)
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
    loss, parts = task.objective(trace, task.dt)
    metrics = {
        **parts,
        "loss": loss,
        "mean_speed": jnp.mean(euclidean_norm(trace["velocity"])),
        "min_clearance": jnp.min(trace["clearance"]),
        "step_collision_fraction": jnp.mean((trace["clearance"] < 0).astype(jnp.float32)),
    }
    return loss, metrics


def make_update(task, network, optimizer, config):
    count = config["training"]["num_envs"]
    horizon = config["algorithm"]["horizon_length"]
    lower, upper = config["env"]["task"]["command_speed_range"]

    @jax.jit
    def update(state):
        key, scene_key, speed_key = jax.random.split(state.key, 3)
        bank = task.scene.sample(scene_key, count)
        speeds = jax.random.uniform(speed_key, (count,), minval=lower, maxval=upper)

        def objective(params):
            return rollout_objective(task, network, params, bank, speeds, horizon)

        (_, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(state.params)
        delta, opt_state = optimizer.update(gradient, state.opt_state, state.params)
        params = optax.apply_updates(state.params, delta)
        return TrainingState(params, opt_state, key, state.updates + 1), {
            **metrics,
            "gradient_norm": optax.tree.norm(gradient),
            "parameter_norm": optax.tree.norm(params),
        }

    return update


def make_development_evaluator(task, network, config):
    count = config["training"]["development_envs"]
    bank = task.scene.sample(jax.random.PRNGKey(20000), count)
    speeds = jnp.linspace(*config["env"]["task"]["command_speed_range"], count)
    horizon = config["algorithm"]["horizon_length"]
    return jax.jit(
        lambda params: rollout_objective(task, network, params, bank, speeds, horizon)[1]
    ), bank


def continuation_contract(config):
    value = copy.deepcopy(config)
    for key in (
        "run_id",
        "checkpoint",
        "evaluation",
        "replay",
        "hydra",
        "migration",
        "visualization",
    ):
        value.pop(key, None)
    for key in ("resume", "max_wall_seconds", "stop_after_updates", "device"):
        value["training"].pop(key, None)
    value["runtime"].pop("device", None)
    return value


def train(config, root: Path, run_id: str):
    from drone_playground.composition import build_environment

    task = build_environment(config, config["runtime"]["device"])
    state, network, optimizer = initialize(task, config)
    selected = None
    resume = config["training"].get("resume")
    if resume:
        state, metadata = load_training_state(resume)
        if continuation_contract(metadata["config"]) != continuation_contract(config):
            raise ValueError("Resume changes the paper's training contract")
        selected = metadata.get("selection")
    updates = config["training"]["policy_updates"]
    stop = config["training"].get("stop_after_updates")
    stop = updates if stop is None else min(updates, int(stop))
    if int(state.updates) >= stop:
        raise ValueError("Requested training stage is already complete")
    update = make_update(task, network, optimizer, config)
    development, dev_bank = make_development_evaluator(task, network, config)
    count = config["training"]["num_envs"]
    per_update = count * config["algorithm"]["horizon_length"]
    milestone = max(1, updates // max(1, config["training"]["num_evals"] - 1))
    start_update = int(state.updates)
    start_params = state.params
    with RunRecorder(root, run_id, config, task_id="DP-005-pointcloud-paper") as rec:
        with capture_console(rec.path / "console.log"):
            save_report(rec.path / "components.json", task.component_identity)
            save_report(rec.path / "sensor-calibration.json", task.sensor_calibration)
            save_report(
                rec.path / "development-scene.json",
                dict(
                    seed=20000,
                    bank_digest=dev_bank.digest(),
                    num_instances=dev_bank.num_instances,
                    navigation8_visible_to_training=False,
                ),
            )
            rec.phase("compiling", step=start_update * per_update)
            start_clock = time.monotonic()
            durations = []

            def snapshot(current):
                nonlocal selected
                metrics = {k: float(v) for k, v in development(current.params).items()}
                if not all(np.isfinite(list(metrics.values()))):
                    raise FloatingPointError("Nonfinite development evaluation")
                path = rec.path / "training-state" / f"update-{int(current.updates):07d}.pkl"
                candidate = dict(
                    checkpoint=str(path.resolve()),
                    updates=int(current.updates),
                    development_loss=metrics["loss"],
                    parameter_sha256=tree_digest(current.params),
                )
                if selected is None or candidate["development_loss"] < selected["development_loss"]:
                    selected = candidate
                save_training_state(path, current, config, selected)
                save_report(
                    rec.path / "eval" / f"development-{int(current.updates):07d}.json", metrics
                )
                save_report(rec.path / "best.json", selected)
                rec.log(
                    int(current.updates) * per_update, {f"dev/{k}": v for k, v in metrics.items()}
                )
                return path

            last_path = snapshot(state)
            for index in range(start_update + 1, stop + 1):
                tick = time.monotonic()
                state, values = update(state)
                metrics = {k: float(v) for k, v in values.items()}
                elapsed = time.monotonic() - tick
                durations.append(elapsed)
                if not all(np.isfinite(list(metrics.values()))):
                    save_training_state(
                        rec.path / "training-state/nonfinite.pkl", state, config, selected
                    )
                    raise FloatingPointError(f"Nonfinite update {index}: {metrics}")
                if index == start_update + 1 or index % 10 == 0:
                    average = float(np.mean(durations[-20:]))
                    rec.phase(
                        "training",
                        index * per_update,
                        updates=index,
                        target_updates=updates,
                        seconds_per_update=average,
                        estimated_remaining_s=(updates - index) * average,
                    )
                    rec.log(
                        index * per_update,
                        {
                            **{f"train/{k}": v for k, v in metrics.items()},
                            "train/update_seconds": elapsed,
                            "train/updates": index,
                        },
                    )
                    print(
                        json.dumps(
                            dict(
                                update=index,
                                target=updates,
                                loss=metrics["loss"],
                                speed=metrics["mean_speed"],
                                seconds=elapsed,
                            )
                        ),
                        flush=True,
                    )
                wall_exhausted = (
                    time.monotonic() - start_clock >= config["training"]["max_wall_seconds"]
                )
                if index % milestone == 0 or index == stop or wall_exhausted:
                    last_path = snapshot(state)
                if wall_exhausted:
                    break
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
                navigation8_evaluated=False,
            )
            rec.finish("completed" if actual == updates else "paused", **result)
            return result
