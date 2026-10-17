"""Bounded training, logging, and resumable checkpoints."""

import json
import signal
import time

import jax
import numpy as np
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from drone_playground.learning.checkpoint import load_model_state, save_model
from drone_playground.learning.selection import EarlyStopping, RewardSelection
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.factory import create_environment, create_method
from drone_playground.simulation.policy import checkpoint_summary, load_policy
from drone_playground.simulation.records import atomic_json, gpu_usage
from drone_playground.simulation.training_visuals import TrainingPreview


def training_control(path):
    """Read atomic per-run controls at a completed-update boundary."""
    try:
        control = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(control, dict) or control.keys() - {"stop", "early_stopping"}:
        raise ValueError("Training control accepts only stop and early_stopping")
    if "stop" in control and type(control["stop"]) is not bool:
        raise ValueError("Training control stop must be boolean")
    return control


def train(config, record):
    """Run a fixed number of training iterations with optional episode selection."""
    if not get_class(config.method.implementation._target_).trainable:
        raise ValueError("Training requires a trainable policy")
    if config.task.name == "navigation" and config.scene.name != "train/procedural":
        raise ValueError(
            "OA training requires randomized obstacles; fixed maps belong to Benchmark"
        )
    budget = config.learning.max_iterations
    if type(budget) is not int or budget < 1:
        raise ValueError("learning.max_iterations must be a positive integer")
    for field in ("checkpoint_interval", "log_interval"):
        if type(config.learning[field]) is not int or config.learning[field] < 1:
            raise ValueError(f"learning.{field} must be a positive integer")
    resolved = OmegaConf.to_container(config, resolve=True)
    method = create_method(config)
    trainer = Trainer(
        create_environment(config),
        kind=config.method.actor.get("kind", "custom"),
        algorithm=config.learning.algorithm,
        seed=config.seed,
        config=resolved["learning"]["options"],
        actor_config=resolved["method"]["actor"],
        critic=instantiate(config.learning.critic) if config.learning.get("critic") else None,
        method=method,
        loss=config.learning.loss,
        backward_model=config.learning.backward_model,
        backward_options=resolved["learning"]["backward_options"],
    )
    state = trainer.initialize()
    identity = {"experiment": resolved, "learning": trainer.resolved_config}
    previous_seconds = 0.0
    saved_selection = None
    saved_stopping = None
    if config.resume:
        state, metadata = load_model_state(config.resume, state)
        previous = metadata["config"]["experiment"]
        for field in (
            "task",
            "scene",
            "sensor",
            "simulation",
            "method",
            "controller",
            "observation",
            "learning",
            "seed",
        ):
            if previous[field] != resolved[field]:
                raise ValueError(f"Resume changes the saved {field} contract; use a new experiment")
        if "initialization" in metadata["config"]:
            identity["initialization"] = metadata["config"]["initialization"]
        previous_seconds = metadata["provenance"]["training_wall_seconds"]
        saved_selection = metadata["provenance"].get("selection")
        saved_stopping = metadata["provenance"].get("early_stopping")
        record.event("resumed", checkpoint=str(config.resume), update=int(state.updates))
    elif config.initial_checkpoint:
        _, parameters, metadata = load_policy(config.initial_checkpoint, actor=trainer.actor)
        if jax.tree.structure(parameters) != jax.tree.structure(state.params):
            raise ValueError("Initialization checkpoint has a different Actor structure")
        for initial, template in zip(
            jax.tree.leaves(parameters), jax.tree.leaves(state.params), strict=True
        ):
            if initial.shape != template.shape or initial.dtype != template.dtype:
                raise ValueError(
                    "Initialization checkpoint has incompatible parameter shapes or dtypes"
                )
            if not np.isfinite(np.asarray(initial)).all():
                raise ValueError("Initialization checkpoint contains nonfinite parameters")
        state = state.replace(params=parameters)
        identity["initialization"] = {
            "checkpoint": str(config.initial_checkpoint),
            "provenance": metadata["provenance"],
        }
        record.event("initialized", **identity["initialization"])
    selection = RewardSelection(config, trainer, saved_selection)
    stopping = EarlyStopping(
        config.learning.early_stopping, config.learning.best.source, saved_stopping
    )
    record.identity.update(
        num_envs=trainer.env.num_envs,
        rollout_horizon=trainer.config["horizon"],
        algorithm=trainer.algorithm,
        task=trainer.env.task.name,
        best_checkpoint=selection.state["best_checkpoint"],
        best_mean_reward=selection.state["best_mean_reward"],
        best_source=config.learning.best.source,
    )
    started = time.monotonic()
    preview = TrainingPreview()
    initial_interactions = int(state.interactions)
    stop_requested = False
    stop_reason = None
    checkpoint_path = None

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    def cost():
        elapsed = time.monotonic() - started
        interactions = int(state.interactions) - initial_interactions
        return {
            "session_interactions": interactions,
            "session_wall_seconds": elapsed,
            "training_wall_seconds": previous_seconds + elapsed,
            "interactions_per_second": interactions / elapsed if elapsed > 0 else 0.0,
        }

    def checkpoint():
        nonlocal checkpoint_path
        checkpoint_path = record.directory / "checkpoints" / f"model_{int(state.updates):06d}"
        save_model(
            checkpoint_path,
            state,
            config=identity,
            provenance={
                **record.identity,
                "updates": int(state.updates),
                "interactions": int(state.interactions),
                "selection": selection.state,
                "early_stopping": stopping.state,
                "stop_reason": stop_reason,
                **cost(),
            },
            kind=trainer.kind,
            actor_spec=trainer.actor.specification,
        )
        record.identity.update(checkpoint_summary(checkpoint_path))
        atomic_json(record.directory / "run.json", record.identity)

    def select(*, final=False):
        measured = selection.measure(state, final=final)
        if measured is None:
            return
        stopping.observe(measured)
        if measured["improved"]:
            path = record.directory / "checkpoints" / f"model_{int(state.updates):06d}"
            selection.state.update(
                best_mean_reward=measured["mean_reward"],
                best_checkpoint=str(path.resolve()),
                best_score=measured.get("selection_score"),
            )
            checkpoint()
        record.event(
            "selection", update=int(state.updates), interactions=int(state.interactions), **measured
        )
        print(
            json.dumps(
                {
                    "event": "selection",
                    "update": int(state.updates),
                    "interactions": int(state.interactions),
                    **measured,
                }
            ),
            flush=True,
        )
        return stopping.reason()

    metrics = {}
    try:
        while not stop_requested and int(state.updates) < budget:
            control = training_control(record.directory / "training-control.json")
            if control.get("stop", False):
                stop_reason = {"reason": "requested_stop"}
                break
            if stopping.configure(control.get("early_stopping", {})):
                rules = stopping.state["options"]
                record.event("early_stopping_configured", **rules)
                if rules["target"] is not None or rules["patience"]:
                    stop_reason = select(final=True) or stopping.reason()
                    if stop_reason:
                        break
            stop_reason = stopping.reason()
            if stop_reason:
                if selection.state["last_interactions"] != int(state.interactions):
                    stop_reason = select(final=True)
                if stop_reason:
                    break
            update_started = time.monotonic()
            candidate, metrics = trainer.update(
                state,
                **(
                    {"sampling_progress": int(state.updates) / budget}
                    if trainer.distance_curriculum
                    else {}
                ),
            )
            metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
            if not all(np.isfinite(list(metrics.values()))):
                raise FloatingPointError(f"Nonfinite update at {int(state.updates) + 1}: {metrics}")
            state = candidate
            selection.observe(metrics)
            update = int(state.updates)
            if preview.session is not None:
                preview.capture(
                    state.env_state,
                    trainer.env.scene,
                    getattr(trainer, "last_preview", {}),
                    update=update,
                )
            if (
                update == 1
                or update % config.learning.log_interval == 0
                or int(state.updates) >= budget
                or stop_requested
            ):
                measured = {
                    "update": update,
                    "interactions": int(state.interactions),
                    **metrics,
                    "completed_episodes": selection.state["episodes_completed"],
                    "episode_return_total": selection.state["episode_return_sum"],
                    "episode_length_total": selection.state["episode_length_sum"],
                    "episode_success_total": selection.state["episode_successes"],
                    "avg_episode_return": (
                        selection.state["episode_return_sum"]
                        / selection.state["episodes_completed"]
                        if selection.state["episodes_completed"]
                        else None
                    ),
                    "update_seconds": time.monotonic() - update_started,
                    **cost(),
                    **gpu_usage(),
                }
                record.event("update", **measured)
                print(json.dumps(measured), flush=True)
            stop_reason = select(final=int(state.updates) >= budget)
            if update % config.learning.checkpoint_interval == 0:
                checkpoint()
            if stop_reason:
                break
        if stop_reason:
            record.event(
                "early_stopped",
                update=int(state.updates),
                interactions=int(state.interactions),
                **stop_reason,
            )
        checkpoint()
        if preview.session is not None:
            preview.capture(
                state.env_state,
                trainer.env.scene,
                getattr(trainer, "last_preview", {}),
                update=int(state.updates),
                final=True,
            )
        record.finish(
            "budget_exhausted"
            if int(state.updates) >= budget
            else "early_stopped"
            if stop_reason
            else "interrupted",
            updates=int(state.updates),
            interactions=int(state.interactions),
            requested_iterations=budget,
            latest_checkpoint=str(checkpoint_path.relative_to(record.directory)),
            last_metrics=metrics,
            selection=selection.state,
            early_stopping=stopping.state,
            stop_reason=stop_reason,
            **cost(),
        )
    except BaseException as error:
        checkpoint()
        record.finish(
            "failed",
            updates=int(state.updates),
            interactions=int(state.interactions),
            error=f"{type(error).__name__}: {error}",
            resume_checkpoint=str(checkpoint_path),
            **cost(),
        )
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        method.close()
