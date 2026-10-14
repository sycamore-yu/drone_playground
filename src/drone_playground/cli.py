"""Hydra entry points for simulation, training, checkpoint evaluation and benchmark."""

import json
import os
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import hydra
import jax
import numpy as np
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from drone_playground.simulation.evaluation import (
    create_environment,
    create_method,
    evaluate,
    next_evaluation_directory,
)
from drone_playground.simulation.methods import Action
from drone_playground.simulation.policy import load_policy, read_checkpoint
from drone_playground.simulation.records import RunRecord, atomic_json, gpu_usage
from drone_playground.simulation.scene import Scene

_NAVIGATION_SCENES = ["S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06"]


def screening_passed(report: dict, task: str, primary_scenes: list[str]) -> bool:
    """Admit a promising candidate without conflating screening with formal C2/C3/C4.

    Light screens are intentionally permissive; a complete independent
    checkpoint validation still applies the original hard task thresholds.
    """
    if not report.get("complete"):
        return False
    results = report.get("reports", {})
    if not primary_scenes or not all(scene in results for scene in primary_scenes):
        return False
    for scene in primary_scenes:
        result = results[scene]
        if result.get("episodes") != 8:
            return False
        target = 6 if task == "navigation" else 7
        if result.get("successes", 0) < target:
            return False
        if task == "tracking":
            rmse = result.get("successful_position_rmse")
            if rmse is None or not np.isfinite(rmse) or rmse > 0.35:
                return False
    return True


def probe_justifies_full_screen(report: dict, *, min_successes: int = 1) -> bool:
    """Escalate a probe on genuine arrival evidence; never declare a pass here."""
    if type(min_successes) is not int or min_successes < 1:
        raise ValueError("Probe minimum successes must be a positive integer")
    rows = report.get("reports", {})
    if not rows or not report.get("complete"):
        return False
    return any(
        isinstance(row, dict)
        and row.get("episodes") == 8
        and type(row.get("successes")) is int
        and row["successes"] >= min_successes
        for row in rows.values()
    )


def diagnostic_probe_scenes(primary_scenes: list[str], evaluation_index: int, full_every: int):
    """Cycle through static/dynamic scene pairs between mandatory full screens."""
    if len(primary_scenes) != 6 or full_every < 2 or evaluation_index < 0:
        raise ValueError("Fast Navigation probes require six scenes and valid evaluation settings")
    probe_round = evaluation_index - evaluation_index // full_every
    index = probe_round % 3
    return [primary_scenes[index], primary_scenes[index + 3]]


def training_budget_reached(interactions: int, limit: int | None) -> bool:
    """Bound a pilot by collected transitions, independent of its PPO horizon."""
    if limit is None:
        return False
    if type(limit) is not int or limit <= 0:
        raise ValueError("learning.max_interactions must be a positive integer or null")
    return interactions >= limit


def train(config: DictConfig, record: RunRecord) -> None:
    """Train under a versioned evaluation contract with one common update lifecycle."""
    from drone_playground.learning.checkpoint import (
        load_model_state,
        load_state,
        save_inference,
        save_model,
        save_state,
    )
    from drone_playground.learning.progress_guard import (
        ProgressGuardConfig,
        evaluate_progress_guard,
    )
    from drone_playground.learning.selection import (
        PatienceConfig,
        screening_score,
        update_patience,
    )
    from drone_playground.learning.trainer import Trainer

    started = time.monotonic()
    if not get_class(config.method.implementation._target_).trainable:
        raise ValueError("Training requires method=policy or an experiment recipe")
    selection = config.learning.selection
    legacy = selection == "last_of_first_three_consecutive_passes"
    patience_settings = (
        None
        if legacy
        else PatienceConfig(**OmegaConf.to_container(config.early_stopping, resolve=True))
    )
    if selection not in ("last_of_first_three_consecutive_passes", "first_full_validation_pass"):
        raise ValueError("Unsupported checkpoint selection rule")
    if config.learning.required_consecutive_passes != (3 if legacy else 1):
        raise ValueError("Learning selection and required_consecutive_passes disagree")
    resolved = OmegaConf.to_container(config, resolve=True)
    max_interactions = resolved["learning"].get("max_interactions")
    training_budget_reached(0, max_interactions)
    scenes = resolved["learning"].get("scenes")
    scene_updates = resolved["learning"].get("scene_updates", 20)
    scene_index, updates_in_scene = 0, 0
    sampling = None
    screening_options = resolved.get("screening", {})
    training_probe_enabled = screening_options.get("training_distribution_probe", False)
    if type(training_probe_enabled) is not bool or (
        training_probe_enabled and (config.task.name != "navigation" or legacy)
    ):
        raise ValueError("Training-distribution probe requires nonlegacy Navigation evaluation")
    fast_probes = screening_options.get("probe_between_full", False)
    full_every = screening_options.get("full_every", 3)
    progress_guard_options = screening_options.get("progress_aware")
    progress_guard_policy = None
    if progress_guard_options is not None and progress_guard_options is not False:
        if (
            type(progress_guard_options) is not dict
            or not progress_guard_options
            or config.task.name != "navigation"
            or legacy
            or not config.benchmark.get("fixed_checkpoint_seeds", False)
            or not config.results.get("keep_best_screening", False)
            or max_interactions is None
        ):
            raise ValueError(
                "Progress-aware screening requires an explicit Navigation "
                "configuration, finite training budget, fixed screening seeds "
                "and best checkpoint retention"
            )
        progress_guard_policy = ProgressGuardConfig(
            **progress_guard_options,
            min_score_delta=patience_settings.min_delta,
        )
    if type(fast_probes) is not bool or type(full_every) is not int or full_every < 2:
        raise ValueError("screening requires boolean probe_between_full and full_every >= 2")
    if fast_probes and (
        config.task.name != "navigation"
        or not config.benchmark.get("fixed_checkpoint_seeds", False)
        or legacy
    ):
        raise ValueError("Fast probes require fixed-seed Navigation checkpoint selection")
    if scenes is not None:
        if (
            config.task.name != "navigation"
            or not isinstance(scenes, list)
            or not scenes
            or any(scene not in _NAVIGATION_SCENES for scene in scenes)
            or len(set(scenes)) != len(scenes)
        ):
            raise ValueError("learning.scenes requires a nonempty unique Navigation8 scene list")
        if (
            not isinstance(scene_updates, int)
            or isinstance(scene_updates, bool)
            or scene_updates < 1
        ):
            raise ValueError("learning.scene_updates must be a positive integer")
        sampling = {
            "scenes": scenes,
            "scene_updates": scene_updates,
            "geometry_sha256": {scene: Scene(scene).geometry_hash for scene in scenes},
        }
        if config.resume:
            _, metadata = read_checkpoint(config.resume, "training")
            if metadata["config"].get("training_sampling") != sampling:
                raise ValueError("Resume changes the saved training sampling schedule or geometry")
            progress = metadata["provenance"].get("training_sampling", {})
            scene_index = progress.get("scene_index")
            updates_in_scene = progress.get("updates_in_scene")
            if (
                type(scene_index) is not int
                or not 0 <= scene_index < len(scenes)
                or type(updates_in_scene) is not int
                or not 0 <= updates_in_scene <= scene_updates
            ):
                raise ValueError("Checkpoint has invalid training sampling progress")
    kind = config.method.actor.get("kind", "custom")
    orbax_layout = getattr(record, "layout_version", 2) == 3
    trainers = {}

    def trainer_for(index):
        if index not in trainers:
            env = (
                create_environment(config)
                if scenes is None
                else create_environment(config, scene=scenes[index])
            )
            trainers[index] = Trainer(
                env,
                kind=kind,
                algorithm=config.learning.algorithm,
                seed=config.seed,
                config=resolved["learning"]["options"],
                actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
                critic=(
                    instantiate(config.learning.critic)
                    if config.learning.get("critic") is not None
                    else None
                ),
                method=create_method(config),
                loss=config.learning.loss,
                backward_model=config.learning.backward_model,
                backward_options=OmegaConf.to_container(
                    config.learning.backward_options, resolve=True
                ),
            )
        return trainers[index]

    trainer = trainer_for(scene_index)
    state = trainer.initialize()
    identity = {"experiment": resolved, "learning": trainer.resolved_config}
    if sampling is not None:
        identity["training_sampling"] = sampling
    if config.initial_checkpoint and not config.resume:
        initial_actor, parameters, metadata = load_policy(
            config.initial_checkpoint, actor=trainer.actor
        )
        if initial_actor.specification != trainer.actor.specification or jax.tree.structure(
            parameters
        ) != jax.tree.structure(state.params):
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
            "sha256": metadata["sha256"],
            "provenance": metadata["provenance"],
        }
        record.event("initialized", **identity["initialization"])
    consecutive = 0
    evaluation_index = 0
    previous_wall_seconds = 0.0
    pending_evaluation = False
    best_screen_score = None
    stagnant_screens = 0
    patience_exhausted = False
    progress_guard_state = None
    if config.resume:
        if Path(config.resume).is_dir():
            state, metadata = load_model_state(config.resume, state)
        else:
            state, metadata = load_state(config.resume, state)
        if "initialization" in metadata["config"]:
            identity["initialization"] = metadata["config"]["initialization"]
        previous = metadata["config"]["experiment"]
        if previous.get("screening", {}).get("progress_aware") != screening_options.get(
            "progress_aware"
        ):
            raise ValueError("Resume changes the progress-aware screening contract")
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
        if scenes is not None and (
            int(state.updates) < updates_in_scene
            or (int(state.updates) - updates_in_scene) % (scene_updates * len(scenes))
            != scene_index * scene_updates
        ):
            raise ValueError("Checkpoint training sampling progress disagrees with update count")
        progress = metadata["provenance"]
        consecutive = progress["consecutive_passes"]
        evaluation_index = progress["evaluation_index"]
        previous_wall_seconds = progress["training_wall_seconds"]
        if patience_settings is not None:
            screening_progress = progress.get("screening_progress", {})
            best_screen_score = screening_progress.get("best_score")
            stagnant_screens = int(screening_progress.get("stagnant_screens", 0))
            if progress_guard_policy is not None:
                if "progress_guard" not in screening_progress:
                    raise ValueError("Resume checkpoint predates progress guard evidence")
                progress_guard_state = screening_progress["progress_guard"]
        interval = config.learning.evaluation_interval
        at_budget = training_budget_reached(int(state.interactions), max_interactions)
        expected_evaluations = int(state.updates) // interval + int(
            at_budget and int(state.updates) % interval != 0
        )
        pending_evaluation = (
            int(state.updates) % interval == 0 or at_budget
        ) and evaluation_index < expected_evaluations
        record.event("resumed", checkpoint=str(config.resume), update=int(state.updates))
    if sampling is not None:
        record.identity["training_sampling"] = sampling
        atomic_json(record.directory / "run.json", record.identity)
        record.event("training_sampling", **sampling)
    evaluation_config = OmegaConf.create(resolved)
    evaluation_config.mode = "checkpoint_eval"
    evaluation_config.benchmark.record_replay = False
    evaluation_config.benchmark.episodes = None
    if scenes is not None:
        evaluation_config.benchmark.scenes = None
    if not legacy:
        primary_groups = OmegaConf.to_container(config.benchmark.primary_scenes, resolve=True)
        primary_scenes = (
            [*primary_groups["static"], *primary_groups["dynamic"]]
            if config.task.name == "navigation"
            else [config.scene.name]
        )
        light_config = OmegaConf.create(resolved)
        light_config.mode = "checkpoint_eval"
        light_config.benchmark.protocol = "diagnostic"
        light_config.benchmark.record_replay = False
        light_config.benchmark.record_trajectories = False
        light_config.benchmark.episodes = 8
        light_config.benchmark.scenes = primary_scenes
        light_config.benchmark.seed_base = 1_000_000
        if training_probe_enabled:
            training_probe_config = OmegaConf.create(
                OmegaConf.to_container(light_config, resolve=True)
            )
            training_probe_config.benchmark.scenes = ["train/procedural"]
            training_probe_config.benchmark.seed_base = 1_100_000
        evaluation_config.benchmark.scenes = primary_scenes
        evaluation_config.benchmark.seed_base = 1_500_000
        if fast_probes:
            probe_config = OmegaConf.create(resolved)
            probe_config.mode = "checkpoint_eval"
            probe_config.benchmark.protocol = "diagnostic"
            probe_config.benchmark.seed_base = 1_000_000
            probe_config.benchmark.episodes = 8
            probe_config.benchmark.record_replay = False
            probe_config.benchmark.record_trajectories = False
            if progress_guard_policy is not None:
                # Only full six-scene screens affect progress-aware patience.
                probe_config.screening.progress_aware = None
    environments = {}
    initial_interactions = int(state.interactions)
    checkpoint_path = record.directory / "checkpoints" / "latest.training.zip"
    last_checkpoint_update = -1
    last_frozen = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    def cost(current):
        elapsed = time.monotonic() - started
        interactions = int(current.interactions) - initial_interactions
        return {
            "session_interactions": interactions,
            "session_wall_seconds": elapsed,
            "training_wall_seconds": (
                previous_wall_seconds + elapsed if previous_wall_seconds is not None else None
            ),
            "interactions_per_second": interactions / elapsed if elapsed > 0 else 0.0,
        }

    def checkpoint(current, *, export_policy=legacy):
        nonlocal last_checkpoint_update, last_frozen, checkpoint_path
        provenance = {
            **record.identity,
            "updates": int(current.updates),
            "interactions": int(current.interactions),
            "consecutive_passes": consecutive,
            "evaluation_index": evaluation_index,
            "screening_progress": {
                "best_score": best_screen_score,
                "stagnant_screens": stagnant_screens,
                **(
                    {"progress_guard": progress_guard_state}
                    if progress_guard_policy is not None
                    else {}
                ),
            },
            **cost(current),
        }
        if scenes is not None:
            provenance["training_sampling"] = {
                "scene_index": scene_index,
                "updates_in_scene": updates_in_scene,
            }
        if orbax_layout:
            frozen = record.directory / "checkpoints" / f"model_{int(current.updates):06d}"
            checkpoint_path = frozen
            save_model(
                frozen,
                current,
                config=identity,
                provenance=provenance,
                kind=kind,
                actor_spec=trainer.actor.specification,
                export_policy=export_policy,
            )
        else:
            save_state(checkpoint_path, current, config=identity, provenance=provenance)
            frozen = (
                record.directory / "checkpoints" / f"step-{int(current.updates):06d}" / "policy.zip"
            )
            if export_policy:
                save_inference(
                    frozen,
                    current.params,
                    kind=kind,
                    config=identity,
                    provenance=provenance,
                    actor_spec=trainer.actor.specification,
                )
        last_checkpoint_update = int(current.updates)
        if export_policy:
            last_frozen = frozen
        return frozen

    def evaluate_checkpoint(*, force_full=False):
        nonlocal consecutive, evaluation_index
        nonlocal best_screen_score, stagnant_screens, patience_exhausted
        nonlocal progress_guard_state
        update = int(state.updates)
        frozen = checkpoint(state, export_policy=False)
        directory = frozen if orbax_layout else frozen.parent
        if (
            (directory / "report.json").exists()
            or (directory / "episodes.csv").exists()
            or (not legacy and (directory / "screening").exists())
        ):
            previous = Path(tempfile.mkdtemp(prefix="incomplete-", dir=directory))
            for item in directory.iterdir():
                if item.name in {
                    "report.json",
                    "episodes.csv",
                    "trajectories.npz",
                    "decisions.json",
                    "replays",
                    "screening",
                    "probe",
                    "training_distribution",
                }:
                    item.rename(previous / item.name)
            record.event(
                "checkpoint_evaluation_retry",
                update=update,
                previous_output=str(previous),
            )
        if training_probe_enabled:
            training_report = evaluate(
                training_probe_config,
                directory / "training_distribution",
                actor=trainer.actor,
                parameters=state.params,
                environments=environments,
            )
            record.event(
                "training_distribution_probe",
                update=update,
                per_scene=training_report["reports"],
            )
        if legacy:
            report = evaluate(
                evaluation_config,
                directory,
                actor=trainer.actor,
                parameters=state.params,
                evaluation_index=evaluation_index + 1,
                environments=environments,
            )
        else:
            report = {"passed": False}
            run_full_screen = (
                force_full or not fast_probes or (evaluation_index + 1) % full_every == 0
            )
            if not run_full_screen:
                # The same six canonical fixed seeds are used during full screens.
                # Rotate three static/dynamic pairs rather than evaluating all six.
                probe_config.benchmark.scenes = diagnostic_probe_scenes(
                    primary_scenes, evaluation_index, full_every
                )
                probe_started = time.monotonic()
                probe = evaluate(
                    probe_config,
                    directory / "probe",
                    actor=trainer.actor,
                    parameters=state.params,
                    evaluation_index=evaluation_index + 1,
                    environments=environments,
                )
                run_full_screen = probe_justifies_full_screen(probe)
                record.event(
                    "checkpoint_probe",
                    update=update,
                    scenes=list(probe_config.benchmark.scenes),
                    successes=sum(row["successes"] for row in probe["reports"].values()),
                    promoted_to_full=run_full_screen,
                    evaluation_seconds=time.monotonic() - probe_started,
                )
            if run_full_screen:
                screening_started = time.monotonic()
                screening = evaluate(
                    light_config,
                    directory / "screening",
                    actor=trainer.actor,
                    parameters=state.params,
                    evaluation_index=evaluation_index + 1,
                    environments=environments,
                )
                score = screening_score(screening, config.task.name, primary_scenes)
                progress_decision = None
                if progress_guard_policy is not None:
                    progress_guard_state, progress_decision = evaluate_progress_guard(
                        screening,
                        primary_scenes,
                        update=update,
                        previous=progress_guard_state,
                        best_screen_score=score,
                        settings=progress_guard_policy,
                        min_updates=patience_settings.min_updates,
                        patience=patience_settings.patience,
                    )
                    best_screen_score = progress_guard_state["best_score"]
                    stagnant_screens = progress_guard_state["stagnant_screens"]
                    improved = (
                        progress_decision["score_improved"]
                        or progress_decision["progress_improved"]
                    )
                    patience_exhausted = progress_decision["patience_exhausted"]
                    if (
                        progress_decision["regression_streak"]
                        == progress_guard_policy.regression_patience
                    ):
                        record.event(
                            "policy_regression",
                            update=update,
                            best_checkpoint_update=progress_guard_state["best_checkpoint_update"],
                            best_successes=progress_guard_state["best_successes"],
                            current_successes=progress_decision["successes"],
                            lower_height_failures=progress_decision["lower_height_out_of_bounds"],
                        )
                else:
                    (
                        best_screen_score,
                        stagnant_screens,
                        improved,
                        patience_exhausted,
                    ) = update_patience(
                        score,
                        update,
                        best_screen_score,
                        stagnant_screens,
                        settings=patience_settings,
                    )
                suitable = screening_passed(screening, config.task.name, primary_scenes)
                record.event(
                    "screening_evaluation",
                    update=update,
                    passed=suitable,
                    episodes=8,
                    scenes=primary_scenes,
                    per_scene=screening["reports"],
                    score=score,
                    best_score=best_screen_score,
                    improved=improved,
                    stagnant_screens=stagnant_screens,
                    patience_exhausted=patience_exhausted,
                    **(
                        {"progress_decision": progress_decision}
                        if progress_decision is not None
                        else {}
                    ),
                    evaluation_seconds=time.monotonic() - screening_started,
                )
                if suitable:
                    report = evaluate(
                        evaluation_config,
                        directory,
                        actor=trainer.actor,
                        parameters=state.params,
                        evaluation_index=evaluation_index + 1,
                        environments=environments,
                    )
                    record.event(
                        "full_validation",
                        update=update,
                        passed=bool(report["passed"]),
                        scenes=primary_scenes,
                        episodes=25 if config.task.name == "navigation" else 100,
                    )
        evaluation_index += 1
        consecutive = (
            (consecutive + 1 if report["passed"] else 0) if legacy else int(bool(report["passed"]))
        )
        checkpoint(state, export_policy=legacy or bool(report["passed"]))
        record.event(
            "checkpoint_evaluation",
            update=update,
            passed=report["passed"],
            consecutive_passes=consecutive,
            selection_rule=selection,
            checkpoint=str(frozen),
        )
        if patience_exhausted and not report["passed"]:
            record.event(
                "early_stopping",
                update=update,
                reason="light_screens_without_significant_improvement",
                best_score=best_screen_score,
                stagnant_screens=stagnant_screens,
                min_updates=patience_settings.min_updates,
                min_delta=patience_settings.min_delta,
            )

    try:
        if pending_evaluation:
            evaluate_checkpoint(
                force_full=training_budget_reached(int(state.interactions), max_interactions)
            )
        while (
            not stop_requested
            and not patience_exhausted
            and consecutive < config.learning.required_consecutive_passes
            and not training_budget_reached(int(state.interactions), max_interactions)
        ):
            if scenes is not None and updates_in_scene == scene_updates:
                next_index = (scene_index + 1) % len(scenes)
                next_trainer = trainer_for(next_index)
                truncated_worlds = int(jax.device_get((~state.env_state.done).sum()))
                next_state = next_trainer.reset_episodes(state)
                previous_scene = scenes[scene_index]
                trainer, state = next_trainer, next_state
                scene_index, updates_in_scene = next_index, 0
                record.event(
                    "training_scene_switch",
                    update=int(state.updates),
                    previous_scene=previous_scene,
                    scene=scenes[scene_index],
                    scene_index=scene_index,
                    truncated_worlds=truncated_worlds,
                    reset_worlds=trainer.env.num_envs,
                )
            update_started = time.monotonic()
            candidate, metrics = trainer.update(state)
            metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
            if not all(np.isfinite(list(metrics.values()))):
                checkpoint(state)
                raise FloatingPointError(f"Nonfinite update at {int(state.updates) + 1}: {metrics}")
            state = candidate
            if scenes is not None:
                updates_in_scene += 1
            update = int(state.updates)
            if update == 1 or update % config.learning.log_interval == 0:
                measured = {
                    "update": update,
                    "interactions": int(state.interactions),
                    **metrics,
                    "update_seconds": time.monotonic() - update_started,
                    **cost(state),
                    **gpu_usage(),
                }
                record.event("update", **measured)
                print(json.dumps(measured), flush=True)
            if (
                update % config.learning.checkpoint_interval == 0
                and update % config.learning.evaluation_interval != 0
            ):
                checkpoint(state)
            at_budget = training_budget_reached(int(state.interactions), max_interactions)
            if update % config.learning.evaluation_interval and not at_budget:
                continue
            evaluate_checkpoint(force_full=at_budget)
        frozen = (
            last_frozen
            if last_checkpoint_update == int(state.updates) and last_frozen is not None
            else checkpoint(state, export_policy=not orbax_layout)
        )
        if patience_exhausted and consecutive < config.learning.required_consecutive_passes:
            record.finish(
                "early_stopped",
                updates=int(state.updates),
                interactions=int(state.interactions),
                best_screen_score=best_screen_score,
                stagnant_screens=stagnant_screens,
                resume_checkpoint=str(checkpoint_path),
                reason="light_screens_without_significant_improvement",
                **cost(state),
            )
            return
        if consecutive < config.learning.required_consecutive_passes:
            record.finish(
                (
                    "budget_exhausted"
                    if training_budget_reached(int(state.interactions), max_interactions)
                    else "interrupted"
                ),
                updates=int(state.updates),
                interactions=int(state.interactions),
                resume_checkpoint=str(checkpoint_path),
                **cost(state),
            )
            return
        record.identity["selection"] = {
            "rule": selection,
            "checkpoint": str(frozen.relative_to(record.directory)),
            "update": int(state.updates),
            "consecutive_passes": consecutive,
            "evaluation_index": evaluation_index,
        }
        if not legacy:
            selected_directory = frozen if orbax_layout else frozen.parent
            record.identity["selection"]["full_validation"] = str(
                (selected_directory / "report.json").relative_to(record.directory)
            )
            record.identity["selection"]["screening"] = str(
                (selected_directory / "screening" / "report.json").relative_to(record.directory)
            )
        atomic_json(record.directory / "run.json", record.identity)
        benchmark_config = OmegaConf.create(resolved)
        benchmark_config.mode = "benchmark"
        benchmark_config.benchmark.seed_base = None
        benchmark_config.benchmark.episodes = None
        if scenes is not None:
            benchmark_config.benchmark.scenes = None
        benchmark_directory = (
            record.directory if orbax_layout else next_evaluation_directory(record.directory)
        )
        report = evaluate(
            benchmark_config,
            benchmark_directory,
            actor=trainer.actor,
            parameters=state.params,
            environments=environments,
        )
        checkpoint(state, export_policy=not orbax_layout)
        record.finish(
            "accepted" if report["passed"] else "frozen_benchmark_failed",
            updates=int(state.updates),
            interactions=int(state.interactions),
            selected_checkpoint=str(frozen.relative_to(record.directory)),
            benchmark_passed=report["passed"],
            benchmark_directory=str(benchmark_directory.relative_to(record.directory)),
            **cost(state),
        )
    except BaseException as error:
        checkpoint(state)
        record.finish(
            "failed",
            updates=int(state.updates),
            interactions=int(state.interactions),
            error=f"{type(error).__name__}: {error}",
            resume_checkpoint=str(checkpoint_path),
            **cost(state),
        )
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if "method" in environments:
            environments["method"].close()


def replay(path: str | Path) -> None:
    """Open one recorded rollout with the upstream native RScope viewer."""
    path = Path(path).resolve()
    if path.suffix != ".mj_unroll" or not path.is_file():
        raise ValueError("replay_path must name an existing .mj_unroll record")
    metadata = path.parent / "rscope_meta.pkl"
    if not metadata.is_file():
        raise ValueError(f"RScope metadata is missing: {metadata}")
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "Native RScope viewer requires a desktop display; this session is headless. "
            "Open the record on a desktop or with the RScope VS Code viewer (docs/replay.md)."
        )
    try:
        import mujoco
        from mujoco import viewer as mujoco_viewer
        from rscope import config as viewer_config
        from rscope.main import main as viewer_main
    except ImportError as error:
        raise RuntimeError(
            "Native RScope viewer unavailable; install the Pixi dependencies"
        ) from error

    original_launch = mujoco_viewer.launch_passive

    @contextmanager
    def launch(*args, **kwargs):
        previous_threads = set(threading.enumerate())
        with original_launch(*args, **kwargs) as viewer:
            viewer_threads = set(threading.enumerate()) - previous_threads
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = args[0].body("drone").id
                viewer.cam.distance = 6.0
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -35.0
            # RScope 0.0.8 nests MuJoCo 3.15's internally locked overlay calls in this lock.
            viewer.lock = nullcontext
            yield viewer
        # MuJoCo closes asynchronously; finish GLFW cleanup before Python unloads its libraries.
        for thread in viewer_threads:
            thread.join()

    previous = viewer_config.BASE_PATH, viewer_config.META_PATH, viewer_config.TEMP_PATH
    try:
        mujoco_viewer.launch_passive = launch
        with tempfile.TemporaryDirectory(prefix="rscope-view-") as temporary:
            base = Path(temporary) / "selected"
            base.mkdir()
            # RScope loads every unroll in BASE_PATH; expose only the selected record.
            (base / path.name).symlink_to(path)
            viewer_config.BASE_PATH = base
            viewer_config.META_PATH = metadata
            viewer_config.TEMP_PATH = Path(temporary) / "cache"
            viewer_main(ssh_enabled=False)
    finally:
        mujoco_viewer.launch_passive = original_launch
        viewer_config.BASE_PATH, viewer_config.META_PATH, viewer_config.TEMP_PATH = previous


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run a resolved Hydra experiment and preserve actual outcomes."""
    if config.mode == "replay":
        if not config.replay_path:
            raise ValueError("replay_path must name an existing .mj_unroll record")
        replay(config.replay_path)
        return
    record = RunRecord(config.output, config, resume=bool(config.resume))
    try:
        if config.mode == "train":
            train(config, record)
            return
        if config.mode not in {"sim", "checkpoint_eval", "benchmark"}:
            raise ValueError(f"Unknown execution mode: {config.mode}")
        actor = parameters = None
        checkpoint_config = None
        if get_class(config.method.implementation._target_).trainable:
            if not config.checkpoint:
                raise ValueError("Frozen policy execution requires checkpoint=<policy.zip>")
            _, metadata = read_checkpoint(config.checkpoint, "inference")
            checkpoint_config = metadata["config"]
            previous = metadata["config"]["experiment"]
            for field in ("task",):
                if previous[field] != OmegaConf.to_container(config[field], resolve=True):
                    raise ValueError(f"Frozen evaluation changes checkpoint {field} contract")
            physical = OmegaConf.to_container(config.simulation, resolve=True)
            for field in ("navigation_world_rotation", "navigation_relative_goal"):
                if previous["observation"].get(field, False) != config.observation.get(
                    field, False
                ):
                    raise ValueError(f"Frozen evaluation changes checkpoint {field} contract")
            if (
                previous["simulation"]["navigation_goal_observation"]
                != physical["navigation_goal_observation"]
            ):
                raise ValueError(
                    "Frozen evaluation changes checkpoint navigation_goal_observation contract"
                )
            if not config.benchmark.allow_environment_change:
                for field in ("dynamics", "drone", "physics_hz", "method_hz"):
                    if previous["simulation"][field] != physical[field]:
                        raise ValueError(f"Frozen evaluation changes physical {field}")
                delay = previous["simulation"]["action_delay_s"]
                if delay != physical["action_delay_s"]:
                    raise ValueError("Frozen evaluation changes action_delay_s contract")
                for field in ("sensor", "observation"):
                    if field in previous and previous[field] != OmegaConf.to_container(
                        config[field], resolve=True
                    ):
                        raise ValueError(f"Frozen evaluation changes checkpoint {field} contract")
                for field in ("randomization", "disturbance"):
                    if previous["simulation"].get(field, {}) != physical[field]:
                        raise ValueError(f"Frozen evaluation changes physical {field}")
            saved_action = metadata["config"]["learning"]["action"]
            requested_action = Action(
                drone=physical["drone"],
                **OmegaConf.to_container(config.method.action, resolve=True),
            )
            archived_action = Action(drone=previous["simulation"]["drone"], **saved_action)
            if requested_action.contract != archived_action.contract:
                raise ValueError(
                    "Frozen evaluation cannot reinterpret checkpoint action units or bounds"
                )
            actor = instantiate(config.method.actor, action_size=requested_action.size)
            actor, parameters, _ = load_policy(config.checkpoint, actor=actor)
        destination = (
            record.directory
            if getattr(record, "layout_version", 2) == 3
            else next_evaluation_directory(record.directory)
        )
        report = evaluate(
            config,
            destination,
            actor=actor,
            parameters=parameters,
            checkpoint_config=checkpoint_config,
        )
        record.finish(
            "completed",
            acceptance_passed=report["passed"],
            evaluation_directory=str(destination.relative_to(record.directory)),
        )
    except BaseException as error:
        if record.identity["status"] == "running":
            record.finish("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
