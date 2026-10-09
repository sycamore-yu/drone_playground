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
from omegaconf import DictConfig, OmegaConf

from drone_playground.simulation.evaluation import create_environment, evaluate
from drone_playground.simulation.policy import load_policy, read_checkpoint
from drone_playground.simulation.records import RunRecord, atomic_json, gpu_usage
from drone_playground.simulation.scene import Scene

_NAVIGATION_SCENES = ["S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06"]


def train(config: DictConfig, record: RunRecord) -> None:
    """Train until the declared consecutive-evaluation criterion, with no step/time cap."""
    from drone_playground.learning.checkpoint import load_state, save_inference, save_state
    from drone_playground.learning.trainer import Trainer

    started = time.monotonic()
    if config.method.name != "policy":
        raise ValueError("Training requires method=policy or an experiment recipe")
    if config.learning.required_consecutive_passes != 3:
        raise ValueError("C5 requires three consecutive full checkpoint evaluations")
    if config.learning.selection != "last_of_first_three_consecutive_passes":
        raise ValueError("Unsupported checkpoint selection rule")
    resolved = OmegaConf.to_container(config, resolve=True)
    scenes = resolved["learning"].get("scenes")
    scene_updates = resolved["learning"].get("scene_updates", 20)
    scene_index, updates_in_scene = 0, 0
    sampling = None
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
    kind = "state" if config.sensor.name == "state" else config.sensor.name
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
            )
        return trainers[index]

    trainer = trainer_for(scene_index)
    state = trainer.initialize()
    identity = {"experiment": resolved, "learning": trainer.resolved_config}
    if sampling is not None:
        identity["training_sampling"] = sampling
    if config.initial_checkpoint and not config.resume:
        initial_actor, parameters, metadata = load_policy(config.initial_checkpoint)
        if initial_actor.kind != kind or jax.tree.structure(parameters) != jax.tree.structure(
            state.params
        ):
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
    if config.resume:
        state, metadata = load_state(config.resume, state)
        if "initialization" in metadata["config"]:
            identity["initialization"] = metadata["config"]["initialization"]
        previous = metadata["config"]["experiment"]
        for field in ("task", "scene", "sensor", "simulation", "method", "learning", "seed"):
            if previous[field] != resolved[field]:
                raise ValueError(f"Resume changes the saved {field} contract; use a new experiment")
        if scenes is not None and (
            int(state.updates) < updates_in_scene
            or (int(state.updates) - updates_in_scene) % (scene_updates * len(scenes))
            != scene_index * scene_updates
        ):
            raise ValueError("Checkpoint training sampling progress disagrees with update count")
        progress = metadata["provenance"]
        consecutive = progress.get("consecutive_passes", 0)
        evaluation_index = progress.get("evaluation_index", 0)
        previous_wall_seconds = progress.get("training_wall_seconds")
        pending_evaluation = (
            "evaluation_index" in progress
            and int(state.updates) % config.learning.evaluation_interval == 0
            and evaluation_index < int(state.updates) // config.learning.evaluation_interval
        )
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
    environments = {}
    initial_interactions = int(state.interactions)
    checkpoint_path = record.directory / "checkpoints" / "latest.training.zip"
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

    def checkpoint(current):
        provenance = {
            **record.identity,
            "updates": int(current.updates),
            "interactions": int(current.interactions),
            "consecutive_passes": consecutive,
            "evaluation_index": evaluation_index,
            **cost(current),
        }
        if scenes is not None:
            provenance["training_sampling"] = {
                "scene_index": scene_index,
                "updates_in_scene": updates_in_scene,
            }
        save_state(checkpoint_path, current, config=identity, provenance=provenance)
        frozen = record.directory / "checkpoints" / f"update-{int(current.updates):08d}.policy.zip"
        save_inference(frozen, current.params, kind=kind, config=identity, provenance=provenance)
        return frozen

    def evaluate_checkpoint():
        nonlocal consecutive, evaluation_index
        update = int(state.updates)
        frozen = checkpoint(state)
        directory = record.directory / "checkpoint_eval" / f"update-{update:08d}"
        if directory.exists():
            previous = Path(
                tempfile.mkdtemp(prefix=f"incomplete-{directory.name}-", dir=directory.parent)
            )
            directory.rename(previous)
            record.event(
                "checkpoint_evaluation_retry",
                update=update,
                previous_output=str(previous),
            )
        report = evaluate(
            evaluation_config,
            directory,
            actor=trainer.actor,
            parameters=state.params,
            evaluation_index=evaluation_index + 1,
            environments=environments,
        )
        evaluation_index += 1
        consecutive = consecutive + 1 if report["passed"] else 0
        checkpoint(state)
        record.event(
            "checkpoint_evaluation",
            update=update,
            passed=report["passed"],
            consecutive_passes=consecutive,
            checkpoint=str(frozen),
        )

    try:
        if pending_evaluation:
            evaluate_checkpoint()
        while not stop_requested and consecutive < config.learning.required_consecutive_passes:
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
            if update % config.learning.checkpoint_interval == 0:
                checkpoint(state)
            if update % config.learning.evaluation_interval:
                continue
            evaluate_checkpoint()
        frozen = checkpoint(state)
        if consecutive < config.learning.required_consecutive_passes:
            record.finish(
                "interrupted",
                updates=int(state.updates),
                interactions=int(state.interactions),
                resume_checkpoint=str(checkpoint_path),
                **cost(state),
            )
            return
        atomic_json(
            record.directory / "selection.json",
            {
                "rule": config.learning.selection,
                "checkpoint": str(frozen),
                "update": int(state.updates),
                "consecutive_passes": consecutive,
                "evaluation_index": evaluation_index,
            },
        )
        benchmark_config = OmegaConf.create(resolved)
        benchmark_config.mode = "benchmark"
        benchmark_config.benchmark.seed_base = None
        benchmark_config.benchmark.episodes = None
        if scenes is not None:
            benchmark_config.benchmark.scenes = None
        benchmark_directory = record.directory / "benchmark"
        if benchmark_directory.exists():
            benchmark_directory = Path(tempfile.mkdtemp(prefix="benchmark-", dir=record.directory))
        report = evaluate(
            benchmark_config,
            benchmark_directory,
            actor=trainer.actor,
            parameters=state.params,
            environments=environments,
        )
        checkpoint(state)
        record.finish(
            "accepted" if report["passed"] else "frozen_benchmark_failed",
            updates=int(state.updates),
            interactions=int(state.interactions),
            selected_checkpoint=str(frozen),
            benchmark_passed=report["passed"],
            benchmark_directory=str(benchmark_directory),
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
        if config.method.name == "policy":
            if not config.checkpoint:
                raise ValueError("Frozen policy execution requires checkpoint=<policy.zip>")
            actor, parameters, metadata = load_policy(config.checkpoint)
            previous = metadata["config"]["experiment"]
            for field in ("task", "sensor"):
                if previous[field] != OmegaConf.to_container(config[field], resolve=True):
                    raise ValueError(f"Frozen evaluation changes checkpoint {field} contract")
            physical = OmegaConf.to_container(config.simulation, resolve=True)
            for field in ("dynamics", "drone", "physics_hz", "method_hz"):
                if previous["simulation"][field] != physical[field]:
                    raise ValueError(f"Frozen evaluation changes physical {field}")
            if previous["method"]["action_delay"] != config.method.action_delay:
                raise ValueError("Frozen evaluation changes checkpoint action_delay contract")
        report = evaluate(config, record.directory, actor=actor, parameters=parameters)
        record.finish("completed", acceptance_passed=report["passed"])
    except BaseException as error:
        if record.identity["status"] == "running":
            record.finish("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
