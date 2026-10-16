"""Command entry points for training, playback, benchmarking, and replay."""

import json
import os
import sys
import tempfile
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path

import hydra
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from drone_playground.simulation.methods import Action
from drone_playground.simulation.policy import load_policy, read_checkpoint, resolve_checkpoint
from drone_playground.simulation.records import RunRecord


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


def playback_config(config, metadata):
    """Restore the saved Actor, sensor, action, and physical execution settings."""
    saved = metadata["config"]["experiment"]
    if saved["task"]["name"] != config.task.name:
        raise ValueError("Frozen evaluation changes checkpoint task contract")
    resolved = OmegaConf.to_container(config, resolve=True)
    resolved["method"] = saved["method"]
    if not config.benchmark.allow_environment_change:
        for field in ("sensor", "observation"):
            resolved[field] = saved[field]
        worlds = resolved["simulation"]["num_envs"]
        resolved["simulation"] = {**saved["simulation"], "num_envs": worlds}
    physical = resolved["simulation"]
    requested = Action(drone=physical["drone"], **resolved["method"]["action"])
    if requested.contract != metadata["config"]["learning"]["action"]:
        raise ValueError("Frozen evaluation cannot reinterpret checkpoint action units or bounds")
    return OmegaConf.create(resolved)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run a resolved experiment and preserve its actual outcomes."""
    if config.mode == "replay":
        if not config.replay_path:
            raise ValueError("replay_path must name an existing .mj_unroll record")
        replay(config.replay_path)
        return
    if config.mode not in {"train", "play", "benchmark"}:
        raise ValueError(f"Unknown execution mode: {config.mode}")
    actor = parameters = checkpoint_config = evaluated_checkpoint = None
    if config.mode != "train" and get_class(config.method.implementation._target_).trainable:
        if not config.checkpoint:
            raise ValueError(
                "Policy playback requires checkpoint=latest, best, or an Orbax model path"
            )
        selector = config.checkpoint
        path = resolve_checkpoint(
            selector,
            root=config.checkpoint_root,
            task=config.task.name,
            sensor=config.sensor.name,
            algorithm=config.learning.algorithm,
            obstacle_motion=config.simulation.get("procedural", {}).get("obstacle_motion"),
        )
        metadata = read_checkpoint(path, "inference")
        config = playback_config(config, metadata)
        actor = instantiate(
            config.method.actor, action_size=len(metadata["config"]["learning"]["action"]["low"])
        )
        actor, parameters, metadata = load_policy(path, actor=actor)
        checkpoint_config = metadata["config"]
        evaluated_checkpoint = {
            "path": str(path),
            "selector": selector,
            "best_fallback_to_latest": selector == "best"
            and metadata["provenance"].get("selection", {}).get("best_checkpoint") != str(path),
        }
        print(json.dumps({"checkpoint": evaluated_checkpoint}), flush=True)
    record = RunRecord(config.output, config, resume=bool(config.resume))
    try:
        if config.mode == "train":
            from drone_playground.learning.runner import train

            train(config, record)
            return
        from drone_playground.benchmark.evaluation import evaluate

        if evaluated_checkpoint is not None:
            record.identity["evaluated_checkpoint"] = evaluated_checkpoint
        report = evaluate(
            config,
            record.directory,
            actor=actor,
            parameters=parameters,
            checkpoint_config=checkpoint_config,
        )
        record.finish("completed", evaluation_complete=report["complete"])
    except BaseException as error:
        if record.identity["status"] == "running":
            record.finish("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
