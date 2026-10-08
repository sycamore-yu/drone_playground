"""Hydra entry point for all composed training and execution experiments."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from drone_playground.configuration import prepare_experiment, validate_config
from drone_playground.resources import resource_path


def script_main(mode):
    """Prepare a Hydra invocation without importing a simulation or training engine."""
    arguments = sys.argv[1:]
    arguments = [arg for arg in arguments if not arg.startswith("mode=")]
    generated = ["mode=" + mode]
    if mode == "play" and not any(arg.startswith("evaluation.episodes=") for arg in arguments):
        generated.append("evaluation.episodes=1")
    # Hydra's argument parser expects positional overrides before display flags.
    position = next(
        (i for i, arg in enumerate(arguments) if arg.startswith("--")),
        len(arguments),
    )
    arguments[position:position] = generated
    sys.argv = [sys.argv[0], *arguments]
    main()


def run_experiment(config, root: Path, run_id, *, overrides=()):
    """Run the configured learner or frozen evaluator, preserving component identity."""
    config = prepare_experiment(config, overrides)
    device = config["runtime"]["device"]
    mixed = (
        config["method"]["implementation"] == "sampling_mpc"
        and config["method"]["decision"]["prediction_device"] != device
    )
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if mixed else ("cpu" if device == "cpu" else "cuda")
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if config.get("checkpoint"):
        config["checkpoint"] = str(Path(config["checkpoint"]))
    for field in ("resume", "warm_start"):
        if config.get("training", {}).get(field):
            config["training"][field] = str(Path(config["training"][field]))
    if config.get("evaluation", {}).get("training_run"):
        config["evaluation"]["training_run"] = str(Path(config["evaluation"]["training_run"]))
    if config["mode"] == "play" and config.get("replay", {}).get("directory"):
        from drone_playground.visualization.viewer import replay

        return replay(
            Path(config["replay"]["directory"]),
            publish=config["visualization"].get("publish", True),
        )
    validate_config(config)
    from drone_playground.runtime.devices import execution_scope

    with execution_scope(config["runtime"]["device"]):
        if config["mode"] == "train":
            return get_method(config["algorithm"]["trainer"])(config, root, run_id)
        if config["mode"] == "play":
            config["evaluation"]["record_replays"] = True
        result = get_method(config["evaluation"]["entrypoint"])(config, root, run_id)
        if config["mode"] == "play":
            from drone_playground.artifacts.layout import experiment_directory

            result["replay_directory"] = str(experiment_directory(root, run_id) / "rollouts")
        return result


@hydra.main(version_base="1.3", config_path=str(resource_path("configs")), config_name="config")
def main(cfg: DictConfig):
    """Dispatch a Hydra-composed experiment to its configured execution path."""
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if config.get("run_id"):
        run_id = config["run_id"]
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        seed = config.get("training", {}).get("seed")
        run_id = stamp + (f"-s{seed}" if isinstance(seed, int) else "")
    result = run_experiment(
        config, Path(config["runtime"]["output_root"]).resolve(), run_id, overrides=sys.argv[1:]
    )
    if config["mode"] == "play" and result.get("replay_directory"):
        from drone_playground.visualization.viewer import replay

        result["viewer"] = replay(
            Path(result["replay_directory"]),
            publish=config["visualization"].get("publish", True),
        )
    print(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
