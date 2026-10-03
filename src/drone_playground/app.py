"""Hydra entry point for all composed training and execution experiments."""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from drone_playground.artifacts.layout import resolve_artifact
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


def _selected_component(saved, requested, arguments, root):
    """Overlay explicitly selected resolved fields onto the saved component."""
    result = copy.deepcopy(saved[root])
    for argument in arguments:
        key = argument.split("=", 1)[0].lstrip("+~")
        target = key.split("@", 1)[-1]
        if target == root:
            return copy.deepcopy(requested[root])
        if not target.startswith(root + "."):
            continue
        parts = target.split(".")[1:]
        source, destination = requested[root], result
        for part in parts[:-1]:
            source = source[part]
            destination = destination.setdefault(part, {})
        leaf = parts[-1]
        if leaf in source:
            destination[leaf] = copy.deepcopy(source[leaf])
        else:
            destination.pop(leaf, None)
    return result


def resolve_checkpoint_execution(config, arguments):
    """Restore frozen identity while retaining explicitly selected execution conditions."""
    from drone_playground.artifacts.schema import require_current

    saved = require_current(
        json.loads(resolve_artifact(config["checkpoint"]).with_suffix(".json").read_text())[
            "config"
        ]
    )
    immutable = ("network.", "method.", "algorithm.", "network=", "algorithm=")
    if any(arg.lstrip("+").startswith("experiment=") for arg in arguments):
        if any(config[name] != saved[name] for name in ("method", "network", "algorithm")):
            raise ValueError("Explicit experiment changes the frozen policy identity")
    if any(arg.lstrip("+").startswith(immutable) for arg in arguments):
        raise ValueError("Checkpoint inference freezes method, network and update identity")
    resolved = copy.deepcopy(saved)
    resolved["env"] = _selected_component(saved, config, arguments, "env")
    resolved["runtime"] = _selected_component(saved, config, arguments, "runtime")
    if resolved["env"]["sensor"] != saved["env"]["sensor"]:
        raise ValueError("Frozen sensor input contract differs")
    if resolved["env"]["task"]["observation"] != saved["env"]["task"]["observation"]:
        raise ValueError("Frozen observation input contract differs")
    for group in ("method", "network", "algorithm", "training"):
        resolved[group] = copy.deepcopy(saved[group])
    for group in (
        "mode",
        "checkpoint",
        "run_id",
        "visualization",
        "replay",
    ):
        resolved[group] = copy.deepcopy(config[group])
    resolved["evaluation"] = _selected_component(saved, config, arguments, "evaluation")
    resolved["evaluation"]["environment"] = "config"
    return resolved


@hydra.main(version_base="1.3", config_path=str(resource_path("configs")), config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    from drone_playground.benchmarks import resolve_protocol_settings

    resolve_protocol_settings(config)
    if isinstance(config.get("replay"), str):
        config["replay"] = {"directory": config["replay"]}
    if config.get("checkpoint") and config["mode"] in ("eval", "play"):
        config = resolve_checkpoint_execution(config, sys.argv[1:])
    device = config["runtime"]["device"]
    mixed = (
        config["method"]["implementation"] == "sampling_mpc"
        and config["method"]["decision"]["prediction_device"] != device
    )
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if mixed else ("cpu" if device == "cpu" else "cuda")
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from drone_playground.composition import run_experiment

    if config.get("run_id"):
        run_id = config["run_id"]
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        seed = config.get("training", {}).get("seed")
        run_id = stamp + (f"-s{seed}" if isinstance(seed, int) else "")
    result = run_experiment(config, Path(config["runtime"]["output_root"]).resolve(), run_id)
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
