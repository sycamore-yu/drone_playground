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

ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_recipe(config):
    implementation = config["method"]["implementation"]
    fixed = {"pointcloud_recurrent": "paper/pointcloud_flight", "lotf_mlp": "paper/lotf"}
    return fixed.get(implementation, "learning/" + config["algorithm"]["name"])


def script_main(mode):
    """Prepare a Hydra invocation without importing a simulation or training engine."""
    arguments = sys.argv[1:]
    checkpoint = next(
        (arg.split("=", 1)[1] for arg in arguments if arg.startswith("checkpoint=")), None
    )
    if checkpoint:
        from drone_playground.runs.migration import require_current

        saved = require_current(
            json.loads(Path(checkpoint).with_suffix(".json").read_text())["config"]
        )
        if not any(arg.startswith("method=") for arg in arguments):
            arguments.insert(0, "method=" + _checkpoint_recipe(saved))
    arguments = [arg for arg in arguments if not arg.startswith("mode=")]
    generated = ["mode=" + mode]
    if mode == "play" and not any(arg.startswith("evaluation.episodes=") for arg in arguments):
        generated.append("evaluation.episodes=1")
    # Hydra's argument parser expects positional overrides before display flags.
    position = next((i for i, arg in enumerate(arguments) if arg.startswith("--")), len(arguments))
    arguments[position:position] = generated
    sys.argv = [sys.argv[0], *arguments]
    main()


def resolve_checkpoint_execution(config, arguments):
    """Restore frozen identity while retaining explicitly selected execution conditions."""
    from drone_playground.runs.migration import require_current

    saved = require_current(
        json.loads(Path(config["checkpoint"]).with_suffix(".json").read_text())["config"]
    )
    if config["method"]["implementation"] != saved["method"]["implementation"]:
        raise ValueError("Selected method and frozen checkpoint identity differ")
    immutable = ("network.", "method.", "algorithm.")
    if any(arg.lstrip("+").startswith(immutable) for arg in arguments):
        raise ValueError("Checkpoint inference freezes method, network and update identity")
    explicit_env = any(
        arg.startswith("env=") or arg.lstrip("+").startswith("env.") for arg in arguments
    )
    resolved = copy.deepcopy(config if explicit_env else saved)
    for group in ("method", "network", "algorithm", "training"):
        resolved[group] = copy.deepcopy(saved[group])
    for group in (
        "mode",
        "checkpoint",
        "run_id",
        "evaluation",
        "runtime",
        "visualization",
        "replay",
    ):
        resolved[group] = copy.deepcopy(config[group])
    resolved["evaluation"]["environment"] = "config"
    return resolved


@hydra.main(version_base="1.3", config_path=str(ROOT / "configs"), config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
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

    run_id = config.get("run_id") or "-".join(
        [
            config["env"]["task"]["name"],
            config["algorithm"]["name"],
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"),
        ]
    )
    result = run_experiment(config, ROOT, run_id)
    if config["mode"] == "play" and result.get("replay_directory"):
        from drone_playground.visualization.viewer import replay

        result["viewer"] = replay(
            Path(result["replay_directory"]), publish=config["visualization"].get("publish", True)
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
