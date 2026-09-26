"""Hydra entry point for all composed training and execution experiments."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    device = config["training"]["device"]
    mixed = (
        config["controller"]["name"] == "sampling_mpc"
        and config["controller"]["prediction_device"] != device
    )
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if mixed else ("cpu" if device == "cpu" else "cuda")
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from drone_playground.composition import run_experiment

    run_id = config.get("run_id") or "-".join(
        [
            config["task"]["name"],
            config["algorithm"]["name"],
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"),
        ]
    )
    result = run_experiment(config, ROOT, run_id)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
