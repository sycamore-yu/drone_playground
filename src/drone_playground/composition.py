"""Compose one experiment and dispatch only its declared run mode."""

from __future__ import annotations

import copy
from pathlib import Path

from hydra.utils import get_method

from drone_playground.configuration import load_config


def compose_experiment(experiment="control/ppo", environment=None, overrides=None):
    """Use the same Hydra groups for CLI and programmatic experiment composition."""
    choices = ["experiment=" + experiment]
    if environment is not None:
        choices.append("env=" + environment)
    config = load_config(overrides=[*choices, *(overrides or [])])
    from drone_playground.benchmarks import resolve_protocol_settings

    resolve_protocol_settings(config)
    return config


def validate_config(config):
    """Check the experiment boundary; components validate their own parameters."""
    if config.get("config_version") != 4:
        raise ValueError("config_version must be 4; migrate older artifacts explicitly")
    for name in ("env", "method", "algorithm", "network", "training", "runtime", "evaluation"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing configuration namespace: {name}")
    mode = config["mode"]
    if mode not in ("train", "eval", "play"):
        raise ValueError("Mode must be train, eval or play")
    env = config["env"]
    for component in ("dynamics", "controller", "reference", "scene", "sensor", "task"):
        if component not in env:
            raise ValueError(f"Missing environment component: {component}")
    if {"adapter", "trainer", "evaluation_entrypoint", "policy_evaluator"} & env["task"].keys():
        raise ValueError("Task config contains retired algorithm or runner fields")
    if mode == "train":
        if not config["method"]["trainable"]:
            raise ValueError("The selected runtime method has no training entry")
        if not config["algorithm"].get("trainer"):
            raise ValueError("Training requires an explicit algorithm.trainer")
        settings = config["training"]
        if settings["seed"] < 0 or settings["num_envs"] < 1:
            raise ValueError("Seed must be nonnegative and environment count positive")
        if settings.get("resume") and settings.get("warm_start"):
            raise ValueError("Choose resume or warm start, not both")
    if config["runtime"].get("timing") != "synchronous":
        raise ValueError("Runtime uses synchronous simulated time")
    method = config["method"]
    if method.get("_target_", "").startswith("drone_playground.integrations."):
        if config["runtime"].get("backend") != "host":
            raise ValueError("External method backend must use host execution")
    if config["evaluation"].get("protocol"):
        from drone_playground.benchmarks import protocol_identity

        protocol_identity(config)


def run_experiment(config, root: Path, run_id):
    """Run the configured learner or frozen evaluator, preserving component identity."""
    from drone_playground.artifacts.layout import resolve_artifact
    from drone_playground.runtime.devices import execution_scope

    config = copy.deepcopy(config)
    if config.get("checkpoint"):
        config["checkpoint"] = str(resolve_artifact(config["checkpoint"]))
    for field in ("resume", "warm_start"):
        if config.get("training", {}).get(field):
            config["training"][field] = str(resolve_artifact(config["training"][field]))
    if config.get("evaluation", {}).get("training_run"):
        config["evaluation"]["training_run"] = str(
            resolve_artifact(config["evaluation"]["training_run"])
        )
    if config["mode"] == "play" and config.get("replay", {}).get("directory"):
        from drone_playground.visualization.viewer import replay

        return replay(
            resolve_artifact(config["replay"]["directory"]),
            publish=config["visualization"].get("publish", True),
        )
    validate_config(config)
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
