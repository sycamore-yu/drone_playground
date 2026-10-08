"""Load the project's Hydra files without constructing experiments or environments."""

import copy
import json
from contextlib import nullcontext
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from drone_playground.resources import resource_path

CONFIG_ROOT = resource_path("configs")


def load_config(name="config", overrides=()):
    """Resolve one configuration while preserving an already active CLI context."""
    current = GlobalHydra.instance()
    if current.is_initialized():
        roots = current.config_loader().get_search_path().get_path()
        if not any(
            entry.provider == "main"
            and Path(entry.path.removeprefix("file://")).resolve() == CONFIG_ROOT
            for entry in roots
        ):
            raise ValueError("Active Hydra belongs to a different configuration root")
        context = nullcontext()
    else:
        context = initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT))
    with context:
        return OmegaConf.to_container(
            compose(config_name=name, overrides=list(overrides)),
            resolve=True,
            throw_on_missing=True,
        )


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
    if "freq" not in env or {"freq", "physics_freq"} & env["task"].keys():
        raise ValueError("Environment clocks belong to env; migrate the task clock fields")
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
    if (
        method.get("_target_", "").startswith("drone_playground.integrations.")
        and config["runtime"].get("backend") != "host"
    ):
        raise ValueError("External method backend must use host execution")
    if config["evaluation"].get("protocol"):
        from drone_playground.benchmarks import protocol_identity

        protocol_identity(config)


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
        json.loads(Path(config["checkpoint"]).with_suffix(".json").read_text())["config"]
    )
    immutable = ("network.", "method.", "algorithm.", "network=", "algorithm=")
    experiment_selected = any(arg.lstrip("+").startswith("experiment=") for arg in arguments)
    if experiment_selected:
        requested_identity = (
            config["method"].get("name"),
            config["method"].get("implementation"),
            config["method"].get("output"),
            config["algorithm"].get("name"),
        )
        saved_identity = (
            saved["method"].get("name"),
            saved["method"].get("implementation"),
            saved["method"].get("output"),
            saved["algorithm"].get("name"),
        )
        if requested_identity != saved_identity:
            raise ValueError("Explicit experiment changes the frozen policy identity")
    if any(arg.lstrip("+").startswith(immutable) for arg in arguments):
        raise ValueError("Checkpoint inference freezes method, network and update identity")
    resolved = copy.deepcopy(saved)
    resolved["env"] = (
        copy.deepcopy(config["env"])
        if experiment_selected
        else _selected_component(saved, config, arguments, "env")
    )
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


def prepare_experiment(config, overrides=()):
    """Apply the same experiment preparation for CLI and Python execution.

    Explicit overrides are required when selecting execution settings for a
    frozen checkpoint; resolved defaults are never treated as user intent.
    """
    from drone_playground.benchmarks import resolve_protocol_settings

    config = copy.deepcopy(config)
    resolve_protocol_settings(config)
    if isinstance(config.get("replay"), str):
        config["replay"] = {"directory": config["replay"]}
    if config.get("checkpoint") and config["mode"] in ("eval", "play"):
        config = resolve_checkpoint_execution(config, overrides)
    return config
