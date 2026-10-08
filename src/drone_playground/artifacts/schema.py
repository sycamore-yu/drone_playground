"""Validate current artifacts without runtime migration or alternate defaults."""

import copy


def require_current(config: dict) -> dict:
    """Require a fully resolved v4 configuration before loading model parameters."""
    value = config.get("components", config)
    if value.get("config_version") != 4:
        raise ValueError("Artifact config_version must be 4; migrate older metadata explicitly")
    env = value.get("env", {})
    if not {"dynamics", "controller", "reference", "scene", "sensor", "task"} <= env.keys():
        raise ValueError("Saved environment does not declare all six components")
    if {"action", "execution", "observation"} & env.keys():
        raise ValueError("Saved environment contains retired component fields")
    if "freq" not in env or {"freq", "physics_freq"} & env["task"].keys():
        raise ValueError("Saved clocks must belong to env; migrate older metadata explicitly")
    return copy.deepcopy(value)
