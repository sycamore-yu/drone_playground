"""Resolve run conditions and directly instantiate the six environment components."""

from __future__ import annotations

import copy
from dataclasses import replace

from hydra.utils import instantiate


def component_identity(config: dict) -> dict:
    """Record the resolved component configuration without a second default source."""
    identity = copy.deepcopy(config["env"])
    if "algorithm" in config:
        identity["gradient"] = copy.deepcopy(config["algorithm"]["gradient"])
    return identity


def observation_spec(config: dict) -> dict:
    """Return the task-owned observation specification."""
    return copy.deepcopy(config["env"]["task"]["observation"])


def build_sensor(config: dict):
    """Instantiate the configured sensor, with absence explicitly represented."""
    spec = config["env"]["sensor"]
    return (
        None if spec is None or spec.get("name") == "none" else instantiate(spec, _convert_="all")
    )


def build_observer(config: dict, sensor=None):
    """Bind observation dimensions to the actual sensor calibration."""
    observer = instantiate(observation_spec(config), _convert_="all")
    bind = getattr(observer, "bind_sensor", None)
    return observer if bind is None else bind(sensor)


def resolve_conditions(config: dict, role: str) -> dict:
    """Resolve training and evaluation changes once, before constructing components."""
    if role not in ("train", "eval"):
        raise ValueError("Environment role must be train or eval")
    runtime = config["runtime"]
    evaluation = config.get("evaluation", {})
    source = config.get("training", {}) if role == "train" else evaluation.get("conditions", {})
    conditions = copy.deepcopy(source)
    conditions["action_delay_ms"] = runtime.get("action_delay_ms")
    conditions["action_delay_steps"] = runtime.get("action_delay_steps", 0)
    conditions["initial_conditions"] = copy.deepcopy(evaluation.get("initial_conditions"))
    if evaluation.get("commanded_speed") is not None:
        conditions["commanded_speed"] = evaluation["commanded_speed"]
    return conditions


def build_dynamics(config: dict, conditions=None):
    """Instantiate physical parameters and the explicitly selected derivative rule."""
    spec = copy.deepcopy(config["env"]["dynamics"])
    conditions = {} if conditions is None else conditions
    if conditions.get("domain_randomization") is not None:
        spec["domain_randomization"] = conditions["domain_randomization"]
    gradient = config.get("algorithm", {}).get("gradient")
    if gradient is not None:
        spec["backward"] = gradient["transition"]
        if "decay_rate" in gradient:
            spec["decay_rate"] = gradient["decay_rate"]
    model = instantiate(spec, _convert_="all")
    if hasattr(model, "disturbance"):
        model = replace(model, disturbance=conditions.get("disturbance"))
    return model


def build_controller(spec):
    """Construct the configured control component; native bounds are bound later."""
    return instantiate(spec, _convert_="all")


def build_environment(config, device=None, role="eval", count=1, *, scene=None, reference=None):
    """Build the same environment for training, frozen evaluation and Python use."""
    from drone_playground.environments.randomization import (
        validate_command_distribution,
        validate_effects,
    )

    cfg = copy.deepcopy(config)
    settings = cfg["env"]
    conditions = resolve_conditions(cfg, role)
    effects = {
        name: conditions.get(name)
        for name in ("reset_randomization", "observation_noise", "action_noise", "disturbance")
    }
    validate_effects(effects)
    command = conditions.get("command_distribution") or settings["task"]["command_distribution"]
    validate_command_distribution(command)
    conditions["command_distribution"] = command
    distribution = conditions.get("scene_distribution") or {"type": "fixed", "scene": None}
    if distribution.get("type") not in ("fixed", "generated", "procedural"):
        raise ValueError("Scene distribution must be fixed, generated or procedural")
    if distribution.get("scene") is not None:
        settings["scene"] = copy.deepcopy(distribution["scene"])
    sensor = build_sensor(cfg)
    observer = build_observer(cfg, sensor)
    task_spec = copy.deepcopy(settings["task"])
    task_spec.pop("observation")
    task_factory = instantiate(task_spec, _convert_="all", _partial_=True)
    task = task_factory(observation=observer)
    dynamics = build_dynamics(cfg, conditions)
    controller = build_controller(settings["controller"])
    method = cfg.get("method", {})
    if controller.native_mode == "acceleration":
        if method.get("network_output_frame") not in (None, "body"):
            raise ValueError("Acceleration policy frame must match its body-to-world transform")
        if method.get("command_units") not in (None, "m/s^2"):
            raise ValueError("Acceleration policy units must be m/s^2")
    scene = instantiate(settings["scene"], _convert_="all") if scene is None else scene
    reference = (
        instantiate(settings["reference"], _convert_="all") if reference is None else reference
    )
    constructor = instantiate({"_target_": settings["_target_"]}, _partial_=True)
    env = constructor(
        dynamics=dynamics,
        controller=controller,
        reference=reference,
        scene=scene,
        sensor=sensor,
        task=task,
        name=settings["name"],
        conditions=conditions,
        device=cfg["runtime"]["device"] if device is None else device,
        role=role,
        count=count,
        seed=cfg["runtime"]["scene_seed_" + role],
    )
    env.scene_distribution = distribution
    from drone_playground.runtime.timing import sensor_schedule

    env.sensor_timing = sensor_schedule(sensor, env.freq)
    if env.sensor_timing is not None:
        env.sensor_calibration = {**env.sensor_calibration, "schedule": env.sensor_timing}
    if not getattr(task, "owns_environment_effects", False):
        from drone_playground.environments.randomization import EnvironmentEffects

        wrapper_effects = copy.deepcopy(effects)
        if getattr(task, "owns_reset_randomization", False):
            wrapper_effects["reset_randomization"] = None
        if any(wrapper_effects.values()):
            env = EnvironmentEffects(env, wrapper_effects)
    delay_steps = conditions["action_delay_steps"]
    delay_range = conditions["action_delay_ms"]
    if delay_steps and delay_range is not None:
        raise ValueError("Select one action-delay representation")
    if delay_steps:
        from drone_playground.control.delay import ActionDelay

        env = ActionDelay(env, delay_steps)
    if delay_range is not None and not getattr(env, "handles_transport_delay", False):
        from drone_playground.control.delay import RandomActionDelay

        env = RandomActionDelay(env, delay_range)
    decoder = cfg.get("method", {}).get("physical_decoder")
    if decoder is not None:
        from drone_playground.control.wrappers import GeometricActionWrapper

        env = GeometricActionWrapper(env, decoder)
    env.component_identity = component_identity(cfg)
    if getattr(env, "sim", None) is not None:
        env.sim.component_identity = copy.deepcopy(env.component_identity)
    env.experiment_config = cfg
    return env


def load(env_name, *, overrides=(), role="eval", count=1):
    """Load an environment through its Hydra group name, without a learner recipe."""
    from drone_playground.configuration import load_config

    config = load_config("environment", ["env=" + env_name, *overrides])
    return build_environment(config, role=role, count=count)


def build_reference_environment(task, device, *, scene=None, reference=None):
    """Build task geometry and canonical initial conditions without a training recipe."""
    from drone_playground.configuration import load_config

    config = load_config("environment", ["env=" + task, "runtime.device=" + device])
    return build_environment(config, role="eval", count=1, scene=scene, reference=reference)
