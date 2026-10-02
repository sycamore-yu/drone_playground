"""Construct the world, task and execution components of a complete environment."""

from __future__ import annotations

import copy
from dataclasses import replace

from hydra.utils import get_class, instantiate



ROLE_SEEDS = {
    "train": 10000,
    "eval": 20000,
}


def component_identity(config: dict) -> dict:
    """A reporting view: instances and execution consume only the canonical config."""
    env = config["env"]
    return copy.deepcopy(
        dict(
            method=config["method"],
            task=env["task"],
            scene=env["scene"],
            sensor=env["sensor"],
            observation=env["observation"],
            action=env["action"],
            dynamics=env["dynamics"],
            gradient=config["algorithm"]["gradient"],
            objective=config["objective"],
        )
    )


def observation_spec(config: dict) -> dict:
    return copy.deepcopy(config["env"]["observation"])


def build_sensor(config: dict):
    settings = config["env"]["sensor"]
    return (
        None
        if settings is None or settings.get("name") == "none"
        else instantiate(settings, _convert_="all")
    )


def build_observer(config: dict, sensor=None):
    spec = observation_spec(config)
    observer = instantiate(spec)
    if sensor is None or spec["name"] in ("flight_state", "state_reference"):
        return observer
    if spec["name"] == "navigation_depth":
        return replace(
            observer,
            history=sensor.history,
            points_per_frame=sensor.points_per_frame,
            channels=sensor.channels,
            near_m=sensor.near_m,
            far_m=sensor.far_m,
        )
    if spec["name"] == "navigation_lidar":
        return replace(
            observer,
            history=sensor.history,
            points_per_frame=sensor.points_per_frame,
            channels=sensor.channels,
            near_m=sensor.range_m[0],
            far_m=sensor.normalise_far_m,
        )
    raise ValueError(f"Sensor and observation interface differ: {spec['name']}")


def _accepts_keyword(target, name: str) -> bool:
    """Whether a dynamics target declares the field, so DR never leaks to others."""
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(target):
        return any(field.name == name for field in dataclasses.fields(target))
    parameters = inspect.signature(target.__init__).parameters
    return name in parameters or any(
        p.kind is p.VAR_KEYWORD for p in parameters.values()
    )


def build_dynamics(config: dict):
    spec = copy.deepcopy(config["env"]["dynamics"])
    randomization = config.get("training", {}).get("domain_randomization")
    if randomization is not None and not isinstance(randomization, dict):
        raise ValueError("training.domain_randomization must be a mapping")
    if randomization and randomization.get("enabled"):
        target = spec.get("_target_")
        if not isinstance(target, str) or not _accepts_keyword(
            get_class(target), "domain_randomization"
        ):
            raise ValueError(
                "training.domain_randomization is enabled for a dynamics model that "
                "does not sample randomization parameters"
            )
        spec["domain_randomization"] = copy.deepcopy(randomization)
    gradient = config["algorithm"]["gradient"]
    spec["backward"] = gradient["transition"]
    if "decay_rate" in gradient:
        spec["decay_rate"] = gradient["decay_rate"]
    try:
        return instantiate(spec, _convert_="all")
    except Exception as error:
        if isinstance(error.__cause__, ValueError):
            raise error.__cause__ from error
        raise



def build_environment(config, device="cpu", role="train", count=32):

    if role not in ROLE_SEEDS or count < 1:
        raise ValueError("Choose train/eval and a positive instance count")
    cfg = copy.deepcopy(config)
    from drone_playground.environments.randomization import (
        EnvironmentEffects,
        validate_command_distribution,
        validate_effects,
    )

    training = cfg.get("training", {})
    distribution = training.get("scene_distribution") or {
        "type": "fixed",
        "scene": None,
    }
    if distribution.get("type") not in (
        "fixed",
        "generated",
        "procedural",
    ) or set(distribution) - {"type", "scene"}:
        raise ValueError(
            "Scene distribution requires fixed/generated/procedural and a scene configuration"
        )
    if (
        "scene" in training
        or "navigation_initialization" in training
        or "point_noise_std_m" in training
    ):
        raise ValueError(
            "Retired training fields: use scene_distribution, reset_randomization and observation_noise"
        )
    effects = {
        name: training.get(name)
        for name in (
            "reset_randomization",
            "observation_noise",
            "action_noise",
            "disturbance",
        )
    }
    if role != "train":
        randomization = cfg.get("training", {}).get("domain_randomization")
        if randomization:
            randomization["enabled"] = False
        effects = {name: None for name in effects}
    validate_effects(effects)
    settings = cfg["env"]
    task = settings["task"]
    if (
        task["name"] != "navigation"
        and task.get("adapter") != "recurrent_flight"
        and distribution["type"] != "fixed"
    ):
        raise ValueError(
            "Reference control tasks currently use fixed scene geometry"
        )
    if (
        task.get("adapter") == "recurrent_acceleration"
        and distribution["type"] == "procedural"
    ):
        raise ValueError(
            "Recurrent navigation uses fixed/generated scene banks; procedural sampling requires the source sampling environment"
        )
    if (
        role == "train"
        and task.get("adapter") == "recurrent_flight"
        and distribution.get("scene")
    ):
        settings["scene"] = copy.deepcopy(distribution["scene"])
    command = (
        training.get("command_distribution") if role == "train" else None
    ) or task["command_distribution"]
    validate_command_distribution(command)
    supported = (
        {"reference"}
        if task["name"] == "racing"
        or task.get("adapter") == "recurrent_reference"
        else {"position", "goal_velocity"}
        if task.get("adapter") in ("recurrent_flight", "recurrent_acceleration")
        else {"position"}
        if task["name"] == "navigation"
        else {"reference", "position", "velocity"}
    )
    if command["kind"] not in supported:
        raise ValueError(f"Task supports command kinds {sorted(supported)}")
    if (
        supported == {"reference"}
        and command["distribution"] != "reference_bank"
    ):
        raise ValueError(
            "This task consumes its reference bank; fixed or sampled position/velocity commands use Tracking or Navigation"
        )
    action = effects.get("action_noise") or {}
    if {"std_normalized", "bias_normalized"} & set(action) and {
        "std_physical",
        "bias_physical",
    } & set(action):
        raise ValueError(
            "Action uncertainty must choose normalized or physical units"
        )
    noise = effects.get("observation_noise") or {}
    if settings["sensor"].get("name") == "none" and (
        noise.get("sensor_std_m", 0)
        or noise.get("sensor_dropout_probability", 0)
    ):
        raise ValueError("Sensor measurement noise requires an actual sensor")
    if task.get("adapter"):
        noise = effects.get("observation_noise") or {}
        reset = effects.get("reset_randomization") or {}
        if (
            "angular_velocity_std_radps" in noise
            or "angular_velocity_std_radps" in reset
        ):
            raise ValueError(
                "Acceleration-driven point mass has no angular-velocity state"
            )
        if task["adapter"] == "recurrent_reference" and "position" in reset:
            raise ValueError(
                "Reference reset uses position_std_m or position_half_width_m, not navigation position mixtures"
            )
        if task["adapter"] == "recurrent_flight" and set(reset) & {
            "position",
            "scene_phase_s",
        }:
            raise ValueError(
                "Procedural point-cloud reset has no catalog mixture or reference phase"
            )
    if task.get("_target_"):
        from hydra.utils import get_class


        target = get_class(task["_target_"])
        arguments = {
            name: value
            for name, value in dict(
                device=device, role=role, count=count
            ).items()
            if _accepts_keyword(target, name)
        }
        env = instantiate(
            {"_target_": task["_target_"]},
            cfg,
            _recursive_=False,
            _convert_="all",
            **arguments,
        )
    else:

        model = build_dynamics(cfg)
        scene = instantiate(settings["scene"], _convert_="all")
        sensor = build_sensor(cfg)
        observer = build_observer(cfg, sensor)
        objective = instantiate(cfg["objective"])
        if task["name"] == "navigation":
            from drone_playground.environments.scenes.procedural_navigation import make_bank
            from drone_playground.environments.tasks.navigation.rigid_body import NavigationEnv

            per_difficulty = (
                task["reference_count"] if role == "train" else count
            )
            if cfg.get("evaluation", {}).get("protocol"):
                # Keep every fixed scene even when evaluation requests one reset per scene.
                per_difficulty = max(
                    per_difficulty, len(settings["scene"]["scene_ids"])
                )
            if role == "train" and distribution.get("scene"):
                training_scene = instantiate(
                    distribution["scene"], _convert_="all"
                )
                if distribution["type"] == "procedural":
                    raise ValueError(
                        "Brax navigation currently uses fixed/generated banks; procedural updates require a sampling environment"
                    )
                bank, manifest = make_bank(
                    training_scene, ROLE_SEEDS[role], per_difficulty
                )
            else:
                bank, manifest = make_bank(
                    scene, ROLE_SEEDS[role], per_difficulty
                )
            controller = build_controller(
                settings["action"]["controller"], model
            )
            env = NavigationEnv(
                scene_bank=bank,
                task="navigation",
                model=model,
                controller=controller,
                observation=observer,
                objective=objective,
                freq=task["freq"],
                physics_freq=task.get("physics_freq", 500),
                duration=task["duration"],
                goal_radius=task["goal_radius"],
                device=device,
                sensor=sensor,
                reset_randomization=effects.get("reset_randomization"),
                training_collision_mode=(
                    cfg["training"].get(
                        "navigation_collision_mode", "terminate"
                    )
                    if role == "train"
                    else "terminate"
                ),
            )
            env.scene_manifest = manifest
        else:
            from drone_playground.environments.tasks.racing import RacingEnv
            from drone_playground.environments.tasks.tracking.rigid_body import TrackingEnv

            planner = instantiate(task["reference_generator"])
            kwargs = dict(
                task=task["name"],
                model=model,
                controller=build_controller(
                    settings["action"]["controller"], model
                ),
                planner=planner,
                scene=scene,
                observation=observer,
                objective=objective,
                freq=task["freq"],
                device=device,
                reference_seed=ROLE_SEEDS[role],
                reference_count=task["reference_count"]
                if role == "train"
                else count,
            )
            env = (
                RacingEnv(**kwargs)
                if task["name"] == "racing"
                else TrackingEnv(
                    **kwargs,
                    reference=task.get("reference"),
                    duration=task["duration"],
                    numerical_guard=task.get("numerical_guard", False),
                )
            )
    env.time_limit_kind = task["time_limit_kind"]
    env.observation_noise = effects.get("observation_noise") or {}
    env.command_distribution = command
    env.environment_effects = effects
    env.scene_distribution = (
        distribution
        if role == "train"
        else {"type": "fixed", "scene": settings["scene"]}
    )
    from drone_playground.runtime.timing import sensor_schedule

    env.sensor_timing = sensor_schedule(getattr(env, "sensor", None), env.freq)
    if env.sensor_timing is not None:
        env.sensor_calibration = {
            **env.sensor_calibration,
            "schedule": env.sensor_timing,
        }
    if hasattr(env, "model") and hasattr(env.model, "physical_parameters"):
        env.reset_info_fields = (
            *getattr(env, "reset_info_fields", ()),
            "physical_parameters",
        )
    wrapper_effects = copy.deepcopy(effects)
    if getattr(env, "task", None) == "navigation" and hasattr(env, "default"):
        # Navigation owns collision-safe positions and moving-scene phase.
        wrapper_effects["reset_randomization"] = None
    if any(wrapper_effects.values()) and hasattr(env, "default"):
        env = EnvironmentEffects(env, wrapper_effects)
    elif hasattr(env.model, "disturbance"):
        from dataclasses import replace

        env.model = replace(env.model, disturbance=effects.get("disturbance"))
        action = effects.get("action_noise") or {}
        if set(action) - {"std_physical", "bias_physical"}:
            raise ValueError(
                "Acceleration action uncertainty requires physical m/s^2 units"
            )
        import numpy as np

        if any(
            np.asarray(value).shape not in ((), (env.action_size,))
            for value in action.values()
        ):
            raise ValueError(
                "Acceleration action uncertainty must match the three action axes"
            )
    delay_steps = cfg["runtime"].get("action_delay_steps", 0)
    if delay_steps:
        from drone_playground.actions.delay import ActionDelay

        env = ActionDelay(env, delay_steps)
    delay_range = cfg["runtime"].get("action_delay_ms")
    if delay_range is not None and not getattr(
        env, "handles_transport_delay", False
    ):
        from drone_playground.actions.delay import RandomActionDelay

        env = RandomActionDelay(env, delay_range)
    if cfg["method"].get("physical_decoder") is not None:
        from drone_playground.actions.wrappers import GeometricActionWrapper

        env = GeometricActionWrapper(
            env,
            cfg["method"]["physical_decoder"],
            settings["action"]["tracker"],
        )
    env.component_identity = component_identity(cfg)
    if getattr(env, "sim", None) is not None:
        env.sim.component_identity = copy.deepcopy(env.component_identity)
    env.experiment_config = cfg
    env.role = role
    return env


def build_controller(spec, model):
    from hydra.utils import get_class

    from drone_playground.actions.controllers.crazyflow import AttitudeControl

    if "_target_" not in spec:
        return AttitudeControl()
    arguments = (
        {"model": model}
        if _accepts_keyword(get_class(spec["_target_"]), "model")
        else {}
    )
    # Pass the constructed dynamics after Hydra resolves configuration: model
    # objects can be dataclasses and must not be converted into config dicts.
    return instantiate(spec, _convert_="all", _partial_=True)(**arguments)


def build_reference_environment(task, device):
    """Build canonical reference geometry without composing any learning experiment."""
    from omegaconf import OmegaConf

    from drone_playground.configuration import CONFIG_ROOT, load_config

    if task not in ("hovering", "tracking", "racing"):
        raise ValueError(f"No canonical control reference for task: {task}")
    common = OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / "config.yaml"), resolve=False)
    common = {key: value for key, value in common.items() if key not in ("defaults", "hydra")}
    preset = load_config("env/" + task)
    config = OmegaConf.to_container(OmegaConf.merge(common, preset), resolve=True)
    config["algorithm"] = {"name": "none", "gradient": {"transition": "direct"}}
    config["method"] = {"name": "reference", "implementation": "reference", "trainable": False, "output": config["env"]["action"]["command"]}
    config["runtime"]["device"] = device
    return build_environment(config, device, "eval", 1)
