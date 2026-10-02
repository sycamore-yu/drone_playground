"""Assemble one current method/environment recipe and validate its contracts."""

from __future__ import annotations

import copy
from pathlib import Path

from hydra.utils import get_class
from omegaconf import OmegaConf

from drone_playground.environments.environment import build_observer, build_sensor

from drone_playground.configuration import CONFIG_ROOT, load_config


def compose_experiment(
    experiment: str = "control/ppo",
    environment: str | None = None,
    overrides: list[str] | None = None,
) -> dict:
    choices = [f"experiment={experiment}"]
    if environment is not None:
        choices.append(f"env={environment}")
    result = load_config(overrides=[*choices, *(overrides or [])])
    for group in ("env", "algorithm", "network", "training", "evaluation"):
        if not isinstance(result.get(group), dict):
            raise ValueError(
                f"{group} must resolve to a mapping; select current component fields or an experiment recipe"
            )
    from drone_playground.benchmarks import resolve_protocol_settings

    resolve_protocol_settings(result)
    return result




def experiment_conditions(config: dict) -> dict:
    """The actual configured conditions needed to interpret a method comparison."""
    config = config.get("components", config)
    from drone_playground.actions.commands import COMMANDS

    env, method = config["env"], config["method"]
    study = config.get("study", {"type": "method_reproduction"})
    if study["type"] not in ("method_reproduction", "controlled_comparison"):
        raise ValueError(
            "study.type must be method_reproduction or controlled_comparison"
        )
    if study["type"] == "controlled_comparison" and not study.get(
        "conditions_id"
    ):
        raise ValueError(
            "Controlled comparison requires a conditions_id identifying the fixed conditions"
        )
    return copy.deepcopy(
        dict(
            study=study,
            scene=env["scene"],
            dynamics=env["dynamics"],
            observation=env["observation"],
            sensor=env["sensor"],
            action_interface=dict(
                command=env["action"]["command"],
                units=COMMANDS[env["action"]["command"]].units,
                frame=COMMANDS[env["action"]["command"]].frame,
            ),
            controller=env["action"]["controller"],
            tracker=env["action"].get("tracker"),
            control_frequency_hz=env["task"]["freq"],
            physics_frequency_hz=env["task"].get("physics_freq", 500),
            latency=dict(
                action_delay_steps=config["runtime"].get(
                    "action_delay_steps", 0
                ),
                action_delay_ms=config["runtime"].get("action_delay_ms"),
                native_delay_s=env["task"].get("delay"),
            ),
            safety_margin=dict(
                body_radius_m=env["task"].get("body_radius"),
                clearance_margin_m=config["objective"].get("clearance_margin"),
            ),
            method=method,
        )
    )












def validate_config(config: dict) -> None:
    if config.get("config_version") != 3:
        raise ValueError("配置版本应为 3；不支持旧版本产物")
    for key in (
        "method",
        "env",
        "algorithm",
        "network",
        "objective",
        "training",
        "runtime",
    ):
        if key not in config:
            raise ValueError(f"Missing component: {key}")
    env, method = config["env"], config["method"]
    for key in ("task", "scene", "sensor", "observation", "action", "dynamics"):
        if key not in env:
            raise ValueError(f"Missing environment component: {key}")
    mode = config.get("mode", "train")
    expected_backend = "jax" if method["trainable"] else "host"
    if config["runtime"].get("backend") != expected_backend:
        raise ValueError(
            f"Method {method['name']} requires runtime backend {expected_backend}"
        )
    if mode not in ("train", "eval", "play"):
        raise ValueError(f"Unknown run mode: {mode}")
    if mode == "train" and not method["trainable"]:
        raise ValueError(
            f"方法 {method['name']} 的配方未定义训练阶段；支持 eval/play"
        )
    if config.get("evaluation", {}).get("protocol"):
        from drone_playground.benchmarks import (
            protocol_identity,
            resolve_protocol_settings,
        )

        resolve_protocol_settings(config)
        protocol_identity(config)
    action, task = env["action"], env["task"]
    if "frequency_hz" in action:
        raise ValueError(
            "action.frequency_hz is invalid; env.task.freq owns the decision clock"
        )
    from drone_playground.actions.commands import require_match

    require_match(method["output"], action["command"])
    controller = action["controller"]["name"]
    if method["output"] in ("trajectory", "waypoint") and not action.get(
        "tracker"
    ):
        raise ValueError(
            "Trajectory/Waypoint command requires one tracking controller"
        )
    if action.get("tracker") and action["tracker"]["name"] not in (
        "trajectory_tracking",
        "attitude_mpc",
        "sampling_mpc",
        "waypoint_tracking",
        "jax_trajectory_tracking",
        "jax_waypoint_tracking",
    ):
        raise ValueError("Unsupported downstream trajectory tracker")
    if action.get("tracker") and (
        (method["output"] == "waypoint")
        != (
            action["tracker"]["name"]
            in ("waypoint_tracking", "jax_waypoint_tracking")
        )
    ):
        raise ValueError("Waypoint/Trajectory tracker input contract differs")
    if (
        method["output"] not in ("trajectory", "waypoint")
        and action.get("tracker") is not None
    ):
        raise ValueError(
            "The command already occupies the tracking stage; duplicate tracker rejected"
        )
    geometric = method.get("physical_decoder")
    if geometric is not None:
        from drone_playground.actions.controllers.trajectory_jax import JaxTrajectoryTracking
        from drone_playground.actions.decoders import PhysicalActionDecoder

        decoder = PhysicalActionDecoder(**geometric)
        if (
            method["implementation"] != "neural"
            or not method["trainable"]
            or config["algorithm"]["name"] not in ("ppo", "bptt", "shac")
            or task["name"] not in ("hovering", "tracking", "racing")
            or decoder.kind != method["output"]
            or method.get("stages")
        ):
            raise ValueError(
                "JAX geometric policies require a control task, neural head and PPO/BPTT/SHAC"
            )
        if (
            method.get("goal_source") != "observation_reference"
            or env["observation"]["name"] != "state_reference"
            or env["observation"].get("n_samples", 0) < 1
        ):
            raise ValueError(
                "JAX geometric goal must come from the declared observation_reference"
            )
        tracker = action.get("tracker") or {}
        if tracker.get("name") != "jax_" + decoder.kind + "_tracking":
            raise ValueError(
                "Geometric policy training requires an explicitly differentiable JAX tracker; RPC/MPC gradients are unavailable"
            )
        tracking = JaxTrajectoryTracking(
            **{k: v for k, v in tracker.items() if k != "name"}
        )
        if decoder.kind == "waypoint" and decoder.count != 1:
            raise ValueError("JAX waypoint PD requires one target per decision")
        if (
            decoder.kind == "trajectory"
            and not 0 < tracking.lead_seconds <= decoder.horizon_seconds
        ):
            raise ValueError(
                "Trajectory tracking lead must be positive and within its horizon"
            )
    if "backward" in env["dynamics"]:
        raise ValueError("Derivative selection belongs to algorithm.gradient")
    if "sensor" in env["observation"]:
        raise ValueError("Sensor configuration belongs to env.sensor")
    if task["freq"] <= 0 or task["duration"] <= 0:
        raise ValueError("Task frequency and duration must be positive")
    physics_freq = task.get("physics_freq", 500)
    if physics_freq <= 0 or physics_freq % task["freq"]:
        raise ValueError("Task frequency must divide physics frequency")
    if config["runtime"].get("timing") != "synchronous":
        raise ValueError(
            "The implemented runtime uses synchronous simulated time"
        )
    delay = config["runtime"].get("action_delay_steps", 0)
    if not isinstance(delay, int) or delay < 0:
        raise ValueError(
            "Action delay must be a nonnegative integer step count"
        )
    random_delay = config["runtime"].get("action_delay_ms")
    if random_delay is not None:
        import math

        if len(random_delay) != 2 or any(
            not math.isfinite(v) for v in random_delay
        ):
            raise ValueError(
                "Random action delay requires two finite millisecond bounds"
            )
        if random_delay[0] < 0 or random_delay[1] < random_delay[0] or delay:
            raise ValueError(
                "Select one ordered, nonnegative action-delay configuration"
            )
        if task.get("adapter") == "recurrent_flight":
            raise ValueError(
                "This native paper recipe requires its qualified transport-delay adapter"
            )
    if delay and (task.get("adapter") == "recurrent_flight"):
        raise ValueError(
            "This paper method retains its native timing; added delay needs a qualified recipe"
        )
    if task.get("adapter") == "recurrent_acceleration":
        from drone_playground.learning.configuration import (
            validate_navigation_adaptation,
        )

        validate_navigation_adaptation(config)
        return
    if task.get("adapter") == "recurrent_reference":
        if delay:
            raise ValueError(
                "Point-cloud control uses its millisecond command-delay adapter"
            )
        if random_delay is None or random_delay[1] > 1000 / task["freq"]:
            raise ValueError(
                "Point-cloud control delay must fit within one policy tick"
            )
        if config["objective"].get("name") != "reference_tracking":
            raise ValueError(
                "Control transfer requires its separately named trajectory objective"
            )
        if (
            method["implementation"] != "pointcloud_recurrent"
            or config["algorithm"]["name"] != "recurrent_bptt"
        ):
            raise ValueError(
                "Point-cloud control transfer requires the recurrent paper policy"
            )
        if task["name"] not in ("hovering", "tracking", "racing"):
            raise ValueError("Unknown point-cloud control transfer task")
        if (
            task["freq"] != 10
            or task["physics_freq"] != 500
            or env["sensor"]["source_rate_hz"] != 10
        ):
            raise ValueError(
                "Control transfer uses 10Hz policy/sensing and 500Hz physical integration"
            )
        if env["dynamics"]["forward"] != "point_mass_lag":
            raise ValueError(
                "Point-cloud transfer requires the qualified lag dynamics"
            )
        if mode == "train":
            settings = config["training"]
            updates, count, horizon = (
                settings["policy_updates"],
                settings["num_envs"],
                config["algorithm"]["horizon_length"],
            )
            if any(
                not isinstance(v, int) or v < 1
                for v in (updates, count, horizon)
            ):
                raise ValueError(
                    "Control transfer requires positive integer training counts"
                )
            if settings.get("num_timesteps") not in (
                None,
                updates * count * horizon,
            ):
                raise ValueError(
                    "Control transfer interaction and update budgets disagree"
                )
            stop = settings.get("stop_after_updates")
            if stop is not None and (
                not isinstance(stop, int) or not 1 <= stop <= updates
            ):
                raise ValueError(
                    "Staged control updates must lie within the declared budget"
                )
            if settings.get("warm_start") and settings.get("resume"):
                raise ValueError(
                    "Select warm start or complete continuation, not both"
                )
            clip = config["algorithm"].get("max_grad_norm")
            if clip is not None and (not math.isfinite(clip) or clip <= 0):
                raise ValueError(
                    "Gradient clipping requires a positive finite norm"
                )
        return
    if task.get("adapter") == "recurrent_flight":
        from drone_playground.learning.configuration import (
            validate_reconstruction_config,
        )

        validate_reconstruction_config(config)
        return
    dynamics = env["dynamics"]
    forward, backward = (
        dynamics["forward"],
        config["algorithm"]["gradient"]["transition"],
    )
    algorithm = config["algorithm"]["name"]
    implementation = method["implementation"]
    bodyrates = action["command"] == "thrust_bodyrates"
    if bodyrates != (controller == "bodyrates"):
        raise ValueError(
            "controller and thrust/body-rate command interface differ"
        )
    if forward.startswith("lotf_") != bodyrates:
        raise ValueError(
            "Selected dynamics and controller command interface differ"
        )
    if implementation in ("attitude_mpc", "sampling_mpc"):
        prediction = method["decision"]
        if prediction["name"] != implementation:
            raise ValueError("MPC implementation and problem adapter differ")
        if implementation == "attitude_mpc" and (
            prediction["prediction"] != "so_rpy"
            or prediction["horizon"] != 25
            or prediction["prediction_seconds"] != 0.5
        ):
            raise ValueError(
                "AttitudeMPC fixes its so_rpy prediction and 25-step/0.5s problem"
            )
        if (
            implementation == "sampling_mpc"
            and prediction["prediction"] != "so_rpy_rotor_drag"
        ):
            raise ValueError(
                "Sampling MPC requires its declared so_rpy_rotor_drag prediction"
            )
        if task["name"] not in ("hovering", "tracking", "racing"):
            raise ValueError(
                "Current MPC methods require a compatible reference-tracking task"
            )
    if task["name"] == "navigation":
        if forward not in (
            "so_rpy",
            "so_rpy_rotor",
            "so_rpy_rotor_drag",
            "first_principles",
            "lotf_high_fidelity",
            "lotf_simplified",
        ) or backward not in ("direct", "analytical_surrogate"):
            raise ValueError(
                "Navigation requires a supported Crazyflow model and direct derivative"
            )
        if controller not in (
            "crazyflow_attitude",
            "velocity_yaw",
            "bodyrates",
        ):
            raise ValueError(
                "Navigation requires the qualified attitude or velocity controller"
            )
        if controller == "velocity_yaw" and (
            method["output"] != "velocity_yaw"
            or implementation not in ("neural", "native_service", "pipeline")
            or action["controller"]["max_speed"] <= 0
        ):
            raise ValueError(
                "Velocity navigation requires a compatible velocity/yaw method and positive speed limit"
            )
        if (
            "families" not in env["scene"]
            or env["scene"]["dynamic"] != task["dynamic"]
        ):
            raise ValueError(
                "Navigation scene and task disagree on dynamic geometry"
            )
        if task["goal_radius"] <= 0:
            raise ValueError("Navigation goal_radius must be positive")
        observation = env["observation"]["name"]
        has_sensor = (
            env["sensor"] is not None and env["sensor"].get("name") != "none"
        )
        if observation not in (
            "navigation_state",
            "navigation_depth",
            "navigation_lidar",
        ):
            raise ValueError(
                f"Unsupported navigation observation: {observation}"
            )
        if has_sensor != (observation != "navigation_state"):
            raise ValueError("Observation and env.sensor contract differ")
        if implementation in (
            "native_ego",
            "native_super",
            "native_service",
            "pipeline",
        ):
            sensor_kind = method.get("input_sensor", method.get("method"))
            expected = {
                "ego": "navigation_depth",
                "depth": "navigation_depth",
                "point_cloud": "navigation_lidar",
                "super": "navigation_lidar",
                "none": "navigation_state",
            }.get(sensor_kind)
            if observation != expected:
                raise ValueError(
                    "Native planner requires its compatible sensor input"
                )
        if config["objective"].get("name") != "navigation":
            raise ValueError(
                "Navigation objective must use the navigation task events"
            )
    else:
        if forward not in (
            "so_rpy",
            "so_rpy_rotor",
            "so_rpy_rotor_drag",
            "first_principles",
            "lotf_high_fidelity",
            "lotf_simplified",
        ) or backward not in ("direct", "analytical_surrogate"):
            raise ValueError(
                "The selected model requires a supported forward model and direct derivative"
            )
        if task["name"] not in ("hovering", "tracking", "racing"):
            raise ValueError(f"Unsupported task: {task['name']}")
        if env["scene"]["name"] != (
            "lsy_level0" if task["name"] == "racing" else "empty"
        ):
            raise ValueError("Task requires a compatible scene adapter")
        native_control = implementation in (
            "native_ego",
            "native_super",
            "native_service",
            "pipeline",
        )
        if native_control and env["observation"]["name"] not in (
            "state",
            "state_reference",
        ):
            raise ValueError(
                "Native control tasks require state or state_reference observations; the raw sensor remains separate"
            )
        required_commands = (
            ("trajectory", "waypoint", "attitude_thrust")
            if implementation in ("native_service", "pipeline")
            or geometric is not None
            else (
                ("trajectory",)
                if native_control
                else ("attitude_thrust", "thrust_bodyrates")
            )
        )
        if (
            controller not in ("crazyflow_attitude", "bodyrates")
            or method["output"] not in required_commands
        ):
            raise ValueError(
                "Tracking controller interface requires attitude_thrust"
            )
    if mode == "train":
        settings = config["training"]
        if algorithm == "ppo":
            if (
                not settings.get("num_timesteps")
                or settings["num_timesteps"] < 1
            ):
                raise ValueError("PPO requires positive training.num_timesteps")
        elif algorithm in ("apg", "bptt", "shac", "dva"):
            horizon = config["algorithm"]["horizon_length"]
            if settings["policy_updates"] < 1 or horizon < 1:
                raise ValueError(
                    "Derivative training requires positive updates and rollout steps"
                )
            expected = (
                settings["policy_updates"] * settings["num_envs"] * horizon
            )
            if settings.get("num_timesteps") not in (None, expected):
                raise ValueError(
                    f"training.num_timesteps conflicts with update budget ({expected})"
                )
        else:
            raise ValueError(f"Training algorithm unavailable: {algorithm}")
        if algorithm == "dva" and (
            task["name"] != "navigation" or not has_sensor
        ):
            raise ValueError("D.VA requires a navigation sensor observation")
        if settings.get("warm_start") and algorithm not in (
            "ppo",
            "bptt",
            "shac",
        ):
            raise ValueError("Parameter warm start requires PPO, BPTT or SHAC")
        if settings.get("resume") and algorithm not in ("bptt", "shac", "dva"):
            raise ValueError(
                "Exact resume requires BPTT/SHAC/D.VA training state"
            )
    if config["training"]["seed"] < 0 or config["training"]["num_envs"] < 1:
        raise ValueError(
            "Seed must be nonnegative and environment count positive"
        )


def build_environment(
    config: dict, device: str = "cpu", role: str = "train", count: int = 32
):
    validate_config(config)
    from drone_playground.environments.environment import build_environment as construct

    return construct(config, device, role, count)


def sensor_layout(config: dict) -> dict | None:
    sensor = build_sensor(config)
    if sensor is None:
        return None
    from drone_playground.networks.perception import SensorLayout

    observer = build_observer(config, sensor)
    grid = (
        (sensor.width // sensor.stride, sensor.height // sensor.stride)
        if observer.name == "navigation_depth"
        else None
    )
    return SensorLayout.from_observation(observer, grid=grid).as_dict()


def native_training_config(config: dict) -> dict:
    """One-way lowering into the selected Brax learner's argument vocabulary."""
    task = config["env"]["task"]
    dynamics = config["env"]["dynamics"]
    out = {**config["algorithm"], **config["network"], **config["training"]}
    out.pop("name", None)
    out.update(
        algorithm=config["algorithm"]["name"],
        task=task["name"],
        dynamics=dynamics["forward"],
        drone=dynamics["drone"],
        freq=task["freq"],
        reference_count=task.get("reference_count", 1),
        numerical_guard=task.get("numerical_guard", False),
        device=config["runtime"]["device"],
        observation_size=build_observer(config, build_sensor(config)).size,
        sensor_layout=sensor_layout(config),
        components=config,
        config_version=3,
    )
    return out


def run_experiment(config: dict, root: Path, run_id: str):
    from drone_playground.artifacts.layout import resolve_artifact

    config = copy.deepcopy(config)
    if config.get("checkpoint"):
        config["checkpoint"] = str(resolve_artifact(config["checkpoint"]))
    for field in ("resume", "warm_start"):
        if config.get("training", {}).get(field):
            config["training"][field] = str(
                resolve_artifact(config["training"][field])
            )
    if config.get("evaluation", {}).get("training_run"):
        config["evaluation"]["training_run"] = str(
            resolve_artifact(config["evaluation"]["training_run"])
        )
    if config.get("replay", {}).get("directory"):
        config["replay"]["directory"] = str(
            resolve_artifact(config["replay"]["directory"])
        )
    if config.get("mode") == "play" and config.get("replay", {}).get(
        "directory"
    ):
        return _run_experiment(config, root, run_id)
    validate_config(config)
    from drone_playground.runtime.devices import execution_scope

    with execution_scope(config["runtime"]["device"]):
        return _run_experiment(config, root, run_id)


def _run_experiment(config: dict, root: Path, run_id: str):
    if config.get("mode") == "play" and config.get("replay", {}).get(
        "directory"
    ):
        from drone_playground.visualization.viewer import replay

        return replay(
            Path(config["replay"]["directory"]),
            publish=config.get("visualization", {}).get("publish", True),
        )
    validate_config(config)
    from hydra.utils import get_method

    task = config["env"]["task"]
    if config["mode"] == "train":
        trainer = get_method(
            task.get("trainer", config["algorithm"]["trainer"])
        )
        return trainer(config, root, run_id)
    evaluator = get_method(
        task.get(
            "evaluation_entrypoint",
            "drone_playground.evaluation.run.evaluate_experiment",
        )
    )
    if config["mode"] == "play":
        config["evaluation"]["record_replays"] = True
    result = evaluator(config, root, run_id)
    if config["mode"] == "play":
        from drone_playground.artifacts.layout import experiment_directory

        result["replay_directory"] = str(
            experiment_directory(root, run_id) / "rollouts"
        )
    return result
