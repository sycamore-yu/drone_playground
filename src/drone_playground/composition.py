"""Assemble one current method/environment recipe and validate its contracts."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
SPLIT_SEEDS = {"train": 10000, "dev": 20000, "heldout": 30000}


def compose_method(
    method: str = "learning/ppo", environment: str | None = None, overrides: list[str] | None = None
) -> dict:
    choices = [f"method={method}"]
    if environment is not None:
        choices.append(f"env={environment}")
    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        roots = global_hydra.config_loader().get_search_path().get_path()
        matching = [
            entry
            for entry in roots
            if entry.provider == "main"
            and Path(entry.path.removeprefix("file://")).resolve() == CONFIG_ROOT
        ]
        if not matching:
            raise ValueError("Active Hydra belongs to a different configuration root")
        context = nullcontext()
    else:
        context = initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT))
    with context:
        cfg = compose(config_name="config", overrides=[*choices, *(overrides or [])])
        result = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return result


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
            execution=env["execution"],
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
    if sensor is None or spec["name"] in ("paper_pointcloud_state", "state_reference"):
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


def build_dynamics(config: dict):
    spec = copy.deepcopy(config["env"]["execution"]["dynamics"])
    gradient = config["algorithm"]["gradient"]
    spec["backward"] = gradient["transition"]
    if "decay_rate" in gradient:
        spec["decay_rate"] = gradient["decay_rate"]
    return instantiate(spec, _convert_="all")


def validate_config(config: dict) -> None:
    if config.get("config_version") != 3:
        raise ValueError("配置版本应为 3；既有产物通过 scripts/tools/migrate_artifact.py 显式迁入")
    for key in ("method", "env", "algorithm", "network", "objective", "training", "runtime"):
        if key not in config:
            raise ValueError(f"Missing component: {key}")
    env, method = config["env"], config["method"]
    for key in ("task", "scene", "sensor", "observation", "execution"):
        if key not in env:
            raise ValueError(f"Missing environment component: {key}")
    mode = config.get("mode", "train")
    expected_backend = "jax" if method["trainable"] else "host"
    if config["runtime"].get("backend") != expected_backend:
        raise ValueError(f"Method {method['name']} requires runtime backend {expected_backend}")
    if mode not in ("train", "eval", "play"):
        raise ValueError(f"Unknown run mode: {mode}")
    if mode == "train" and not method["trainable"]:
        raise ValueError(f"方法 {method['name']} 的配方未定义训练阶段；支持 eval/play")
    if config.get("evaluation", {}).get("protocol"):
        from drone_playground.evaluation.protocols import protocol_identity

        protocol_identity(config)
    execution, task = env["execution"], env["task"]
    from drone_playground.contracts import require_match

    require_match(method["output"], execution["command"])
    controller = execution["controller"]["name"]
    if method["output"] == "trajectory" and not execution.get("tracker"):
        raise ValueError("Trajectory command requires one tracking controller")
    if execution.get("tracker") and execution["tracker"]["name"] not in (
        "trajectory_tracking",
        "attitude_mpc",
        "sampling_mpc",
    ):
        raise ValueError("Unsupported downstream trajectory tracker")
    if method["output"] != "trajectory" and execution.get("tracker") is not None:
        raise ValueError(
            "The command already occupies the tracking stage; duplicate tracker rejected"
        )
    if "backward" in execution["dynamics"]:
        raise ValueError("Derivative selection belongs to algorithm.gradient")
    if "sensor" in env["observation"]:
        raise ValueError("Sensor configuration belongs to env.sensor")
    if task["freq"] <= 0 or task["duration"] <= 0:
        raise ValueError("Task frequency and duration must be positive")
    if config["runtime"].get("timing") != "synchronous":
        raise ValueError("The implemented runtime uses synchronous simulated time")
    delay = config["runtime"].get("action_delay_steps", 0)
    if not isinstance(delay, int) or delay < 0:
        raise ValueError("Action delay must be a nonnegative integer step count")
    random_delay = config["runtime"].get("action_delay_ms")
    if random_delay is not None:
        import math

        if len(random_delay) != 2 or any(not math.isfinite(v) for v in random_delay):
            raise ValueError("Random action delay requires two finite millisecond bounds")
        if random_delay[0] < 0 or random_delay[1] < random_delay[0] or delay:
            raise ValueError("Select one ordered, nonnegative action-delay configuration")
        if task["name"] == "pointcloud_avoidance" or task["name"].startswith("lotf_"):
            raise ValueError(
                "This native paper recipe requires its qualified transport-delay adapter"
            )
    if delay and (task["name"] == "pointcloud_avoidance" or task["name"].startswith("lotf_")):
        raise ValueError(
            "This paper method retains its native timing; added delay needs a qualified recipe"
        )
    if task["name"] in ("pointcloud_navigation", "depth_navigation"):
        from drone_playground.environments.tasks.pointcloud_navigation import (
            validate_navigation_adaptation,
        )

        validate_navigation_adaptation(config)
        return
    if task["name"] == "pointcloud_control":
        if delay:
            raise ValueError("Point-cloud control uses its millisecond command-delay adapter")
        if random_delay is None or random_delay[1] > 1000 / task["freq"]:
            raise ValueError("Point-cloud control delay must fit within one policy tick")
        if config["objective"].get("name") != "pointcloud_control":
            raise ValueError("Control transfer requires its separately named trajectory objective")
        if (
            method["implementation"] != "pointcloud_recurrent"
            or config["algorithm"]["name"] != "pointcloud_bptt"
        ):
            raise ValueError("Point-cloud control transfer requires the recurrent paper policy")
        if task["control_task"] not in ("hovering", "tracking", "racing"):
            raise ValueError("Unknown point-cloud control transfer task")
        if (
            task["freq"] != 10
            or task["physics_freq"] != 500
            or env["sensor"]["source_rate_hz"] != 10
        ):
            raise ValueError(
                "Control transfer uses 10Hz policy/sensing and 500Hz physical integration"
            )
        if env["execution"]["dynamics"]["forward"] != "point_mass_lag":
            raise ValueError("Point-cloud transfer requires the qualified lag dynamics")
        if mode == "train":
            settings = config["training"]
            updates, count, horizon = (
                settings["policy_updates"],
                settings["num_envs"],
                config["algorithm"]["horizon_length"],
            )
            if any(not isinstance(v, int) or v < 1 for v in (updates, count, horizon)):
                raise ValueError("Control transfer requires positive integer training counts")
            if settings.get("num_timesteps") not in (None, updates * count * horizon):
                raise ValueError("Control transfer interaction and update budgets disagree")
            stop = settings.get("stop_after_updates")
            if stop is not None and (not isinstance(stop, int) or not 1 <= stop <= updates):
                raise ValueError("Staged control updates must lie within the declared budget")
            if settings.get("warm_start") and settings.get("resume"):
                raise ValueError("Select warm start or complete continuation, not both")
            clip = config["algorithm"].get("max_grad_norm")
            if clip is not None and (not math.isfinite(clip) or clip <= 0):
                raise ValueError("Gradient clipping requires a positive finite norm")
        return
    if task["name"] == "pointcloud_avoidance":
        from drone_playground.environments.tasks.pointcloud import validate_paper_config

        validate_paper_config(config)
        return
    dynamics = execution["dynamics"]
    forward, backward = dynamics["forward"], config["algorithm"]["gradient"]["transition"]
    algorithm = config["algorithm"]["name"]
    native_lotf = forward in ("lotf_high_fidelity", "lotf_simplified")
    implementation = method["implementation"]
    if implementation in ("attitude_mpc", "sampling_mpc"):
        prediction = method["decision"]
        if prediction["name"] != implementation:
            raise ValueError("MPC implementation and problem adapter differ")
        if implementation == "attitude_mpc" and (
            prediction["prediction"] != "so_rpy"
            or prediction["horizon"] != 25
            or prediction["prediction_seconds"] != 0.5
        ):
            raise ValueError("AttitudeMPC fixes its so_rpy prediction and 25-step/0.5s problem")
        if implementation == "sampling_mpc" and prediction["prediction"] != "so_rpy_rotor_drag":
            raise ValueError("Sampling MPC requires its declared so_rpy_rotor_drag prediction")
        if task["name"] not in ("hovering", "figure8", "random", "racing"):
            raise ValueError("Current MPC methods require a compatible reference-tracking task")
    if task["name"] == "navigation":
        if (
            forward not in ("so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag", "first_principles")
            or backward != "direct"
        ):
            raise ValueError(
                "Navigation requires a supported Crazyflow model and direct derivative"
            )
        if controller not in ("crazyflow_attitude", "velocity_yaw"):
            raise ValueError("Navigation requires the qualified attitude or velocity controller")
        if controller == "velocity_yaw" and (
            method["output"] != "velocity_yaw"
            or implementation != "neural"
            or execution["controller"]["max_speed"] != 20.0
        ):
            raise ValueError(
                "Velocity navigation is an explicit neural velocity/yaw recipe with 20m/s maximum"
            )
        if "families" not in env["scene"] or env["scene"]["dynamic"] != task["dynamic"]:
            raise ValueError("Navigation scene and task disagree on dynamic geometry")
        if (
            task["freq"] != 50
            or task["goal_radius"] != 0.5
            or task["duration"] not in (40.0, 300.0)
        ):
            raise ValueError(
                "Navigation protocols require 50 Hz, 0.5m arrival, and a v1/v2 deadline"
            )
        observation = env["observation"]["name"]
        has_sensor = env["sensor"] is not None and env["sensor"].get("name") != "none"
        if observation not in ("navigation_state", "navigation_depth", "navigation_lidar"):
            raise ValueError(f"Unsupported navigation observation: {observation}")
        if has_sensor != (observation != "navigation_state"):
            raise ValueError("Observation and env.sensor contract differ")
        if implementation in ("native_ego", "native_super", "native_service"):
            sensor_kind = method.get("input_sensor", method.get("method"))
            expected = "navigation_depth" if sensor_kind in ("ego", "depth") else "navigation_lidar"
            if observation != expected:
                raise ValueError("Native planner requires its compatible sensor input")
        if config["objective"].get("name") != "navigation":
            raise ValueError("Navigation objective must use the navigation task events")
    elif native_lotf:
        if (
            env["scene"]["name"] != "lotf_world"
            or env["scene"]["randomization"] != "upstream_initial_state"
        ):
            raise ValueError("LOTF requires its declared native scene")
        if (
            controller != "lotf_betaflight"
            or execution["controller"]["frequency"] != 1000
            or method["output"] != "thrust_bodyrates"
        ):
            raise ValueError(
                "LOTF controller interface requires 1000Hz Betaflight and thrust_bodyrates"
            )
        if task["name"] not in ("lotf_hover", "lotf_tracking") or backward not in (
            "direct",
            "lotf_analytical",
        ):
            raise ValueError("LOTF task or gradient contract differs")
        if task["name"] == "lotf_tracking" and task["freq"] != 50:
            raise ValueError("LOTF reference is sampled at 50Hz; explicit resampling is required")
        if task["name"] == "lotf_hover" and task["reference_generator"]["name"] != "hover":
            raise ValueError("LOTF hovering requires a hover reference")
        if env["observation"]["name"] != "lotf_state" or not env["observation"]["action_history"]:
            raise ValueError("LOTF observation requires state and delayed-action history")
        if env["observation"]["normalization"] != config["network"]["normalize_observations"]:
            raise ValueError("LOTF normalization differs from network contract")
        if config["objective"]["name"] != task["name"]:
            raise ValueError("LOTF objective and task differ")
    else:
        if (
            forward not in ("so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag", "first_principles")
            or backward != "direct"
        ):
            raise ValueError(
                "The selected model requires a supported forward model and direct derivative"
            )
        if task["name"] not in ("hovering", "figure8", "random", "racing"):
            raise ValueError(f"Unsupported task: {task['name']}")
        if env["scene"]["name"] != ("lsy_level0" if task["name"] == "racing" else "empty"):
            raise ValueError("Task requires a compatible scene adapter")
        native_control = implementation in ("native_ego", "native_super", "native_service")
        if native_control and env["observation"]["name"] != "state_reference":
            raise ValueError(
                "Native control tasks require observation@env.observation=state_reference; the raw sensor remains separate"
            )
        required_command = "trajectory" if native_control else "attitude_thrust"
        if controller != "crazyflow_attitude" or method["output"] != required_command:
            raise ValueError("Tracking controller interface requires attitude_thrust")
    if mode == "train":
        settings = config["training"]
        if algorithm == "ppo":
            if not settings.get("num_timesteps") or settings["num_timesteps"] < 1:
                raise ValueError("PPO requires positive training.num_timesteps")
        elif algorithm in ("apg", "bptt", "shac", "dva", "lotf_bptt"):
            horizon = (
                round(task["duration"] * task["freq"])
                if algorithm == "lotf_bptt"
                else config["algorithm"]["horizon_length"]
            )
            if settings["policy_updates"] < 1 or horizon < 1:
                raise ValueError("Derivative training requires positive updates and rollout steps")
            expected = settings["policy_updates"] * settings["num_envs"] * horizon
            if settings.get("num_timesteps") not in (None, expected):
                raise ValueError(
                    f"training.num_timesteps conflicts with update budget ({expected})"
                )
        else:
            raise ValueError(f"Training algorithm unavailable: {algorithm}")
        if native_lotf != (algorithm == "lotf_bptt"):
            raise ValueError("LOTF training requires its native full-episode BPTT contract")
        if algorithm == "lotf_bptt" and (
            config["algorithm"].get("schedule") != "cosine"
            or config["algorithm"].get("rollout", "full_episode") != "full_episode"
        ):
            raise ValueError("LOTF BPTT requires cosine schedule and full-episode rollout")
        if algorithm == "dva" and (task["name"] != "navigation" or not has_sensor):
            raise ValueError("D.VA requires a navigation sensor observation")
        if settings.get("warm_start") and algorithm not in ("ppo", "bptt", "shac"):
            raise ValueError("Parameter warm start requires PPO, BPTT or SHAC")
        if settings.get("resume") and algorithm not in ("lotf_bptt", "bptt", "shac", "dva"):
            raise ValueError("Exact resume requires LOTF/SHAC/D.VA training state")
    if config["training"]["seed"] < 0 or config["training"]["num_envs"] < 1:
        raise ValueError("Seed must be nonnegative and environment count positive")


def build_environment(config: dict, device: str = "cpu", split: str = "train", count: int = 32):
    validate_config(config)
    from drone_playground.environments.environment import build_environment as construct

    return construct(config, device, split, count)


def sensor_layout(config: dict) -> dict | None:
    sensor = build_sensor(config)
    if sensor is None:
        return None
    from drone_playground.networks.encoders import SensorLayout

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
    dynamics = config["env"]["execution"]["dynamics"]
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
    if config.get("mode") == "play" and config.get("replay", {}).get("directory"):
        return _run_experiment(config, root, run_id)
    validate_config(config)
    from drone_playground.runtime.devices import execution_scope

    with execution_scope(config["runtime"]["device"]):
        return _run_experiment(config, root, run_id)


def _run_experiment(config: dict, root: Path, run_id: str):
    if config.get("mode") == "play" and config.get("replay", {}).get("directory"):
        from drone_playground.visualization.viewer import replay

        return replay(
            Path(config["replay"]["directory"]),
            publish=config.get("visualization", {}).get("publish", True),
        )
    validate_config(config)
    if config["mode"] == "train":
        if config["env"]["task"]["name"] == "pointcloud_control":
            from drone_playground.learning.algorithms.pointcloud_control import train

            return train(config, root, run_id)
        implementation = config["method"]["implementation"]
        if implementation in ("pointcloud_recurrent", "depth_recurrent", "lotf_mlp"):
            from importlib import import_module

            trainer = import_module(
                "drone_playground.learning.algorithms." + config["algorithm"]["name"]
            )
            return trainer.train(config, root, run_id)
        from drone_playground.learning.train import train

        return train(
            native_training_config(config),
            root,
            run_id,
            config["runtime"]["device"],
            config["training"].get("warm_start"),
        )
    if config["env"]["task"]["name"] in ("pointcloud_navigation", "depth_navigation"):
        from drone_playground.evaluation.pointcloud_navigation import evaluate

        result = evaluate(config, root, run_id)
    elif config["env"]["task"]["name"] == "pointcloud_control":
        from drone_playground.evaluation.pointcloud_control import evaluate_pointcloud_control

        result = evaluate_pointcloud_control(config, root, run_id)
    elif config["method"]["implementation"] == "pointcloud_recurrent":
        from drone_playground.evaluation.pointcloud import evaluate_pointcloud

        result = evaluate_pointcloud(config, root, run_id)
    else:
        from drone_playground.evaluation.evaluator import evaluate_experiment

        result = evaluate_experiment(config, root, run_id)
    if config["mode"] == "play":
        result["replay_directory"] = str(Path(root) / "experiments" / run_id / "rollouts")
    return result
