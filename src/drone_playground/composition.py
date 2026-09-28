"""Assemble one current method/environment recipe and validate its contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from hydra import compose, initialize_config_dir
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
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
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
    if sensor is None or spec["name"] == "paper_pointcloud_state":
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
    if delay and (task["name"] == "pointcloud_avoidance" or task["name"].startswith("lotf_")):
        raise ValueError(
            "This paper method retains its native timing; added delay needs a qualified recipe"
        )
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
        if controller != "crazyflow_attitude":
            raise ValueError("Navigation controller interface requires crazyflow_attitude")
        if "families" not in env["scene"] or env["scene"]["dynamic"] != task["dynamic"]:
            raise ValueError("Navigation scene and task disagree on dynamic geometry")
        if task["freq"] != 50 or task["duration"] != 40.0 or task["goal_radius"] != 0.5:
            raise ValueError("Navigation protocol fixes 50 Hz, 40 s and 0.5 m goal radius")
        observation = env["observation"]["name"]
        has_sensor = env["sensor"] is not None and env["sensor"].get("name") != "none"
        if observation not in ("navigation_state", "navigation_depth", "navigation_lidar"):
            raise ValueError(f"Unsupported navigation observation: {observation}")
        if has_sensor != (observation != "navigation_state"):
            raise ValueError("Observation and env.sensor contract differ")
        if implementation in ("native_ego", "native_super"):
            expected = "navigation_depth" if method["method"] == "ego" else "navigation_lidar"
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
        if controller != "crazyflow_attitude" or method["output"] != "attitude_thrust":
            raise ValueError("Tracking controller interface requires attitude_thrust")
    if mode == "train":
        settings = config["training"]
        if algorithm == "ppo":
            if not settings.get("num_timesteps") or settings["num_timesteps"] < 1:
                raise ValueError("PPO requires positive training.num_timesteps")
        elif algorithm in ("apg", "shac", "dva", "lotf_bptt"):
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
        if settings.get("warm_start") and algorithm != "ppo":
            raise ValueError("Parameter warm start is implemented for PPO")
        if settings.get("resume") and algorithm not in ("lotf_bptt", "shac", "dva"):
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
        implementation = config["method"]["implementation"]
        if implementation in ("pointcloud_recurrent", "lotf_mlp"):
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
    if config["method"]["implementation"] == "pointcloud_recurrent":
        from drone_playground.evaluation.pointcloud import evaluate_pointcloud

        result = evaluate_pointcloud(config, root, run_id)
    else:
        from drone_playground.evaluation.evaluator import evaluate_experiment

        result = evaluate_experiment(config, root, run_id)
    if config["mode"] == "play":
        result["replay_directory"] = str(Path(root) / "experiments" / run_id / "rollouts")
    return result
