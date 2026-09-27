"""Hydra composition and construction of real flight components.

All commands consume one resolved experiment. Model-native state lives inside
the selected task; consumers use its reset/step and physical observation surface.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
SPLIT_SEEDS = {"train": 10000, "dev": 20000, "heldout": 30000}


def compose_config(experiment: str, overrides: list[str] | None = None) -> dict:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        cfg = compose(
            config_name="config", overrides=[f"experiment={experiment}", *(overrides or [])]
        )
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def observation_spec(config: dict) -> dict:
    """Observation fields without the environment-owned sensor sub-group."""
    return {key: value for key, value in config["observation"].items() if key != "sensor"}


def build_observer(config: dict, sensor=None):
    """Instantiate the observation encoder, taking its frame shape from the sensor.

    The sensor is the authority for frame size and range, so the encoder cannot
    disagree with the calibration that produced the frames.
    """
    spec = observation_spec(config)
    if sensor is None:
        return instantiate(spec)
    name = spec["name"]
    if name == "navigation_depth":
        return replace(
            instantiate(spec),
            history=sensor.history,
            points_per_frame=sensor.points_per_frame,
            channels=sensor.channels,
            near_m=sensor.near_m,
            far_m=sensor.far_m,
        )
    if name == "navigation_lidar":
        return replace(
            instantiate(spec),
            history=sensor.history,
            points_per_frame=sensor.points_per_frame,
            channels=sensor.channels,
            near_m=sensor.range_m[0],
            far_m=sensor.normalise_far_m,
        )
    raise ValueError(f"Unknown perception observation: {name}")


def validate_config(config: dict) -> None:
    for name in (
        "policy",
        "controller",
        "dynamics",
        "task",
        "scene",
        "observation",
        "algorithm",
        "network",
        "objective",
        "training",
    ):
        if name not in config:
            raise ValueError(f"Missing experiment component: {name}")
    forward = config["dynamics"]["forward"]
    backward = config["dynamics"]["backward"]
    control = config["controller"]["name"]
    algorithm = config["algorithm"]["name"]
    mode = config.get("mode", "train")
    frequency = config["task"]["freq"]
    if frequency <= 0 or config["task"]["duration"] <= 0:
        raise ValueError("Task frequency and duration must be positive")
    if control == "attitude_mpc" and (
        config["controller"]["prediction"] != "so_rpy"
        or config["controller"]["horizon"] != 25
        or config["controller"]["prediction_seconds"] != 0.5
    ):
        raise ValueError(
            "The pinned AttitudeMPC preset fixes so_rpy and its 25-step/0.5s optimization problem"
        )
    if control == "sampling_mpc" and config["controller"]["prediction"] != "so_rpy_rotor_drag":
        raise ValueError("The pinned sampling controller predicts with so_rpy_rotor_drag")
    if control == "lotf_betaflight" and config["controller"]["frequency"] != 1000:
        raise ValueError("The LOTF preset uses the native 1000Hz low-level controller")
    native_lotf = forward in ("lotf_high_fidelity", "lotf_simplified")
    policy_output = config["policy"]["output"]
    if config["task"]["name"] == "navigation":
        if forward not in ("so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag", "first_principles"):
            raise ValueError(f"Unknown navigation forward dynamics: {forward}")
        if backward != "direct":
            raise ValueError("Navigation uses the direct Crazyflow derivative")
        if control != "crazyflow_attitude" or policy_output != "attitude_thrust":
            raise ValueError(
                "Navigation uses the shared attitude_thrust controller contract"
            )
        scene = config["scene"]
        if "families" not in scene or "dynamic" not in scene:
            raise ValueError("The navigation task requires a navigation scene preset")
        if bool(scene["dynamic"]) != bool(config["task"]["dynamic"]):
            raise ValueError("Navigation task and scene disagree on static versus dynamic")
        if frequency != 50:
            raise ValueError("The frozen navigation protocol runs the policy at 50 Hz")
        if config["task"]["duration"] != 40.0:
            raise ValueError("The frozen navigation protocol caps an episode at 40 s")
        if config["task"]["goal_radius"] != 0.5:
            raise ValueError("The frozen navigation protocol uses a 0.5 m goal radius")
        if config["objective"]["name"] != "navigation":
            raise ValueError("Every navigation unit shares the one navigation reward")
        observation = config["observation"]
        perception = observation["name"] != "navigation_state"
        has_sensor = bool(observation.get("sensor"))
        if perception and not has_sensor:
            raise ValueError("A perception observation requires an observation.sensor preset")
        if not perception and has_sensor:
            raise ValueError("A state-only observation must not declare a sensor")
        if perception and observation["name"] not in ("navigation_depth", "navigation_lidar"):
            raise ValueError(f"Unknown navigation perception observation: {observation['name']}")
    elif native_lotf:
        if (
            config["scene"]["name"] != "lotf_world"
            or config["scene"]["randomization"] != "upstream_initial_state"
        ):
            raise ValueError(
                "The LOTF preset declares the native world and initial-state randomization"
            )
        if (
            config["observation"]["name"] != "lotf_state"
            or not config["observation"]["action_history"]
        ):
            raise ValueError(
                "The native LOTF policy interface includes state and delayed action history"
            )
        if config["objective"]["name"] != config["task"]["name"]:
            raise ValueError("Native LOTF reward and task must use the same declared task protocol")
        if config["observation"]["normalization"] != config["network"]["normalize_observations"]:
            raise ValueError("LOTF normalization has one owner in the network configuration")
        if control != "lotf_betaflight" or policy_output != "thrust_bodyrates":
            raise ValueError(
                "LOTF controller/command interface requires thrust_bodyrates and lotf_betaflight"
            )
        if backward not in ("lotf_analytical", "direct"):
            raise ValueError(
                "LOTF backward rule must name its analytical surrogate or direct dynamics"
            )
        if config["task"]["name"] not in ("lotf_hover", "lotf_tracking"):
            raise ValueError("LOTF native state requires its compatible task adapter")
        if config["task"]["name"] == "lotf_tracking" and frequency != 50:
            raise ValueError(
                "The native LOTF reference trajectory is sampled at 50Hz; "
                "a different task frequency requires an explicitly resampled trajectory preset"
            )
        if algorithm == "lotf_bptt" and (
            config["algorithm"].get("schedule") != "cosine"
            or config["algorithm"].get("rollout", "full_episode") != "full_episode"
        ):
            raise ValueError(
                "The native LOTF BPTT recipe uses cosine learning rate and a full-episode rollout"
            )
        if (
            config["task"]["name"] == "lotf_hover"
            and config["policy"]["planning"]["name"] != "hover"
        ):
            raise ValueError(
                "The hovering task expects a hover reference; choose the tracking task for CSV trajectories"
            )
    else:
        if forward not in ("so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag", "first_principles"):
            raise ValueError(f"Unknown forward dynamics: {forward}")
        if backward != "direct":
            raise ValueError(
                "A compatible state projection is required for non-native Crazyflow gradients"
            )
        expected_scene = "lsy_level0" if config["task"]["name"] == "racing" else "empty"
        if config["scene"]["name"] != expected_scene:
            raise ValueError(
                f"Task requires the currently implemented scene adapter: {expected_scene}"
            )
        if control not in ("crazyflow_attitude", "attitude_mpc", "sampling_mpc"):
            raise ValueError(f"Incompatible Crazyflow controller: {control}")
        expected = (
            "trajectory" if control in ("attitude_mpc", "sampling_mpc") else "attitude_thrust"
        )
        if policy_output != expected:
            raise ValueError(
                f"Controller command interface requires {expected}, received {policy_output}"
            )
    if mode == "train":
        settings = config["training"]
        if algorithm == "ppo":
            if settings.get("num_timesteps") is None or settings["num_timesteps"] < 1:
                raise ValueError("PPO requires a positive training.num_timesteps budget")
        elif algorithm in ("lotf_bptt", "apg", "shac"):
            if settings["policy_updates"] < 1:
                raise ValueError("Derivative training requires positive policy_updates")
            horizon = (
                int(round(config["task"]["duration"] * frequency))
                if algorithm == "lotf_bptt"
                else config["algorithm"]["horizon_length"]
            )
            if horizon < 1:
                raise ValueError("The training rollout must contain at least one step")
            expected = settings["policy_updates"] * settings["num_envs"] * horizon
            if settings.get("num_timesteps") not in (None, expected):
                raise ValueError(
                    f"training.num_timesteps conflicts with the update budget ({expected}); "
                    "set it to null or to policy_updates * num_envs * rollout_steps"
                )
        if control in ("attitude_mpc", "sampling_mpc"):
            raise ValueError(
                "External optimization controllers lack a declared BPTT gradient; use simulate/evaluate"
            )
        if algorithm == "lotf_bptt" and not native_lotf:
            raise ValueError("LOTF BPTT requires its native environment/normalization contract")
        if native_lotf and algorithm != "lotf_bptt":
            raise ValueError("Select lotf_bptt for the native LOTF training contract")
        if config["training"].get("warm_start") and algorithm != "ppo":
            raise ValueError(
                "Warm-start inference parameters are supported by the native PPO trainer"
            )
        if config["training"].get("resume") and algorithm not in ("lotf_bptt", "shac"):
            raise ValueError("Exact continuation requires a LOTF or SHAC full-state checkpoint")
    if config["training"]["seed"] < 0:
        raise ValueError("Training seed must be nonnegative")
    if config["training"]["num_envs"] < 1:
        raise ValueError("Training requires at least one environment")


def build_environment(config: dict, device: str = "cpu", split: str = "train", count: int = 32):
    validate_config(config)
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split: {split}")
    cfg = copy.deepcopy(config)
    if cfg["task"]["name"] == "navigation":
        from drone_playground.controllers.crazyflow import AttitudeControl
        from drone_playground.tasks.navigation import NavigationEnv
        from drone_playground.tasks.scenes.navigation import make_bank

        model = instantiate(cfg["dynamics"])
        scene = instantiate(cfg["scene"])
        objective = instantiate(cfg["objective"])
        # The optional sensor sub-group belongs to the environment, not to the
        # observation encoder, so it is split out before instantiation.
        sensor = None
        if cfg["observation"].get("sensor"):
            sensor = instantiate(cfg["observation"]["sensor"])
        observer_fields = build_observer(cfg, sensor)
        per_difficulty = cfg["task"]["reference_count"] if split == "train" else count
        bank, manifest = make_bank(scene, SPLIT_SEEDS[split], per_difficulty)
        env = NavigationEnv(
            scene_bank=bank,
            task=cfg["task"]["name"],
            model=model,
            controller=AttitudeControl(),
            observation=observer_fields,
            objective=objective,
            freq=cfg["task"]["freq"],
            duration=cfg["task"]["duration"],
            goal_radius=cfg["task"]["goal_radius"],
            device=device,
            sensor=sensor,
        )
        env.scene_manifest = manifest
    elif cfg["dynamics"]["forward"].startswith("lotf_"):
        from drone_playground.tasks.lotf import LOTFTask

        env = LOTFTask(cfg, device=device, split=split)
    else:
        from drone_playground.controllers.crazyflow import AttitudeControl
        from drone_playground.tasks.racing import RacingEnv
        from drone_playground.tasks.tracking import TrackingEnv

        model = instantiate(cfg["dynamics"])
        scene = instantiate(cfg["scene"])
        planner = instantiate(cfg["policy"]["planning"])
        observer = instantiate(cfg["observation"])
        objective = instantiate(cfg["objective"])
        # The optimization controller owns reference tracking; native attitude
        # conversion inside the task is shared with neural-policy execution.
        controller = AttitudeControl()
        kwargs = dict(
            task=cfg["task"]["name"],
            model=model,
            controller=controller,
            planner=planner,
            scene=scene,
            observation=observer,
            objective=objective,
            freq=cfg["task"]["freq"],
            device=device,
            reference_seed=SPLIT_SEEDS[split],
            reference_count=cfg["task"]["reference_count"] if split == "train" else count,
        )
        if cfg["task"]["name"] == "racing":
            env = RacingEnv(**kwargs)
        else:
            env = TrackingEnv(
                **kwargs,
                duration=cfg["task"]["duration"],
                numerical_guard=cfg["task"]["numerical_guard"],
            )
    env.component_identity = {
        key: copy.deepcopy(cfg[key])
        for key in ("policy", "controller", "dynamics", "task", "scene", "observation", "objective")
    }
    if not cfg["dynamics"]["forward"].startswith("lotf_"):
        env.sim.component_identity = copy.deepcopy(env.component_identity)
    env.experiment_config = cfg
    return env


def native_training_config(config: dict) -> dict:
    """Translate resolved component groups into the native learner's arguments."""
    out = {**config["algorithm"], **config["network"], **config["training"]}
    out.pop("name", None)
    out.update(
        algorithm=config["algorithm"]["name"],
        task=config["task"]["name"],
        dynamics=config["dynamics"]["forward"],
        drone=config["dynamics"]["drone"],
        freq=config["task"]["freq"],
        reference_count=config["task"]["reference_count"],
        numerical_guard=config["task"].get("numerical_guard", False),
        observation_size=instantiate(observation_spec(config)).size,
        components=config,
        config_version=2,
    )
    return out


def run_experiment(config: dict, root: Path, run_id: str):
    validate_config(config)
    mode = config.get("mode", "train")
    if mode == "train":
        if config["algorithm"]["name"] == "lotf_bptt":
            from drone_playground.learning.lotf_bptt import train

            return train(config, root, run_id)
        from drone_playground.learning.train import train

        return train(
            native_training_config(config),
            root,
            run_id,
            config["training"]["device"],
            config["training"].get("warm_start"),
        )
    if mode in ("simulate", "evaluate"):
        from drone_playground.evaluation.execution import evaluate_experiment

        return evaluate_experiment(config, root, run_id)
    if mode == "replay":
        from drone_playground.runs.rscope_io import publish_run

        return {"active_directory": str(publish_run(Path(config["replay"]["directory"])))}
    raise ValueError(f"Unknown experiment mode: {mode}")
