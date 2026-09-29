"""Construct the world, task and execution components of a complete environment."""

from __future__ import annotations

import copy

from hydra.utils import instantiate


def build_environment(config, device="cpu", split="train", count=32):
    from drone_playground.composition import (
        SPLIT_SEEDS,
        build_dynamics,
        build_observer,
        build_sensor,
        component_identity,
    )

    if split not in SPLIT_SEEDS or count < 1:
        raise ValueError("Choose train/dev/heldout and a positive instance count")
    cfg = copy.deepcopy(config)
    settings = cfg["env"]
    task = settings["task"]
    if task["name"] == "depth_navigation":
        from .tasks.depth_navigation import DepthNavigationTask

        env = DepthNavigationTask(cfg)
    elif task["name"] == "pointcloud_navigation":
        from .tasks.pointcloud_navigation import PointCloudNavigationTask

        env = PointCloudNavigationTask(cfg)
    elif task["name"] == "pointcloud_control":
        from .tasks.pointcloud_control import ReferencePointCloudTask

        env = ReferencePointCloudTask(cfg, device=device)
    elif task["name"] == "pointcloud_avoidance":
        from .tasks.pointcloud import PointCloudTask

        env = PointCloudTask(cfg)
    elif settings["execution"]["dynamics"]["forward"].startswith("lotf_"):
        from .tasks.lotf import LOTFTask

        env = LOTFTask(cfg, device=device, split=split)
    else:
        from drone_playground.execution.controllers.crazyflow import AttitudeControl

        model = build_dynamics(cfg)
        scene = instantiate(settings["scene"], _convert_="all")
        sensor = build_sensor(cfg)
        observer = build_observer(cfg, sensor)
        objective = instantiate(cfg["objective"])
        if task["name"] == "navigation":
            from .scenes.navigation import make_bank
            from .tasks.navigation import NavigationEnv

            per_difficulty = task["reference_count"] if split == "train" else count
            bank, manifest = make_bank(scene, SPLIT_SEEDS[split], per_difficulty)
            controller_spec = settings["execution"]["controller"]
            controller = (
                instantiate(controller_spec, _convert_="all")
                if "_target_" in controller_spec
                else AttitudeControl()
            )
            env = NavigationEnv(
                scene_bank=bank,
                task="navigation",
                model=model,
                controller=controller,
                observation=observer,
                objective=objective,
                freq=task["freq"],
                duration=task["duration"],
                goal_radius=task["goal_radius"],
                device=device,
                sensor=sensor,
                training_initialization=(
                    cfg["training"].get("navigation_initialization") if split == "train" else None
                ),
                training_collision_mode=(
                    cfg["training"].get("navigation_collision_mode", "terminate")
                    if split == "train"
                    else "terminate"
                ),
            )
            env.scene_manifest = manifest
        else:
            from .tasks.racing import RacingEnv
            from .tasks.tracking import TrackingEnv

            planner = instantiate(task["reference_generator"])
            kwargs = dict(
                task=task["name"],
                model=model,
                controller=AttitudeControl(),
                planner=planner,
                scene=scene,
                observation=observer,
                objective=objective,
                freq=task["freq"],
                device=device,
                reference_seed=SPLIT_SEEDS[split],
                reference_count=task["reference_count"] if split == "train" else count,
            )
            env = (
                RacingEnv(**kwargs)
                if task["name"] == "racing"
                else TrackingEnv(
                    **kwargs,
                    duration=task["duration"],
                    numerical_guard=task.get("numerical_guard", False),
                )
            )
    delay_steps = cfg["runtime"].get("action_delay_steps", 0)
    if delay_steps:
        from drone_playground.execution.delay import ActionDelay

        env = ActionDelay(env, delay_steps)
    delay_range = cfg["runtime"].get("action_delay_ms")
    if delay_range is not None and task["name"] not in (
        "pointcloud_control",
        "pointcloud_navigation",
        "depth_navigation",
    ):
        from drone_playground.execution.delay import RandomActionDelay

        env = RandomActionDelay(env, delay_range)
    if cfg['method'].get('physical_decoder') is not None:
        from drone_playground.execution.geometric_policy import GeometricPolicyExecution

        env = GeometricPolicyExecution(env,cfg['method']['physical_decoder'],settings['execution']['tracker'])
    env.component_identity = component_identity(cfg)
    if getattr(env, "sim", None) is not None:
        env.sim.component_identity = copy.deepcopy(env.component_identity)
    env.experiment_config = cfg
    return env
