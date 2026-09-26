"""Construct a complete controller preset from a resolved experiment."""

import numpy as np
from ml_collections import ConfigDict


def build_controller(config, env, state, workdir):
    settings = config["controller"]
    if settings["name"] == "attitude_mpc":
        from .lsy_mpc import LSYAttitudeMPC

        native_config = getattr(
            env, "config", ConfigDict(dict(sim=dict(drone=env.drone), env=dict(freq=env.freq)))
        )
        ctrl = LSYAttitudeMPC(env.controller_observation(state), {}, native_config, workdir=workdir)
        if config["policy"]["planning"]["name"] != "lsy_course":
            points = np.asarray(env.trajectories[0])
            ctrl.native._waypoints_pos = points
            ctrl.native._waypoints_vel = np.gradient(points, env.dt, axis=0)
            ctrl.native._waypoints_yaw = np.zeros(len(points))
            ctrl.native._tick_max = len(points) - 1 - ctrl.native._N
        return ctrl
    if settings["name"] == "sampling_mpc":
        from .sampling import SamplingMPC

        obstacles = (
            np.asarray(env.default.obstacles_pos[0])
            if hasattr(env.default, "obstacles_pos")
            else None
        )
        return SamplingMPC(
            drone=env.drone,
            reference=np.asarray(env.trajectories[0]),
            frequency=env.freq,
            samples=settings["samples"],
            horizon=settings["horizon"],
            prediction_seconds=settings["prediction_seconds"],
            device=settings["prediction_device"],
            obstacles=obstacles,
        )
    raise ValueError(f"No optimization controller factory for {settings['name']}")
