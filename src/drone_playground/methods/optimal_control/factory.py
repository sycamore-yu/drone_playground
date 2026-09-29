"""Construct a complete controller preset from a resolved experiment."""

import numpy as np
from ml_collections import ConfigDict


def tracking_reference(points, frequency, horizon, periodic=False):
    """Supply a real future horizon through the last physical task tick.

    A periodic task continues its defined period; a finite path holds its final
    position. The task still executes its original number of control decisions.
    """
    points = np.asarray(points)
    indices = np.arange(len(points) + horizon)
    indices = indices % len(points) if periodic else np.minimum(indices, len(points) - 1)
    extended = points[indices]
    velocity = np.gradient(extended, 1.0 / frequency, axis=0)
    return extended, velocity


def build_controller(config, env, state, workdir):
    settings = config["method"]["decision"]
    if settings["name"] == "attitude_mpc":
        from .lsy_mpc import LSYAttitudeMPC

        native_config = getattr(
            env, "config", ConfigDict(dict(sim=dict(drone=env.drone), env=dict(freq=env.freq)))
        )
        ctrl = LSYAttitudeMPC(env.controller_observation(state), {}, native_config, workdir=workdir)
        if config["env"]["task"]["reference_generator"]["name"] != "lsy_course":
            points = np.asarray(env.trajectories[0])
            extended, velocity = tracking_reference(
                points, env.freq, ctrl.native._N, periodic=env.task == "figure8"
            )
            ctrl.native._waypoints_pos = extended
            ctrl.native._waypoints_vel = velocity
            ctrl.native._waypoints_yaw = np.zeros(len(extended))
            ctrl.native._tick_max = len(points) - 1
        delay = settings.get('delay_compensation_ms', 0.)
        if not np.isfinite(delay) or delay < 0:
            raise ValueError('delay_compensation_ms must be finite and nonnegative')
        if delay:
            ctrl.enable_delay_compensation(delay, env.physical_action(env.hover_action))
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
