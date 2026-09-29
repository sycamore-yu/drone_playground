"""Select a downstream tracker without changing a native planner's algorithm."""

from pathlib import Path

import numpy as np
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation

from .controllers.trajectory import TrajectoryTracking


class NativeTracking:
    def __init__(self, settings, env, state, directory):
        self.env, self.name = env, settings["name"]
        self.hold = np.asarray(env.controller_observation(state)["pos"])
        self.fallback = TrajectoryTracking().bind(env.low, env.high)
        self.controller = None
        self.missing = self.short_horizon = 0
        if self.name == "trajectory_tracking":
            self.fallback = TrajectoryTracking(
                **{k: v for k, v in settings.items() if k != "name"}
            ).bind(env.low, env.high)
        elif self.name == "attitude_mpc":
            from drone_playground.methods.optimal_control.lsy_mpc import LSYAttitudeMPC

            self.controller = LSYAttitudeMPC(
                env.controller_observation(state),
                {},
                ConfigDict(dict(sim=dict(drone=env.drone), env=dict(freq=env.freq))),
                workdir=Path(directory) / "acados",
            )
        elif self.name == "sampling_mpc":
            from drone_playground.methods.optimal_control.sampling import SamplingMPC

            self.controller = SamplingMPC(
                drone=env.drone,
                reference=np.stack([self.hold] * 2),
                frequency=env.freq,
                samples=settings["samples"],
                horizon=settings["horizon"],
                prediction_seconds=settings["prediction_seconds"],
                device=settings["prediction_device"],
            )
        else:
            raise ValueError("Unsupported native downstream tracker: " + self.name)

    def command(self, reply, state, tick):
        body = self.env.controller_observation(state)
        default = self.env.default if hasattr(self.env, "default") else self.env.sim.default_data
        mass = float(np.asarray(default.params.mass).reshape(-1)[0])
        sample, curve = reply.get("reference"), reply.get("trajectory")
        if self.name == "trajectory_tracking" and sample is not None:
            self.hold = np.asarray(body["pos"])
            return self.fallback.command(body, sample, mass)
        if self.controller is not None and curve is not None:
            now = tick / self.env.freq
            horizon = (
                self.controller.native._T_HORIZON
                if self.name == "attitude_mpc"
                else self.controller.horizon * self.controller.predict_dt
            )
            if curve.start_time <= now and now + horizon <= curve.end_time:
                yaw = (
                    sample["yaw"]
                    if sample is not None
                    else Rotation.from_quat(body["quat"]).as_euler("xyz")[2]
                )
                if self.name == "attitude_mpc":
                    command = self.controller.compute_trajectory(body, curve, now, yaw=yaw)
                else:
                    command = self.controller.compute_control(body, tick, trajectory=curve, yaw=yaw)
                self.hold = np.asarray(body["pos"])
                return np.clip(command, np.asarray(self.env.low), np.asarray(self.env.high))
            self.short_horizon += 1
        self.missing += 1
        return self.fallback.command(
            body,
            dict(
                position=self.hold,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=Rotation.from_quat(body["quat"]).as_euler("xyz")[2],
            ),
            mass,
        )

    def close(self):
        if self.controller is not None:
            self.controller.close()
