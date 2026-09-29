"""Select a downstream tracker without changing a native planner's algorithm."""

from pathlib import Path

import numpy as np
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation

from drone_playground.native.contracts import MotionCommand, Waypoint

from .controllers.trajectory import TrajectoryTracking


class NativeTracking:
    def __init__(self, settings, env, state, directory):
        settings = settings or dict(name="direct")
        self.env, self.name = env, settings["name"]
        self.hold = np.asarray(env.controller_observation(state)["pos"])
        self.fallback = TrajectoryTracking().bind(env.low, env.high)
        self.controller = None
        self.missing = self.short_horizon = 0
        self.consumed = self.clipped = 0
        self.waypoints, self.waypoint_index = None, 0
        if self.name in ("trajectory_tracking", "waypoint_tracking"):
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
        elif self.name != "direct":
            raise ValueError("Unsupported native downstream tracker: " + self.name)

    def command(self, reply, state, tick):
        body = self.env.controller_observation(state)
        default = self.env.default if hasattr(self.env, "default") else self.env.sim.default_data
        mass = float(np.asarray(default.params.mass).reshape(-1)[0])
        sample, curve = reply.get("reference"), reply.get("trajectory")
        output = reply.get("output")
        if output is not None and reply.get("valid_until", tick / self.env.freq) < tick / self.env.freq:
            raise ValueError("Expired physical output reached the execution controller")
        if self.name == "direct" and output is not None:
            if not isinstance(output, MotionCommand) or output.kind != self.env.controller.input_kind:
                raise ValueError("Motion command does not match the configured execution controller")
            command = np.clip(output.values, np.asarray(self.env.low), np.asarray(self.env.high))
            if output.kind == "velocity_yaw":
                command[:3] *= min(1., self.env.controller.max_speed / max(np.linalg.norm(command[:3]), 1e-9))
            self.clipped += int(not np.array_equal(command, output.values))
            self.consumed += 1
            self.hold = np.asarray(body["pos"])
            return command
        if self.name == "waypoint_tracking" and output is not None:
            if not isinstance(output, Waypoint):
                raise ValueError("Waypoint tracker requires ordered physical waypoints")
            if self.waypoints is None or not np.array_equal(output.positions, self.waypoints):
                self.waypoints, self.waypoint_index = output.positions, 0
            while (self.waypoint_index < len(output.positions) - 1 and
                   np.linalg.norm(body["pos"] - output.positions[self.waypoint_index]) <= output.tolerance):
                self.waypoint_index += 1
            sample = dict(position=output.positions[self.waypoint_index], velocity=np.zeros(3),
                          acceleration=np.zeros(3), yaw=Rotation.from_quat(body["quat"]).as_euler("xyz")[2])
        if self.name in ("trajectory_tracking", "waypoint_tracking") and sample is not None:
            self.consumed += 1
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
                self.consumed += 1
                return np.clip(command, np.asarray(self.env.low), np.asarray(self.env.high))
            self.short_horizon += 1
        self.missing += 1
        if self.env.controller.input_kind == "velocity_yaw":
            # Zero world velocity, retaining the current heading, is the declared
            # unavailable-command fallback for this physical interface.
            return np.r_[np.zeros(3), Rotation.from_quat(body["quat"]).as_euler("xyz")[2]]
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
