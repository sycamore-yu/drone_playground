"""Adapters from timed references to the existing MPC implementations."""

import numpy as np
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation

from drone_playground.control.controllers.attitude import AttitudeControl


class AttitudeMPCTracking(AttitudeControl):
    """Track every acados horizon stage from an explicit timed trajectory."""

    input_kind = "trajectory"
    differentiable = False

    def __init__(self, input_kind="trajectory"):
        if input_kind != "trajectory":
            raise ValueError("MPC tracking requires a full trajectory reference")
        self.optimizer = None

    def initialize(self, env, state, directory):
        from drone_playground.control.controllers.mpc.lsy_mpc import LSYAttitudeMPC

        config = ConfigDict(dict(sim=dict(drone=env.drone), env=dict(freq=env.freq)))
        self.optimizer = LSYAttitudeMPC(
            env.controller_observation(state), {}, config, workdir=directory / "acados"
        )
        self.optimizer.reset()
        self.horizon_seconds = self.optimizer.native._T_HORIZON

    def compute_reference(self, body, trajectory, now, tick):
        del tick
        if now < trajectory.start_time or now + self.horizon_seconds > trajectory.end_time + 1e-9:
            return None
        yaw = Rotation.from_quat(body["quat"]).as_euler("xyz")[2]
        return self.optimizer.compute_trajectory(body, trajectory, now, yaw=yaw)

    def close(self):
        if self.optimizer is not None:
            self.optimizer.close()
            self.optimizer = None


class SamplingMPCTracking(AttitudeControl):
    """Supply a live polynomial horizon to the existing sampling controller."""

    input_kind = "trajectory"
    differentiable = False

    def __init__(
        self,
        samples=2000,
        horizon=25,
        prediction_seconds=1.0,
        prediction_device="cpu",
        input_kind="trajectory",
    ):
        if input_kind != "trajectory":
            raise ValueError("MPC tracking requires a full trajectory reference")
        self.samples, self.horizon, self.prediction_seconds = samples, horizon, prediction_seconds
        self.prediction_device, self.optimizer = prediction_device, None

    def initialize(self, env, state, directory):
        from drone_playground.control.controllers.mpc.sampling import SamplingMPC

        del directory
        reference = np.tile(env.controller_observation(state)["pos"], (2, 1))
        self.optimizer = SamplingMPC(
            drone=env.drone,
            reference=reference,
            frequency=env.freq,
            samples=self.samples,
            horizon=self.horizon,
            prediction_seconds=self.prediction_seconds,
            device=self.prediction_device,
        )
        self.optimizer.reset()

    def compute_reference(self, body, trajectory, now, tick):
        if (
            now < trajectory.start_time
            or now + self.prediction_seconds > trajectory.end_time + 1e-9
        ):
            return None
        yaw = Rotation.from_quat(body["quat"]).as_euler("xyz")[2]
        return self.optimizer.compute_control(body, tick, trajectory=trajectory, yaw=yaw)

    def close(self):
        if self.optimizer is not None:
            self.optimizer.close()
            self.optimizer = None
