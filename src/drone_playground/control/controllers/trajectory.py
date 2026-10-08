"""Position/velocity/acceleration reference tracking into the shared attitude loop."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.control.controllers.attitude import AttitudeControl


class TrajectoryTracking(AttitudeControl):
    name = "trajectory_tracking"
    input_kind = "trajectory"
    differentiable = False

    def __init__(
        self,
        kp=4.0,
        kv=3.0,
        max_acceleration=5.0,
        input_kind="trajectory",
        reference_source="execution_sample",
    ):
        if input_kind not in ("trajectory", "waypoint"):
            raise ValueError("TrajectoryTracking accepts trajectory or waypoint")
        if reference_source not in ("execution_sample", "trajectory"):
            raise ValueError("Unknown trajectory reference source")
        self.input_kind, self.reference_source = input_kind, reference_source
        self.kp, self.kv, self.max_acceleration = kp, kv, max_acceleration

    def command(self, state, reference, mass):
        """Convert a finite world-frame reference to physical attitude and newtons."""
        pos, vel, quat = (np.asarray(state[k], dtype=float) for k in ("pos", "vel", "quat"))
        target = np.asarray(reference["position"], dtype=float)
        velocity = np.asarray(reference["velocity"], dtype=float)
        acceleration = np.asarray(reference["acceleration"], dtype=float)
        yaw = float(reference["yaw"])
        if not np.isfinite(np.r_[target, velocity, acceleration, yaw]).all():
            raise ValueError("Non-finite trajectory reference")
        acc = acceleration + self.kp * (target - pos) + self.kv * (velocity - vel)
        acc *= min(1.0, self.max_acceleration / max(np.linalg.norm(acc), 1e-9))
        force = acc + np.array([0.0, 0.0, 9.81])
        z = force / np.linalg.norm(force)
        y = np.cross(z, [np.cos(yaw), np.sin(yaw), 0.0])
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        angles = Rotation.from_matrix(np.column_stack([x, y, z])).as_euler("xyz")
        current_z = Rotation.from_quat(quat).as_matrix()[:, 2]
        thrust = mass * np.dot(force, current_z)
        return np.clip(np.r_[angles, thrust], np.asarray(self.low), np.asarray(self.high))
