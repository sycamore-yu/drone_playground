"""A finite future reference for downstream control, with explicit missing yaw."""

import numpy as np

from drone_playground.native.contracts import Trajectory


def reference_horizon(trajectory, time, offsets, *, yaw=None):
    if not isinstance(trajectory, Trajectory):
        raise TypeError("A future horizon requires a full Trajectory")
    result = trajectory.sample_many(time + np.asarray(offsets))
    if not trajectory.yaw_defined:
        if yaw is None or not np.isfinite(yaw):
            raise ValueError("Position-only trajectory requires an explicit downstream yaw")
        result["yaw"] = np.full(len(result["time"]), yaw)
    return result
