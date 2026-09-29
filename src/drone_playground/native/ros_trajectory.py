"""Lossless adapters for pinned ROS trajectory messages; no ROS imports here."""

from math import comb

import numpy as np

from .contracts import Trajectory


def ego_trajectory(message):
    curve = Trajectory.from_bspline(
        message.start_time.to_sec() - 1.0,
        message.knots,
        [[p.x, p.y, p.z] for p in message.pos_pts],
        message.order,
    )
    # Upstream traj_server calculates yaw causally; yaw_pts is not consumed.
    return Trajectory(curve.start_time, curve.durations, curve.coefficients, yaw_defined=False)


def super_trajectory(message):
    n, k = message.piece_num_pos, message.order_pos + 1
    if n < 1 or not 1 <= k <= 16:
        raise ValueError("SUPER position trajectory has invalid shape")
    coefficients = np.zeros((n, 4, k))
    for axis, name in enumerate(("coef_pos_x", "coef_pos_y", "coef_pos_z")):
        coefficients[:, axis] = np.asarray(getattr(message, name)).reshape(n, k)[:, ::-1]
    start = message.start_WT_pos.to_sec() - 1.0
    position = Trajectory(start, message.time_pos, coefficients, yaw_defined=False)
    if not message.type & message.YAW_TRAJ:
        return position
    ny, ky = message.piece_num_yaw, message.order_yaw + 1
    yaw_coefficients = np.zeros((ny, 4, ky))
    yaw_coefficients[:, 3] = np.asarray(message.coef_yaw).reshape(ny, ky)[:, ::-1]
    yaw = Trajectory(message.start_WT_yaw.to_sec() - 1.0, message.time_yaw, yaw_coefficients)
    # Different knot partitions are split analytically at their union. Polynomial
    # shifting uses the binomial theorem, so derivatives and future values survive.
    left, right = max(position.start_time, yaw.start_time), min(position.end_time, yaw.end_time)
    if right <= left:
        raise ValueError("SUPER position/yaw time intervals do not overlap")
    edges = [p.start_time + np.r_[0.0, np.cumsum(p.durations)] for p in (position, yaw)]
    breaks = np.unique(np.concatenate(([left, right], *edges)))
    breaks = breaks[(breaks >= left) & (breaks <= right)]
    result = np.zeros((len(breaks) - 1, 4, max(k, ky)))
    for curve, knots, axes in ((position, edges[0], range(3)), (yaw, edges[1], (3,))):
        for i, t in enumerate(breaks[:-1]):
            j = min(np.searchsorted(knots[1:], t, side="right"), len(curve.durations) - 1)
            delta = t - knots[j]
            for power in range(curve.coefficients.shape[-1]):
                for shifted in range(power + 1):
                    for axis in axes:
                        result[i, axis, shifted] += (
                            curve.coefficients[j, axis, power]
                            * comb(power, shifted)
                            * delta ** (power - shifted)
                        )
    return Trajectory(left, np.diff(breaks), result)
