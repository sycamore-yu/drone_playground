"""Backend-independent physical outputs; all lengths, times and angles are SI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_playground.contracts import COMMANDS


def _finite(value):
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Physical interface values must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Trajectory:
    """World xyz/yaw polynomials in ascending powers of local segment seconds.

    Coefficients have shape [segments, 4, degree+1]. This representation retains
    complete B-spline/Hermite/polynomial curves without a resampling approximation.
    Querying outside the valid horizon is an error, never implicit extrapolation.
    """

    start_time: float
    durations: np.ndarray
    coefficients: np.ndarray
    yaw_defined: bool = True

    def __post_init__(self):
        durations, coefficients = _finite(self.durations), _finite(self.coefficients)
        if not np.isfinite(self.start_time):
            raise ValueError("Trajectory start time must be finite")
        if durations.ndim != 1 or len(durations) == 0 or np.any(durations <= 0):
            raise ValueError("Trajectory durations must be a positive vector")
        if (
            coefficients.ndim != 3
            or coefficients.shape[:2] != (len(durations), 4)
            or not 1 <= coefficients.shape[2] <= 16
        ):
            raise ValueError("Trajectory coefficients must be [segments,4,1..16]")
        if not np.isfinite(self.start_time + durations.sum()):
            raise ValueError("Trajectory end time must be finite")
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def end_time(self):
        return float(self.start_time + self.durations.sum())

    def sample_many(self, times):
        times = np.atleast_1d(_finite(times))
        if times.ndim != 1:
            raise ValueError("Trajectory sample times must be a vector")
        if np.any(times < self.start_time - 1e-9) or np.any(times > self.end_time + 1e-9):
            raise ValueError("Requested time is outside the trajectory's valid interval")
        offsets = np.r_[0.0, np.cumsum(self.durations)]
        elapsed = np.clip(times - self.start_time, 0, offsets[-1])
        indices = np.minimum(
            np.searchsorted(offsets[1:], elapsed, side="right"), len(self.durations) - 1
        )
        local_time = elapsed - offsets[indices]
        coefficients = self.coefficients[indices]
        values = []
        for derivative in range(3):
            result = np.zeros((len(times), 4))
            for power in range(coefficients.shape[-1] - 1, derivative - 1, -1):
                scale = 1 if derivative == 0 else power
                if derivative == 2:
                    scale *= power - 1
                result = result * local_time[:, None] + scale * coefficients[..., power]
            values.append(result)
        return dict(
            position=values[0][:, :3],
            velocity=values[1][:, :3],
            acceleration=values[2][:, :3],
            yaw=values[0][:, 3],
            time=times,
        )

    def sample(self, time):
        return {name: value[0] for name, value in self.sample_many([time]).items()}

    @classmethod
    def from_bspline(cls, start_time, knots, control_points, degree, yaw=0.0):
        from scipy.interpolate import BSpline, PPoly

        knots, points = _finite(knots), _finite(control_points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("B-spline control points must be [N,3]")
        if not isinstance(degree, (int, np.integer)) or not 1 <= degree <= 15:
            raise ValueError("B-spline degree must be an integer in 1..15")
        if len(knots) != len(points) + degree + 1 or np.any(np.diff(knots) < 0):
            raise ValueError("B-spline knot count/order disagrees with its control points")
        curve = BSpline(knots, points, degree, extrapolate=False)
        axes = [PPoly.from_spline((curve.t, curve.c[:, axis], degree)) for axis in range(3)]
        breaks = axes[0].x
        selected = (
            (np.diff(breaks) > 0)
            & (breaks[:-1] >= knots[degree])
            & (breaks[1:] <= knots[-degree - 1])
        )
        durations = np.diff(breaks)[selected]
        coefficients = np.zeros((len(durations), 4, degree + 1))
        for axis, polynomial in enumerate(axes):
            coefficients[:, axis] = polynomial.c[::-1, selected].T
        coefficients[:, 3, 0] = yaw
        return cls(start_time, durations, coefficients)


@dataclass(frozen=True)
class Waypoint:
    positions: np.ndarray
    tolerance: float

    def __post_init__(self):
        positions = _finite(self.positions)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("Waypoint positions must be [N,3]")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Waypoint tolerance must be positive")
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class MotionCommand:
    kind: str
    values: np.ndarray

    def __post_init__(self):
        values = _finite(self.values)
        if self.kind not in COMMANDS or self.kind in ("trajectory", "waypoint"):
            raise ValueError("Motion command must name an implemented physical command")
        if values.shape != (len(COMMANDS[self.kind].fields),):
            raise ValueError("Motion command values disagree with its physical fields")
        object.__setattr__(self, "values", values)
