"""Physical command specifications, values and reference sampling in SI units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drone_playground.planning.geometry import PlannerGeometry

import numpy as np


@dataclass(frozen=True)
class CommandSpec:
    name: str
    fields: tuple[str, ...]
    units: tuple[str, ...]
    frame: str


ATTITUDE_THRUST = CommandSpec(
    "attitude_thrust",
    ("roll", "pitch", "yaw", "thrust"),
    ("rad", "rad", "rad", "N"),
    "world attitude / body thrust",
)
THRUST_BODYRATES = CommandSpec(
    "thrust_bodyrates",
    ("thrust", "roll_rate", "pitch_rate", "yaw_rate"),
    ("N", "rad/s", "rad/s", "rad/s"),
    "body",
)
MOTOR_RPM = CommandSpec(
    "motor_rpm", ("motor0", "motor1", "motor2", "motor3"), ("rpm",) * 4, "rotor"
)
TRAJECTORY = CommandSpec(
    "trajectory",
    ("start_time", "segments", "yaw_defined"),
    ("s", "SI polynomial coefficients and seconds", "boolean"),
    "world",
)
WAYPOINT = CommandSpec(
    "waypoint", ("positions", "tolerance"), ("m", "m"), "world"
)
WORLD_ACCELERATION = CommandSpec(
    "world_acceleration",
    ("ax", "ay", "az"),
    ("m/s^2",) * 3,
    "world; gravity-compensated net acceleration",
)
VELOCITY_YAW = CommandSpec(
    "velocity_yaw",
    ("vx", "vy", "vz", "yaw"),
    ("m/s", "m/s", "m/s", "rad"),
    "world",
)
COMMANDS = {
    spec.name: spec
    for spec in (
        ATTITUDE_THRUST,
        THRUST_BODYRATES,
        MOTOR_RPM,
        TRAJECTORY,
        WAYPOINT,
        WORLD_ACCELERATION,
        VELOCITY_YAW,
    )
}


def require_match(output: str, input_: str) -> None:
    if output not in COMMANDS or input_ not in COMMANDS:
        raise ValueError(f"Unknown command contract: {output} -> {input_}")
    if output != input_:
        raise ValueError(f"Command interface mismatch: {output} -> {input_}")


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

    The time contract is ``[start_time, end_time]`` on the simulation clock: an
    upstream producer that generates this curve at simulation time ``t`` owns a
    duration, and every consumer samples it in simulation time, never wall time.
    ``frame`` names the coordinate system of the coefficients, matching the
    explicit frames on measurements and planner geometry.
    """

    start_time: float
    durations: np.ndarray
    coefficients: np.ndarray
    yaw_defined: bool = True
    frame: str = "world"

    def __post_init__(self):
        durations, coefficients = _finite(self.durations), _finite(
            self.coefficients
        )
        if not np.isfinite(self.start_time) or self.start_time < 0:
            raise ValueError(
                "Trajectory start time must be finite and nonnegative"
            )
        if durations.ndim != 1 or len(durations) == 0 or np.any(durations <= 0):
            raise ValueError("Trajectory durations must be a positive vector")
        if (
            coefficients.ndim != 3
            or coefficients.shape[:2] != (len(durations), 4)
            or not 1 <= coefficients.shape[2] <= 16
        ):
            raise ValueError(
                "Trajectory coefficients must be [segments,4,1..16]"
            )
        if not np.isfinite(self.start_time + durations.sum()):
            raise ValueError("Trajectory end time must be finite")
        if self.frame != "world":
            raise ValueError("Trajectory coefficients must use the world frame")
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def end_time(self):
        return float(self.start_time + self.durations.sum())

    def sample_many(self, times):
        times = np.atleast_1d(_finite(times))
        if times.ndim != 1:
            raise ValueError("Trajectory sample times must be a vector")
        if np.any(times < self.start_time - 1e-9) or np.any(
            times > self.end_time + 1e-9
        ):
            raise ValueError(
                "Requested time is outside the trajectory's valid interval"
            )
        offsets = np.r_[0.0, np.cumsum(self.durations)]
        elapsed = np.clip(times - self.start_time, 0, offsets[-1])
        indices = np.minimum(
            np.searchsorted(offsets[1:], elapsed, side="right"),
            len(self.durations) - 1,
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
                result = (
                    result * local_time[:, None]
                    + scale * coefficients[..., power]
                )
            values.append(result)
        return dict(
            position=values[0][:, :3],
            velocity=values[1][:, :3],
            acceleration=values[2][:, :3],
            yaw=values[0][:, 3],
            time=times,
        )

    def sample(self, time):
        return {
            name: value[0] for name, value in self.sample_many([time]).items()
        }

    @classmethod
    def from_bspline(cls, start_time, knots, control_points, degree, yaw=0.0):
        from scipy.interpolate import BSpline, PPoly

        knots, points = _finite(knots), _finite(control_points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("B-spline control points must be [N,3]")
        if not isinstance(degree, (int, np.integer)) or not 1 <= degree <= 15:
            raise ValueError("B-spline degree must be an integer in 1..15")
        if len(knots) != len(points) + degree + 1 or np.any(np.diff(knots) < 0):
            raise ValueError(
                "B-spline knot count/order disagrees with its control points"
            )
        curve = BSpline(knots, points, degree, extrapolate=False)
        axes = [
            PPoly.from_spline((curve.t, curve.c[:, axis], degree))
            for axis in range(3)
        ]
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
    """Ordered world goals with an explicit generation time and validity end.

    A waypoint carries no reach-by duration of its own: unlike a trajectory it is
    a setpoint whose life ends when a later decision replaces it or when
    ``valid_until`` passes on the simulation clock. ``generated_at`` records when
    the producing module created the goal, so a slower upstream can stay cached
    across faster downstream ticks without becoming indistinguishable from a
    fresh goal.
    """

    positions: np.ndarray
    tolerance: float
    generated_at: float = 0.0
    # Zero means the producer declared no horizon: the goal stays valid until a
    # later decision replaces it. This matches the protobuf encoding, where an
    # absent valid_until is also 0, and stays JSON-serializable for the archive.
    valid_until: float = 0.0

    def __post_init__(self):
        positions = _finite(self.positions)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or len(positions) == 0
        ):
            raise ValueError("Waypoint positions must be [N,3]")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Waypoint tolerance must be positive")
        if not np.isfinite(self.generated_at) or self.generated_at < 0:
            raise ValueError(
                "Waypoint generation time must be finite and nonnegative"
            )
        if not np.isfinite(self.valid_until) or self.valid_until < 0:
            raise ValueError(
                "Waypoint validity must be a finite nonnegative time"
            )
        if self.valid_until and self.valid_until < self.generated_at:
            raise ValueError(
                "A declared Waypoint validity must not precede its generation time"
            )
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class MotionCommand:
    kind: str
    values: np.ndarray

    def __post_init__(self):
        values = _finite(self.values)
        if self.kind not in COMMANDS or self.kind in ("trajectory", "waypoint"):
            raise ValueError(
                "Motion command must name an implemented physical command"
            )
        if values.shape != (len(COMMANDS[self.kind].fields),):
            raise ValueError(
                "Motion command values disagree with its physical fields"
            )
        object.__setattr__(self, "values", values)


def reference_horizon(trajectory, time, offsets, *, yaw=None):
    if not isinstance(trajectory, Trajectory):
        raise TypeError("A future horizon requires a full Trajectory")
    result = trajectory.sample_many(time + np.asarray(offsets))
    if not trajectory.yaw_defined:
        if yaw is None or not np.isfinite(yaw):
            raise ValueError(
                "Position-only trajectory requires an explicit downstream yaw"
            )
        result["yaw"] = np.full(len(result["time"]), yaw)
    return result


@dataclass(frozen=True)
class Decision:
    status: str
    output: Trajectory | Waypoint | MotionCommand | None
    plan_id: str
    generated_at: float
    valid_until: float
    diagnostics: dict
    explanation: str = ""
    sampled_reference: dict | None = None
    planner_geometry: PlannerGeometry | None = None
