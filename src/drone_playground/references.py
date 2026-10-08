"""Waypoint and timed polynomial references for host and differentiable control.

Host construction checks values and time intervals. JAX sampling keeps the same
polynomial representation and returns nonfinite samples outside its valid domain,
so a compiled consumer cannot silently extrapolate an expired reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.interpolate import CubicSpline

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    # The native planner bridge uses the same NumPy representation without JAX.
    jax, jnp = None, np


def _jax_value(value):
    return jax is not None and isinstance(value, (jax.Array, jax.core.Tracer))


def _register_pytree(cls):
    return cls if jax is None else jax.tree_util.register_pytree_node_class(cls)


def _finite(value):
    if _jax_value(value):
        return value
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Physical interface values must be finite")
    result.setflags(write=False)
    return result


def _traced(*values):
    return jax is not None and any(
        isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(values)
    )


@_register_pytree
@dataclass(frozen=True)
class Trajectory:
    """World xyz/yaw polynomials, in ascending powers of local segment seconds."""

    start_time: float
    durations: np.ndarray
    coefficients: np.ndarray
    yaw_defined: bool = True
    frame: str = "world"
    kind: ClassVar[str] = "trajectory"

    def __post_init__(self):
        """Validate and prepare the Trajectory instance after initialization."""
        durations = _finite(self.durations)
        coefficients = _finite(self.coefficients)
        if durations.ndim != 1 or len(durations) == 0:
            raise ValueError("Trajectory durations must be a positive vector")
        if coefficients.ndim != 3 or coefficients.shape[:2] != (len(durations), 4):
            raise ValueError("Trajectory coefficients must be [segments,4,degree+1]")
        if not 1 <= coefficients.shape[2] <= 16:
            raise ValueError("Trajectory coefficients must use degrees 0..15")
        if self.frame != "world":
            raise ValueError("Trajectory coefficients must use the world frame")
        if not _traced(self.start_time, durations, coefficients):
            if not np.isfinite(self.start_time) or self.start_time < 0:
                raise ValueError("Trajectory start time must be finite and nonnegative")
            if not np.isfinite(durations).all() or np.any(np.asarray(durations) <= 0):
                raise ValueError("Trajectory durations must be positive and finite")
            if not np.isfinite(np.asarray(coefficients)).all():
                raise ValueError("Trajectory coefficients must be finite")
            if not np.isfinite(self.start_time + np.asarray(durations).sum()):
                raise ValueError("Trajectory end time must be finite")
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "coefficients", coefficients)

    def tree_flatten(self):
        """Expose only time and coefficients to JAX transformations."""
        return (self.start_time, self.durations, self.coefficients), (self.yaw_defined, self.frame)

    @classmethod
    def tree_unflatten(cls, metadata, values):
        """Reconstruct transformed arrays without running host value checks."""
        result = object.__new__(cls)
        for name, value in zip(
            ("start_time", "durations", "coefficients", "yaw_defined", "frame"),
            (*values, *metadata),
            strict=True,
        ):
            object.__setattr__(result, name, value)
        return result

    @property
    def end_time(self):
        return self.start_time + self.durations.sum()

    def sample_many(self, times):
        """Sample position, velocity and acceleration on the simulation clock."""
        use_jax = any(
            _jax_value(value)
            for value in (times, self.start_time, self.durations, self.coefficients)
        )
        xp = jnp if use_jax else np
        times = xp.atleast_1d(xp.asarray(times))
        if times.ndim != 1:
            raise ValueError("Trajectory sample times must be a vector")
        valid = (times >= self.start_time - 1e-9) & (times <= self.end_time + 1e-9)
        if not _traced(times, self.start_time, self.durations) and not np.all(np.asarray(valid)):
            raise ValueError("Requested time is outside the trajectory's valid interval")
        if not _traced(times) and not np.isfinite(np.asarray(times)).all():
            raise ValueError("Trajectory sample times must be finite")
        offsets = xp.concatenate((xp.zeros(1), xp.cumsum(self.durations)))
        elapsed = xp.clip(times - self.start_time, 0, offsets[-1])
        indices = xp.minimum(
            xp.searchsorted(offsets[1:], elapsed, side="right"), len(self.durations) - 1
        )
        local_time = elapsed - offsets[indices]
        coefficients = xp.asarray(self.coefficients)[indices]
        values = []
        for derivative in range(3):
            result = xp.zeros((len(times), 4), dtype=coefficients.dtype)
            for power in range(coefficients.shape[-1] - 1, derivative - 1, -1):
                scale = 1 if derivative == 0 else power
                if derivative == 2:
                    scale *= power - 1
                result = result * local_time[:, None] + scale * coefficients[..., power]
            values.append(xp.where(valid[:, None], result, xp.nan))
        return dict(
            position=values[0][:, :3],
            velocity=values[1][:, :3],
            acceleration=values[2][:, :3],
            yaw=values[0][:, 3],
            time=times,
        )

    def sample(self, time):
        """Sample one instantaneous reference."""
        return {name: value[0] for name, value in self.sample_many(time).items()}

    @classmethod
    def from_bspline(cls, start_time, knots, control_points, degree, yaw=0.0):
        """Convert an upstream B-spline exactly, without resampling its curve."""
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


@_register_pytree
@dataclass(frozen=True)
class Waypoint:
    """Ordered world goals with tolerance and an explicit validity envelope."""

    positions: np.ndarray
    tolerance: float
    generated_at: float = 0.0
    valid_until: float = 0.0
    kind: ClassVar[str] = "waypoint"
    frame: ClassVar[str] = "world"

    def __post_init__(self):
        """Validate and prepare the Waypoint instance after initialization."""
        positions = _finite(self.positions)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("Waypoint positions must be [N,3]")
        if not _traced(positions, self.tolerance, self.generated_at, self.valid_until):
            if not np.isfinite(np.asarray(positions)).all():
                raise ValueError("Waypoint positions must be finite")
            if not np.isfinite(self.tolerance) or self.tolerance <= 0:
                raise ValueError("Waypoint tolerance must be positive")
            if (
                not np.isfinite([self.generated_at, self.valid_until]).all()
                or min(self.generated_at, self.valid_until) < 0
            ):
                raise ValueError("Waypoint times must be finite and nonnegative")
            if self.valid_until and self.valid_until < self.generated_at:
                raise ValueError("Waypoint validity must not precede generation time")
        object.__setattr__(self, "positions", positions)

    def tree_flatten(self):
        """Keep positions and their simulation-clock envelope in the pytree."""
        return (self.positions, self.tolerance, self.generated_at, self.valid_until), None

    @classmethod
    def tree_unflatten(cls, metadata, values):
        """Reconstruct transformed arrays without host validation."""
        del metadata
        result = object.__new__(cls)
        for name, value in zip(
            ("positions", "tolerance", "generated_at", "valid_until"), values, strict=True
        ):
            object.__setattr__(result, name, value)
        return result


Reference = Waypoint | Trajectory


def reference_horizon(trajectory, time, offsets, *, yaw=None):
    """Sample a true future horizon, preserving explicit yaw availability."""
    if not isinstance(trajectory, Trajectory):
        raise TypeError("A future horizon requires a full Trajectory")
    result = trajectory.sample_many(time + np.asarray(offsets))
    if not trajectory.yaw_defined:
        if yaw is None or not np.isfinite(yaw):
            raise ValueError("Position-only trajectory requires an explicit downstream yaw")
        result["yaw"] = np.full(len(result["time"]), yaw)
    return result


def random_trajectory(seed: int, duration: float = 15.0, freq: int = 50) -> np.ndarray:
    """Sample a reproducible random trajectory reference for flight tracking."""
    takeoff = np.array([-1.5, 1.0, 0.07])
    waypoints = np.random.RandomState(seed).uniform(-1, 1, (10, 3))
    waypoints = waypoints * [1.2, 1.2, 0.5] + 0.3 * takeoff + [0, 0, 0.7]
    waypoints[:3] = [[-1.5, 1.0, 0.07], [-1.0, 0.55, 0.4], [0.3, 0.35, 0.7]]
    spline = CubicSpline(
        np.linspace(0, duration, 10),
        waypoints,
        bc_type=((1, np.array([0.0, 0.0, 0.4])), "not-a-knot"),
    )
    return spline(np.linspace(0, duration, int(np.ceil(duration * freq)))).astype(np.float32)


def race_reference(start, freq=50, duration=18.75):
    """Construct the reference motion used by a racing scenario."""
    knots = np.array(
        [
            start,
            [-1, 0.75, 0.4],
            [0.3, 0.35, 0.7],
            [1.3, -0.15, 0.9],
            [0.85, 0.85, 1.2],
            [-0.5, -0.05, 0.7],
            [-1.2, -0.2, 0.8],
            [-1.2, -0.2, 1.2],
            [0, -0.7, 1.2],
            [1.2, -0.15, 1.2],
            [1.05, 0.75, 1.2],
            [0.25, 1.25, 1.2],
        ]
    )
    spline = CubicSpline(np.linspace(0, duration, len(knots)), knots)
    stamps = np.linspace(0, duration, int(freq * duration))
    return spline(stamps).astype(np.float32), spline.derivative()(stamps).astype(np.float32)


@dataclass(frozen=True)
class ReferenceGenerator:
    name: str = "figure8"
    amplitude: float = 1.0
    kind: str = "fixed_generator"
    target: tuple[float, float, float] = (0.0, 0.0, 1.5)

    def build(self, seed: int, count: int, duration: float, freq: int, start=None):
        if self.name == "hover":
            target = np.asarray(self.target, np.float32)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("Hover target must be three finite world coordinates")
            return np.broadcast_to(target, (1, round(duration * freq), 3)).copy()
        if self.name == "random":
            return np.stack([random_trajectory(seed + i, duration, freq) for i in range(count)])
        if self.name == "figure8":
            t = np.linspace(0, 2 * np.pi, int(np.ceil(duration * freq)))
            return np.array(
                [
                    self.amplitude * np.sin(t),
                    np.zeros_like(t),
                    0.5 * self.amplitude * np.sin(2 * t) + 1,
                ]
            ).T[None]
        if self.name == "lsy_course":
            positions, _ = race_reference(start, freq)
            return np.pad(
                positions,
                ((0, round(duration * freq) - len(positions)), (0, 0)),
                mode="edge",
            )[None]
        raise ValueError(f"Unknown trajectory generator: {self.name}")
