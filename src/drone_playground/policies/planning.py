"""Frozen reference generators shared by neural policies and optimization control."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


def random_trajectory(seed: int, duration: float = 15.0, freq: int = 50) -> np.ndarray:
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
class TrajectoryPlan:
    name: str = "figure8"
    amplitude: float = 1.0
    kind: str = "fixed_generator"

    def build(self, seed: int, count: int, duration: float, freq: int, start=None):
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
                positions, ((0, round(duration * freq) - len(positions)), (0, 0)), mode="edge"
            )[None]
        raise ValueError(f"Unknown trajectory generator: {self.name}")
