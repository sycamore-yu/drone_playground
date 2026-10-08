"""Fixed-boundary minimum-jerk interpolation, with explicit heuristic timing.

Each segment minimizes the integral of squared jerk subject to its endpoint
position, velocity and acceleration constraints. It is not obstacle avoidance.
The first segment preserves current velocity; interior waypoints are full stops.
"""

import numpy as np

from drone_playground.references import Trajectory, Waypoint
from drone_playground.runtime.decision import output_reply


def minimum_jerk_path(
    waypoints,
    position,
    velocity,
    time,
    *,
    cruise_speed=2.0,
    acceleration_scale=3.0,
    minimum_segment_seconds=0.5,
):
    """Generate a time-parameterized, minimum-jerk reference through waypoints."""
    if not isinstance(waypoints, Waypoint):
        raise TypeError("Minimum-jerk planning requires ordered Waypoint input")
    if (
        not np.isfinite([cruise_speed, acceleration_scale, minimum_segment_seconds]).all()
        or min(cruise_speed, acceleration_scale, minimum_segment_seconds) <= 0
    ):
        raise ValueError("Positive finite speed, acceleration and duration scales required")
    points = np.vstack([position, waypoints.positions])
    velocity = np.asarray(velocity, dtype=float)
    durations = np.maximum(
        minimum_segment_seconds,
        np.maximum(
            1.875 * np.linalg.norm(np.diff(points, axis=0), axis=1) / cruise_speed,
            np.sqrt(5.774 * np.linalg.norm(np.diff(points, axis=0), axis=1) / acceleration_scale),
        ),
    )
    coefficients = np.zeros((len(durations), 4, 6))
    for i, duration in enumerate(durations):
        initial_velocity = velocity if i == 0 else np.zeros(3)
        coefficients[i, :3, 0], coefficients[i, :3, 1] = (
            points[i],
            initial_velocity,
        )
        # Solve in unit time to avoid powers of a short duration in the matrix.
        residual = np.stack(
            [
                points[i + 1] - points[i] - initial_velocity * duration,
                -initial_velocity * duration,
                np.zeros(3),
            ]
        )
        high = np.linalg.solve(
            np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]]),
            residual,
        )
        coefficients[i, :3, 3:] = (high / duration ** np.arange(3, 6)[:, None]).T
    return Trajectory(time, durations, coefficients, yaw_defined=False)


class MinimumJerkPlanner:
    input_kind, output_kind, derivatives = "waypoint", "trajectory", "none"

    def __init__(self, **settings):
        self.settings = settings
        self.curve, self.waypoints = None, None
        self.generated_at = None
        self.count = 0

    def start(self, calibration, goal, limits, task):
        self.curve, self.waypoints, self.generated_at, self.count = (
            None,
            None,
            None,
            0,
        )

    def step(self, packet, upstream):
        from drone_playground.planning.minimum_jerk import minimum_jerk_path

        # Preserve a generated curve while its physical input is unchanged.
        # Rebuilding from the current state at every tick would never progress
        # through the time parameterization of a rest-to-rest segment.
        if (
            self.curve is None
            or packet["time"] > self.curve.end_time
            or not np.array_equal(upstream.positions, self.waypoints)
        ):
            self.curve = minimum_jerk_path(
                upstream,
                packet["position"],
                packet["velocity"],
                packet["time"],
                **self.settings,
            )
            self.waypoints = upstream.positions
            self.generated_at = float(packet["time"])
            self.count += 1
        return output_reply(self.curve, self.generated_at, str(self.count), self.curve.end_time)

    def close(self):
        pass
