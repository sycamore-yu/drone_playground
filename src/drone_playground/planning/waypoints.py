"""Stateful ordered goals for native planners that accept one target at a time."""

import numpy as np

from drone_playground.actions.commands import Waypoint


class OrderedWaypointGoals:

    def __init__(self):
        self.positions, self.index, self.target = None, 0, None

    def update(self, waypoints, position):
        if not isinstance(waypoints, Waypoint):
            raise TypeError(
                "Native goal routing requires physical Waypoint input"
            )
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(
                "Waypoint progress requires a finite world position"
            )
        if self.positions is None or not np.array_equal(
            waypoints.positions, self.positions
        ):
            self.positions, self.index = waypoints.positions.copy(), 0
        while (
            self.index < len(self.positions) - 1
            and np.linalg.norm(position - self.positions[self.index])
            <= waypoints.tolerance
        ):
            self.index += 1
        target = self.positions[self.index].copy()
        changed = self.target is None or not np.array_equal(target, self.target)
        self.target = target
        return target, changed
