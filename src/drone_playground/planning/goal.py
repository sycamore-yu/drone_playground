"""Expose a task goal as an explicitly timed waypoint reference."""

import numpy as np

from drone_playground.references import Waypoint
from drone_playground.runtime.decision import output_reply


class GoalWaypoints:
    input_kind, output_kind, derivatives = None, "waypoint", "none"

    def start(self, calibration, goal, limits, task):
        # A task goal has no producing tick, so it stays valid until replaced.
        self.goal = Waypoint([goal], 0.5)

    def step(self, packet, upstream):
        if "goal" in packet and not np.array_equal(
            np.asarray(packet["goal"], dtype=float), self.goal.positions[0]
        ):
            # A replaced target is a new setpoint, so its producing tick is the
            # one that replaces it; an unchanged target keeps its own origin.
            self.goal = Waypoint([packet["goal"]], 0.5, generated_at=float(packet["time"]))
        return output_reply(self.goal, self.goal.generated_at, "goal", packet["time"] + 1.0)

    def close(self):
        pass
