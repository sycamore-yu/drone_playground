"""Host scheduling against real physics; the planner fixture is not a native solve."""

import numpy as np

from drone_playground.simulation.environment import Environment
from drone_playground.simulation.ros_planner import RosTrajectory
from drone_playground.simulation.runner import rollout_ros


def test_native_clock_measurements_and_control_between_decisions():
    """Verify native clock measurements and control between decisions."""

    class PlannerFixture:
        replan_interval = 0.2

        def reset(self, seed):
            self.calls = []

        def plan(self, observation, time, *, force_replan):
            assert force_replan
            self.calls.append((time, observation["measurement"]))
            position = np.broadcast_to(observation["position"], (2, 3))
            return RosTrajectory(
                np.array([time, time + 0.5]),
                position,
                np.zeros((2, 3)),
                np.zeros((2, 3)),
                {},
                1,
                len(self.calls),
            )

    env = Environment(task="navigation", scene="S01", sensor="depth", duration=0.42)
    planner = PlannerFixture()
    episodes, traces, plans, decisions = rollout_ros(
        env, planner, seed=5, planner_name="scheduling-fixture"
    )
    assert [time for time, _ in planner.calls] == [0, 0.2, 0.4]
    assert [m.timestamp_ns for _, m in planner.calls] == [0, 200_000_000, 400_000_000]
    assert episodes[0]["event"] == "TIMEOUT"
    assert len(decisions) == 3
    assert len(plans) == len(traces["time"]) == 22
    assert all(plan is not None for plan in plans[1:])
    assert np.isfinite(traces["position"]).all()
