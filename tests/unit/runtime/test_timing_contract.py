"""Trajectory and Waypoint carry an explicit simulation-clock time contract.

Trajectory is sampled only inside its own interval; Waypoint records when it was
produced and when it goes stale. Both survive the protobuf boundary unchanged, so
a slower upstream can stay cached across faster downstream ticks and a stale
decision is rejected rather than silently executed.
"""

import numpy as np
import pytest

from drone_playground.planning.goal import GoalWaypoints
from drone_playground.references import Trajectory, Waypoint
from drone_playground.rpc.proto import algorithm_pb2 as pb
from drone_playground.rpc.wire import body_state, decode_output, encode_output, validate_step
from drone_playground.runtime.pipeline import Pipeline


def _curve(start=1.0, duration=2.0):
    coefficients = np.array([[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
    return Trajectory(start, [duration], coefficients, yaw_defined=False)


def test_trajectory_time_contract_is_the_only_admissible_sampling_interval():
    curve = _curve(start=1.0, duration=2.0)
    assert (curve.start_time, curve.end_time) == (1.0, 3.0)
    np.testing.assert_allclose(curve.sample(1.5)["position"], [0.5, 0.0, 1.0])
    # Before the origin and after the horizon are both outside the contract.
    for outside in (0.99, 3.01):
        with pytest.raises(ValueError, match=r"valid interval"):
            curve.sample(outside)


def test_waypoint_defaults_declare_no_horizon_and_keep_the_untimed_goal_usable():
    goal = Waypoint([[1, 2, 3]], tolerance=0.5)
    assert goal.generated_at == 0.0 and goal.valid_until == 0.0


def test_waypoint_rejects_an_inverted_or_nonfinite_validity_window():
    with pytest.raises(ValueError, match=r"generation time"):
        Waypoint([[1, 2, 3]], 0.5, generated_at=2.0, valid_until=1.0)
    with pytest.raises(ValueError, match=r"generation time"):
        Waypoint([[1, 2, 3]], 0.5, generated_at=float("nan"))
    with pytest.raises(ValueError, match=r"validity"):
        Waypoint([[1, 2, 3]], 0.5, valid_until=float("inf"))


def test_trajectory_rejects_a_negative_origin_and_a_non_world_frame():
    coefficients = np.zeros((1, 4, 2))
    with pytest.raises(ValueError, match=r"nonnegative"):
        Trajectory(-1.0, [1.0], coefficients)
    with pytest.raises(ValueError, match=r"world frame"):
        Trajectory(0.0, [1.0], coefficients, frame="body")


def test_wire_round_trip_preserves_waypoint_and_trajectory_timing():
    goal = Waypoint([[1, 2, 3]], 0.5, generated_at=4.0, valid_until=4.5)
    restored = decode_output(encode_output(goal))
    assert (restored.generated_at, restored.valid_until) == (4.0, 4.5)
    curve = decode_output(encode_output(_curve(start=2.0, duration=1.5)))
    assert (curve.start_time, curve.end_time, curve.frame) == (2.0, 3.5, "world")


def test_wire_reports_an_untimed_waypoint_as_open_ended():
    restored = decode_output(encode_output(Waypoint([[1, 2, 3]], 0.5)))
    assert restored.valid_until == 0.0


def _step_with_upstream(waypoint, now, goal=None):
    request = pb.StepRequest(
        header=pb.Header(simulation_time=now),
        state=body_state(dict(position=[0, 0, 1], velocity=[0, 0, 0], quaternion=[0, 0, 0, 1])),
        solve_budget_seconds=0.05,
    )
    request.waypoints.CopyFrom(encode_output(waypoint))
    if goal is not None:
        request.goal.CopyFrom(encode_output(goal))
    return request


def test_stale_or_future_upstream_waypoint_is_rejected_before_execution():
    fresh = Waypoint([[1, 0, 1]], 0.5, generated_at=1.0, valid_until=2.0)
    validate_step(_step_with_upstream(fresh, 1.5))
    with pytest.raises(ValueError, match=r"time window"):
        validate_step(_step_with_upstream(fresh, 2.5))
    with pytest.raises(ValueError, match=r"time window"):
        validate_step(_step_with_upstream(fresh, 0.5))


def test_untimed_task_goal_is_accepted_as_a_plain_setpoint():
    validate_step(
        _step_with_upstream(Waypoint([[1, 0, 1]], 0.5), 3.0, goal=Waypoint([[1, 0, 1]], 0.5))
    )


def test_goal_module_restamps_only_a_changed_target():
    module = GoalWaypoints()
    module.start({}, [3, 0, 1], None, {})
    first = module.step(dict(time=1.0), None)["output"]
    assert first.generated_at == 0.0  # Task goal carries no producing tick.
    repeated = module.step(dict(time=2.0, goal=[3, 0, 1]), None)["output"]
    assert repeated.generated_at == 0.0  # Identical target is not relabelled.
    changed = module.step(dict(time=3.0, goal=[5, 0, 1]), None)["output"]
    assert changed.generated_at == 3.0
    np.testing.assert_allclose(changed.positions, [[5, 0, 1]])


def test_ten_hertz_source_stays_cached_across_fifty_hertz_execution(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import drone_playground.runtime.pipeline as pipeline

    class Source:
        input_kind, output_kind, derivatives = None, "waypoint", "none"

        def start(self, *args):
            self.calls = []

        def step(self, packet, upstream):
            self.calls.append(packet["time"])
            # One 10 Hz decision stays valid for less than two 50 Hz ticks, so
            # the cache must expire rather than silently reuse a stale goal. The
            # module stamps only the output; the stage envelope mirrors it.
            return dict(
                output=Waypoint(
                    [[3, 0, 1]], 0.1, generated_at=packet["time"], valid_until=packet["time"] + 0.05
                ),
                plan_id=str(len(self.calls)),
                decision_status="valid",
            )

        def close(self):
            pass

    source = Source()
    monkeypatch.setattr(pipeline, "instantiate", lambda _: source)
    planner = Pipeline(
        dict(output="waypoint", stages=[dict(implementation="python", frequency_hz=10)]),
        tmp_path,
        SimpleNamespace(freq=50),
    )
    planner.start({}, [3, 0, 1])
    replies = [planner.step(dict(time=t)) for t in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)]
    assert source.calls == [0.0, 0.10]  # One production tick per 100 ms.
    # The 50 ms decision is executable at the tick it was produced and for the
    # two cached ticks inside its window; past that it expires instead of being
    # handed downstream, until the next production tick refreshes it.
    assert [reply["output"] is not None for reply in replies] == [
        True,
        True,
        True,
        False,
        False,
        True,
    ]
    # Expiry is attributed to the cached stage, and the surviving cached output
    # keeps its original producing time rather than looking freshly generated.
    assert replies[3]["stages"][0]["expired_stage"] == 0
    assert [reply["generated_at"] for reply in replies[:5]] == [0.0] * 5
    assert replies[1]["stages"][0]["generated_at"] == 0.0


def test_reset_clears_the_waypoint_time_state(tmp_path):
    from types import SimpleNamespace

    planner = Pipeline(
        dict(output="waypoint", stages=[dict(implementation="goal")]),
        tmp_path,
        SimpleNamespace(freq=50),
    )
    planner.start({}, [3, 0, 1])
    assert planner.step(dict(time=0.0))["output"].generated_at == 0.0
    planner.step(dict(time=0.02, goal=[9, 0, 1]))
    assert planner.modules[0].goal.generated_at == 0.02
    planner.start({}, [3, 0, 1])
    assert planner.modules[0].goal.generated_at == 0.0
    planner.close()
