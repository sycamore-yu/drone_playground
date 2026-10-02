import importlib.util

import numpy as np
import pytest

from drone_playground.actions.commands import Waypoint
from drone_playground.rpc.proto import algorithm_pb2 as pb
from drone_playground.planning.waypoints import OrderedWaypointGoals
from drone_playground.rpc.wire import body_state, encode_output, validate_step
from tests.helpers.paths import REPO_ROOT


def test_ordered_native_goals_keep_progress_across_identical_packets_and_hold_final_target():
    router = OrderedWaypointGoals()
    route = Waypoint([[1,0,1],[2,0,2],[3,0,1]], .1)
    target, changed = router.update(route, [3,0,1])
    np.testing.assert_array_equal(target, [1,0,1])  # No nearest/final-point shortcut.
    assert changed and router.index == 0
    target, changed = router.update(route, [1.01,0,1])
    np.testing.assert_array_equal(target, [2,0,2])
    assert changed and router.index == 1
    assert not router.update(Waypoint(route.positions.copy(), .1), [1,0,1])[1]
    assert router.index == 1
    router.update(route, [2,0,2])
    target, changed = router.update(route, [3,0,1])
    np.testing.assert_array_equal(target, [3,0,1])
    assert not changed and router.index == 2
    replacement = Waypoint([[5,0,2],[6,0,1]], .1)
    assert router.update(replacement, [3,0,1])[1] and router.index == 0
    assert OrderedWaypointGoals().update(route, [0,0,1])[1]


@pytest.mark.parametrize('method',['super','ego'])
def test_ros_upstream_waypoint_mode_advertises_and_enforces_actual_input(monkeypatch,method):
    path=REPO_ROOT/'ros_integrations/ros1/bridge/worker.py'
    spec=importlib.util.spec_from_file_location('ros_waypoint_worker',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'native_runtime_identity',lambda _: {})
    algorithm=object.__new__(module.RosAlgorithm)
    algorithm.method=method
    request=pb.InitializeRequest(algorithm=method)
    original=algorithm.initialize(request)
    assert not original.capabilities.accepted_upstream
    request.parameters.update(dict(use_upstream_waypoints=True))
    reply=algorithm.initialize(request)
    assert list(reply.capabilities.accepted_upstream)==['waypoint']
    assert reply.capabilities.derivatives=='none'
    step=pb.StepRequest(solve_budget_seconds=.01,state=body_state(dict(
        position=[0,0,1],velocity=[0,0,0],quaternion=[0,0,0,1])))
    with pytest.raises(ValueError,match=r'requires inputs'):
        validate_step(step,reply.capabilities)
    step.waypoints.CopyFrom(encode_output(Waypoint([[1,0,1],[2,0,2]],.1)))
    validate_step(step,reply.capabilities)
    request.parameters.update(dict(use_upstream_waypoints='false'))
    with pytest.raises(ValueError,match=r'boolean'):
        algorithm.initialize(request)
