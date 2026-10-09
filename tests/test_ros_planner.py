"""Wire/client behavior; ros_planner_worker/integration.py exercises real C++ solves."""

import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
import pytest

from drone_playground.simulation.ros_planner import (
    RosPlanner,
    RosPlannerError,
    cloud_measurement,
    depth_measurement,
)
from drone_playground.simulation.ros_planner import planner_pb2 as rpc
from drone_playground.simulation.ros_planner import planner_pb2_grpc as rpc_grpc


class WireFixture(rpc_grpc.RosPlannerServicer):
    """Protocol fixture, deliberately not evidence of a native planning solve."""

    delay = 0.0
    failure = rpc.SOLVED
    wrong_sequence = False

    def Initialize(self, request, context):  # noqa: N802 - Protobuf RPC method name.
        """Provide Initialize for the surrounding execution."""
        return rpc.Session(id="wire-fixture", episode=1, planner=request.planner)

    def Reset(self, request, context):  # noqa: N802 - Protobuf RPC method name.
        """Provide Reset for the surrounding execution."""
        return rpc.Session(id=request.id, episode=request.episode + 1, planner=request.planner)

    def Close(self, request, context):  # noqa: N802 - Protobuf RPC method name.
        """Provide Close for the surrounding execution."""
        return rpc.Closed()

    def Decide(self, request, context):  # noqa: N802 - Protobuf RPC method name.
        """Provide Decide for the surrounding execution."""
        time.sleep(self.delay)
        t = request.state.timestamp_ns
        return rpc.Decision(
            status=self.failure,
            detail="fixture",
            session=request.session,
            sequence=request.sequence + int(self.wrong_sequence),
            input_timestamp_ns=t,
            valid_from_ns=t,
            valid_until_ns=t + 100_000_000,
            samples=[
                rpc.Sample(timestamp_ns=t, position=rpc.Vec3(z=2)),
                rpc.Sample(timestamp_ns=t + 100_000_000, position=rpc.Vec3(x=1, z=2)),
            ],
        )


@pytest.fixture
def wire():
    """Provide wire for the surrounding execution."""
    fixture = WireFixture()
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    rpc_grpc.add_RosPlannerServicer_to_server(fixture, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield fixture, f"127.0.0.1:{port}"
    server.stop(0).wait()


def observation(t=0.0):
    """Provide observation for the surrounding execution."""
    return dict(
        position=(0, 0, 2),
        velocity=(0, 0, 0),
        acceleration=(0, 0, 0),
        quaternion=(1, 0, 0, 0),
        goal=(4, 0, 2),
        measurement=cloud_measurement([[8, 0, 2]], t),
    )


def test_lifecycle_replan_and_expired_cache(wire):
    """Verify lifecycle replan and expired cache."""
    _, address = wire
    with RosPlanner("ego", address, replan_interval=0.2) as planner:
        first = planner(observation(), 0.0)
        assert planner(observation(0.05), 0.05) is first
        with pytest.raises(RosPlannerError, match="STALE_OUTPUT"):
            planner(observation(0.15), 0.15)
        second = planner(observation(0.2), 0.2)
        assert second.sequence == 2
        planner.reset()
        assert planner.session.episode == 2
        assert planner(observation(), 0.0).sequence == 1
    with pytest.raises(RosPlannerError, match="CLOSED"):
        planner(observation(), 0.0)


@pytest.mark.parametrize("failure", [rpc.NO_SOLUTION, rpc.STALE_INPUT, rpc.INVALID_INPUT])
def test_failed_replan_never_reuses_previous_output(wire, failure):
    """Verify failed replan never reuses previous output."""
    fixture, address = wire
    with RosPlanner("super", address) as planner:
        planner(observation(), 0.0)
        fixture.failure = failure
        with pytest.raises(RosPlannerError):
            planner(observation(0.01), 0.01, force_replan=True)
        with pytest.raises(RosPlannerError):
            planner(observation(0.02), 0.02)


def test_mismatched_response_and_reversed_clock(wire):
    """Verify mismatched response and reversed clock."""
    fixture, address = wire
    with RosPlanner("ego", address) as planner:
        fixture.wrong_sequence = True
        with pytest.raises(RosPlannerError, match="STALE_OUTPUT"):
            planner(observation(1.0), 1.0)
        with pytest.raises(RosPlannerError, match="STALE_INPUT"):
            planner(observation(), 0.0)


def test_trajectory_start_accepts_wire_timestamp_rounding(wire):
    """Verify trajectory start accepts wire timestamp rounding."""
    _, address = wire
    timestamp = float(np.float32(0.3))
    with RosPlanner("ego", address) as planner:
        trajectory = planner(observation(timestamp), timestamp)
        np.testing.assert_allclose(trajectory.sample(timestamp)[0], [0, 0, 2])


def test_deadline_invalidates_session_until_reset(wire):
    """Verify deadline invalidates session until reset."""
    fixture, address = wire
    with RosPlanner("ego", address, timeout=0.1) as planner:
        fixture.delay = 0.3
        with pytest.raises(RosPlannerError, match="DEADLINE_EXCEEDED"):
            planner(observation(), 0.0)
        with pytest.raises(RosPlannerError, match="RESET_REQUIRED"):
            planner(observation(0.2), 0.2)
        fixture.delay = 0.0
        planner.reset()
        assert planner(observation(), 0.0).sequence == 1


def test_measurement_units_pose_and_validation():
    """Verify measurement units pose and validation."""
    cloud = cloud_measurement([[1, 2, 3]], 1.25, available_time=1.3, position=(4, 5, 6))
    assert cloud.timestamp_ns == 1_250_000_000
    assert cloud.available_ns == 1_300_000_000
    assert cloud.world_from_sensor.position.x == 4
    depth = depth_measurement(
        np.ones((3, 4)),
        0.0,
        fx=2.0,
        fy=2.0,
        cx=1.5,
        cy=1.0,
        position=(0, 0, 0),
        quaternion=(1, 0, 0, 0),
    )
    assert depth.depth.width == 4 and len(depth.depth.metres) == 12
    with pytest.raises(ValueError):
        cloud_measurement([[np.nan, 0, 0]], 0.0)
    with pytest.raises(ValueError):
        cloud_measurement([[1, 2, 3]], -1.0)
    with pytest.raises(ValueError):
        cloud_measurement([[1, 2, 3]], 0.0, quaternion=(0, 0, 0, 0))


def test_sensor_generator_nearest_hits():
    # The standalone integration sensor really occludes the room with the box.
    """Verify sensor generator nearest hits."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ros_planner_integration",
        Path(__file__).parents[1] / "src" / "ros_planner_worker" / "integration.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clear, blocked = module.sensor_returns(), module.sensor_returns(True)
    assert np.isfinite(blocked).all()
    assert len(clear) == len(blocked)
    assert np.any(np.linalg.norm(clear - blocked, axis=1) > 1)
