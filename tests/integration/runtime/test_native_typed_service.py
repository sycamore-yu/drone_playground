"""Exercise the actual C++ SDK with typed physical outputs and reset state."""

import os
from pathlib import Path

import numpy as np
import pytest

from drone_playground.control.setpoints import AttitudeSetpoint, RateSetpoint, StateSetpoint
from drone_playground.references import Trajectory, Waypoint
from drone_playground.rpc.client import NativeClient


def native_binary():
    path = Path(
        os.environ.get(
            "DRONE_NATIVE_TEST_BINARY",
            "tmp/direct-composition-refactor/native-sdk-v2/interop_server",
        )
    )
    if not path.is_file():
        pytest.skip("Build the native SDK interop fixture before this integration test")
    return str(path.resolve())


def physical_state():
    return dict(
        position=[0, 0, 1], velocity=[0, 0, 0], quaternion=[0, 0, 0, 1], angular_velocity=[0, 0, 0]
    )


@pytest.mark.parametrize(
    "value",
    [
        StateSetpoint(velocity=np.array([1.0, 2.0, 3.0]), yaw=0.2),
        StateSetpoint(acceleration=np.array([1.0, 2.0, 3.0])),
        AttitudeSetpoint(rpy=np.zeros(3), thrust=0.35),
        RateSetpoint(thrust=0.35, body_rates=np.zeros(3)),
        Waypoint([[1.0, 2.0, 3.0]], 0.5, generated_at=0.0, valid_until=2.0),
        Trajectory(0.0, [2.0], np.zeros((1, 4, 2))),
    ],
)
def test_native_echo_and_reset_preserve_physical_type(value, tmp_path):
    with NativeClient("echo", command=[native_binary(), "{address}"], directory=tmp_path) as client:
        client.reset(state=physical_state())
        first = client.step(time=0.0, state=physical_state(), upstream=value)
        assert type(first.output) is type(value)
        assert first.plan_id == "1"
        assert client.step(time=0.02, state=physical_state(), upstream=value).plan_id == "2"
        client.reset(state=physical_state())
        assert client.step(time=0.0, state=physical_state(), upstream=value).plan_id == "1"


def test_native_corridor_is_recorded_and_rendered_as_independent_evidence(tmp_path):
    from drone_playground.artifacts.decisions import NativeDecisionRecorder, load_native_decisions
    from drone_playground.control.controllers.crazyflow import AttitudeControl

    with NativeClient(
        "visualized", command=[native_binary(), "{address}"], directory=tmp_path / "native"
    ) as client:
        client.reset(state=physical_state())
        decision = client.step(time=0.2, state=physical_state())
    assert len(decision.corridors) == 1
    assert decision.corridors[0].generated_at == pytest.approx(0.2)
    reply = dict(
        output=decision.output,
        corridors=decision.corridors,
        generated_at=decision.generated_at,
        valid_until=decision.valid_until,
    )
    with NativeDecisionRecorder(tmp_path / "decisions", AttitudeControl().contract()) as recorder:
        recorder.record(0, 0.2, physical_state(), reply, None)
    restored = list(load_native_decisions(tmp_path / "decisions"))[0]
    np.testing.assert_array_equal(
        restored["reply"]["corridors"][0].polytopes[0].halfspaces,
        decision.corridors[0].polytopes[0].halfspaces,
    )


def test_native_late_result_is_not_executable(tmp_path):
    with NativeClient(
        "rates",
        command=[native_binary(), "{address}"],
        directory=tmp_path,
        parameters={"delay_seconds": 0.02},
    ) as client:
        client.reset(state=physical_state())
        result = client.step(time=0.0, state=physical_state(), solve_budget_seconds=0.001)
    assert result.status == "budget_exhausted"
    assert result.output is None
