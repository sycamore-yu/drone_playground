"""Foreign runtimes exchange physical types, fields and independent SFC validity."""

import numpy as np
import pytest

from drone_playground.control.setpoints import (
    AttitudeSetpoint,
    ForceTorque,
    MotorRPM,
    RateSetpoint,
    StateSetpoint,
)


@pytest.mark.parametrize(
    "value",
    [
        StateSetpoint(velocity=np.array([1.0, 2.0, 3.0]), yaw=0.2),
        StateSetpoint(acceleration=np.array([1.0, 2.0, 3.0])),
        AttitudeSetpoint(rpy=np.array([0.1, 0.2, 0.3]), thrust=0.4),
        RateSetpoint(thrust=0.4, body_rates=np.array([0.1, 0.2, 0.3])),
        ForceTorque(thrust=0.4, torque=np.array([0.01, 0.02, 0.03])),
        MotorRPM(rpm=np.array([100.0, 200.0, 300.0, 400.0])),
    ],
)
def test_typed_setpoint_roundtrip_preserves_physical_fields(value):
    from dataclasses import fields

    from drone_playground.rpc.wire import decode_output, encode_output

    actual = decode_output(encode_output(value))
    assert type(actual) is type(value)
    for field in fields(value):
        expected = getattr(value, field.name)
        if expected is None:
            assert getattr(actual, field.name) is None
        else:
            np.testing.assert_allclose(getattr(actual, field.name), expected)


def test_corridor_roundtrip_preserves_its_own_time_window():
    from drone_playground.planning.corridors import ConvexPolytope, SafeFlightCorridor
    from drone_playground.rpc.wire import decode_corridor, encode_corridor

    polytope = ConvexPolytope(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    corridor = SafeFlightCorridor("candidate", (polytope,), generated_at=1.2, valid_until=1.5)
    actual = decode_corridor(encode_corridor(corridor))
    assert actual.generated_at == 1.2 and actual.valid_until == 1.5
    np.testing.assert_array_equal(actual.polytopes[0].vertices, polytope.vertices)


def test_wrong_or_nonfinite_wire_values_are_rejected():
    from drone_playground.rpc.proto import algorithm_pb2 as pb
    from drone_playground.rpc.wire import decode_output

    with pytest.raises(ValueError, match="finite"):
        decode_output(pb.RateSetpoint(thrust=float("nan"), body_rates=pb.Vec3()))
    with pytest.raises(ValueError, match="body_rates"):
        decode_output(pb.RateSetpoint(thrust=0.4))
    with pytest.raises(ValueError, match="field"):
        decode_output(pb.StateSetpoint())
