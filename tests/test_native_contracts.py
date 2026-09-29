"""Physical outputs retain a full reference horizon across native boundaries."""

import numpy as np
import pytest

from drone_playground.native.contracts import MotionCommand, Trajectory, Waypoint


def test_polynomial_reference_keeps_time_origin_and_exact_derivatives():
    # x=1+2t+3t², y=-t, z=2, yaw=0.2+0.1t on [10,12].
    coefficients = np.array([[[1, 2, 3], [0, -1, 0], [2, 0, 0], [0.2, 0.1, 0]]])
    trajectory = Trajectory(10.0, [2.0], coefficients)
    point = trajectory.sample(11.0)
    np.testing.assert_allclose(point["position"], [6, -1, 2])
    np.testing.assert_allclose(point["velocity"], [8, -1, 0])
    np.testing.assert_allclose(point["acceleration"], [6, 0, 0])
    assert point["yaw"] == pytest.approx(0.3)
    assert trajectory.end_time == 12
    with pytest.raises(ValueError, match="valid"):
        trajectory.sample(12.01)
    with pytest.raises(ValueError, match="valid"):
        trajectory.sample(9.99)


def test_piecewise_reference_samples_future_without_repeating_current_point():
    coefficients = np.zeros((2, 4, 2))
    coefficients[0, 0] = [0, 1]
    coefficients[1, 0] = [1, 2]
    trajectory = Trajectory(0.0, [1.0, 2.0], coefficients)
    result = trajectory.sample_many(np.array([0.5, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result["position"][:, 0], [0.5, 1, 3, 5])
    np.testing.assert_allclose(result["velocity"][:, 0], [1, 2, 2, 2])


@pytest.mark.parametrize("durations", [[0], [-1], [float("nan")]])
def test_invalid_trajectory_duration_cannot_enter_execution(durations):
    with pytest.raises(ValueError):
        Trajectory(0.0, durations, np.zeros((1, 4, 2)))


def test_waypoint_and_motion_command_are_distinct_physical_interfaces():
    waypoint = Waypoint([[1, 2, 3]], tolerance=0.5)
    assert waypoint.positions.shape == (1, 3)
    command = MotionCommand("velocity_yaw", [1, 2, 3, 0])
    np.testing.assert_array_equal(command.values, [1, 2, 3, 0])
    with pytest.raises(ValueError):
        MotionCommand("attitude_thrust", [0, 0, 1])
    with pytest.raises(ValueError):
        MotionCommand("trajectory", [1, 2, 3])
    with pytest.raises(ValueError):
        Waypoint([[1, 2, np.nan]], tolerance=0.5)


def test_bspline_adapter_preserves_upstream_curve_without_sampling_loss():
    from scipy.interpolate import BSpline

    knots = np.array([0, 0, 0, 0, 1, 2, 2, 2, 2], dtype=float)
    points = np.array([[0, 0, 1], [1, 1, 1], [2, -1, 2], [3, 1, 1], [4, 0, 1]])
    trajectory = Trajectory.from_bspline(5.0, knots, points, degree=3)
    times = np.linspace(0, 2, 21)
    curve = BSpline(knots, points, 3)
    values = trajectory.sample_many(5 + times)
    for order, field in enumerate(("position", "velocity", "acceleration")):
        np.testing.assert_allclose(values[field], curve(times, nu=order), atol=1e-12)


def test_super_position_and_yaw_with_different_knots_remain_exact():
    from types import SimpleNamespace as NS

    from drone_playground.native.ros_trajectory import super_trajectory

    # ROS starts at simulation+1; xyz: x=t², z=1; yaw=t before t=1, then 1+2(t-1).
    stamp = NS(to_sec=lambda: 1.0)
    message = NS(
        piece_num_pos=1,
        order_pos=2,
        coef_pos_x=[1, 0, 0],
        coef_pos_y=[0, 0, 0],
        coef_pos_z=[0, 0, 1],
        time_pos=[2],
        start_WT_pos=stamp,
        type=4,
        YAW_TRAJ=4,
        piece_num_yaw=2,
        order_yaw=1,
        coef_yaw=[1, 0, 2, 1],
        time_yaw=[1, 1],
        start_WT_yaw=stamp,
    )
    trajectory = super_trajectory(message)
    values = trajectory.sample_many([0.5, 1, 1.5])
    np.testing.assert_allclose(values["position"][:, 0], [0.25, 1, 2.25])
    np.testing.assert_allclose(values["velocity"][:, 0], [1, 2, 3])
    np.testing.assert_allclose(values["acceleration"][:, 0], [2, 2, 2])
    np.testing.assert_allclose(values["yaw"], [0.5, 1, 2])
    assert trajectory.yaw_defined


def test_ego_does_not_invent_a_future_yaw_trajectory():
    from types import SimpleNamespace as NS

    from drone_playground.native.ros_trajectory import ego_trajectory

    message = NS(
        start_time=NS(to_sec=lambda: 1.0),
        knots=[0, 0, 1, 1],
        pos_pts=[NS(x=0, y=0, z=1), NS(x=1, y=0, z=1)],
        order=1,
    )
    assert not ego_trajectory(message).yaw_defined


def test_mpc_horizon_needs_real_future_and_explicit_missing_yaw():
    from drone_playground.execution.reference import reference_horizon

    curve = Trajectory(0, [2], np.array([[[0, 1], [0, 0], [1, 0], [0, 0]]]), yaw_defined=False)
    with pytest.raises(ValueError, match="yaw"):
        reference_horizon(curve, 0, [0, 0.5, 1])
    ref = reference_horizon(curve, 0, [0, 0.5, 1], yaw=0.4)
    np.testing.assert_allclose(ref["position"][:, 0], [0, 0.5, 1])
    np.testing.assert_allclose(ref["yaw"], 0.4)
    with pytest.raises(ValueError, match="valid"):
        reference_horizon(curve, 1, [0, 1, 2], yaw=0)
