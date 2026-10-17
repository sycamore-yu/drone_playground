"""The controller's fixed horizon may extend beyond a new native plan's endpoint."""

import numpy as np
import pytest

from drone_playground.simulation.methods import Trajectory, sample_planner_horizon


def test_valid_short_native_plan_has_explicit_stationary_terminal_reference():
    """Controller predictions never request samples outside the native validity window."""
    plan = Trajectory(
        times=np.array([32.0, 32.32, 32.64]),
        positions=np.array([[2.0, 0.0, 3.0], [2.5, 0.0, 3.0], [2.8, 0.0, 3.0]]),
        velocities=np.array([[1.0, 0.0, 0.0]] * 3),
        accelerations=np.zeros((3, 3)),
    )
    horizon = 32.0 + np.arange(1, 26) * 0.04
    reference, extension = sample_planner_horizon(plan, horizon)
    p, v, a = [np.asarray(array[0]) for array in reference]
    assert p.shape == v.shape == a.shape == (25, 3)
    np.testing.assert_allclose(p[-1], plan.positions[-1])
    np.testing.assert_allclose(v[-1], np.zeros(3))
    np.testing.assert_allclose(a[-1], np.zeros(3))
    assert extension == pytest.approx(0.36)
    assert plan.times[-1] == pytest.approx(32.64)
    with pytest.raises(ValueError, match="not valid"):
        plan.sample(32.92)


def test_expired_planner_output_cannot_be_reused():
    """A controller must replan rather than silently use an expired current setpoint."""
    plan = Trajectory(
        times=np.array([0.0, 0.2]),
        positions=np.zeros((2, 3)),
        velocities=np.zeros((2, 3)),
        accelerations=np.zeros((2, 3)),
    )
    with pytest.raises(RuntimeError, match="current controller step"):
        sample_planner_horizon(plan, np.array([0.3, 0.4]))


def test_incomplete_native_derivatives_fail_before_control():
    """Do not replace missing native velocity or acceleration with invented data."""
    plan = Trajectory(
        times=np.array([0.0, 0.2]),
        positions=np.zeros((2, 3)),
    )
    with pytest.raises(ValueError, match="position, velocity, and acceleration"):
        sample_planner_horizon(plan, np.array([0.1, 0.3]))
