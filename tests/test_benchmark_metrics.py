"""Check FlightBench units and successful-episode denominators on known paths."""

import numpy as np
import pytest

from drone_playground.benchmark.metrics import summarize, trajectory_metrics


def test_constant_speed_line():
    """A straight constant-speed flight has zero acceleration, jerk, and curvature."""
    t = np.linspace(0, 2, 101)
    p = np.stack((3 * t, t * 0, t * 0), axis=1)
    v = np.broadcast_to([3, 0, 0], p.shape)
    metrics = trajectory_metrics(t, p, v)
    assert metrics["path_length_metres"] == pytest.approx(6)
    assert metrics["average_speed_m_s"] == pytest.approx(3)
    assert metrics["average_acceleration_m_s3"] == pytest.approx(0, abs=1e-20)
    assert metrics["average_jerk_m_s5"] == pytest.approx(0, abs=1e-18)
    assert metrics["average_curvature_per_m"] == 0


def test_circle_normalizes_squared_derivatives_by_path_length():
    """A radius-two circle with angular speed three has known normalized metrics."""
    radius, omega = 2, 3
    t = np.linspace(0, 2 * np.pi / omega, 2001)
    angle = omega * t
    p = np.stack((radius * np.cos(angle), radius * np.sin(angle), t * 0), axis=1)
    v = np.stack((-radius * omega * np.sin(angle), radius * omega * np.cos(angle), t * 0), axis=1)
    metrics = trajectory_metrics(t, p, v)
    assert metrics["average_speed_m_s"] == pytest.approx(radius * omega, rel=1e-4)
    assert metrics["average_acceleration_m_s3"] == pytest.approx(radius * omega**3, rel=1e-4)
    assert metrics["average_jerk_m_s5"] == pytest.approx(radius * omega**5, rel=1e-4)
    assert metrics["average_curvature_per_m"] == pytest.approx(1 / radius, rel=1e-4)


def test_stationary_and_invalid_trajectories_do_not_fabricate_metrics():
    """Undefined normalization and nonfinite data remain explicitly unavailable."""
    t = np.arange(5)
    p = np.zeros((5, 3))
    metrics = trajectory_metrics(t, p, p)
    assert metrics["trajectory_metric_status"] == "zero_length"
    assert metrics["average_jerk_m_s5"] is None
    assert metrics["average_speed_m_s"] == 0
    p[1, 0] = np.nan
    assert trajectory_metrics(t, p, p)["trajectory_metric_status"] == "nonfinite_trajectory"
    with pytest.raises(ValueError, match="increase strictly"):
        trajectory_metrics([0, 0], np.zeros((2, 3)), np.zeros((2, 3)))


def test_failures_count_towards_success_rate_but_not_successful_flight_means():
    """Early failures cannot improve the reported flight-quality means."""
    report = summarize(
        [
            {"event": "SUCCESS", "average_speed_m_s": 4},
            {"event": "SUCCESS", "average_speed_m_s": 6},
            {"event": "COLLISION", "average_speed_m_s": 100},
        ]
    )
    assert report["success_rate"] == pytest.approx(2 / 3)
    assert report["successful_flight_metrics"]["average_speed_m_s"] == {"mean": 5, "count": 2}
    assert report["successful_flight_metrics"]["average_jerk_m_s5"] == {"mean": None, "count": 0}
    assert "passed" not in report
