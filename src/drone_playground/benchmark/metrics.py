"""FlightBench trajectory metrics and success-conditioned summaries."""

from collections import Counter

import numpy as np

FLIGHT_METRICS = (
    "path_length_metres",
    "average_speed_m_s",
    "average_acceleration_m_s3",
    "average_jerk_m_s5",
    "average_curvature_per_m",
)


def trajectory_metrics(times, positions, velocities):
    """Measure realized motion; differentiate velocity on its recorded time grid."""
    t, p, v = (np.asarray(value, dtype=np.float64) for value in (times, positions, velocities))
    if t.ndim != 1 or p.shape != (len(t), 3) or v.shape != p.shape:
        raise ValueError("Trajectory requires time [T] and position/velocity [T,3]")
    result = dict.fromkeys(FLIGHT_METRICS)
    result.update(trajectory_metric_status="insufficient_samples", trajectory_samples=len(t))
    if not all(np.isfinite(value).all() for value in (t, p, v)):
        return {**result, "trajectory_metric_status": "nonfinite_trajectory"}
    if len(t) < 2:
        return result
    if np.any(np.diff(t) <= 0):
        raise ValueError("Trajectory times must increase strictly")
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    result.update(path_length_metres=length, average_speed_m_s=length / (t[-1] - t[0]))
    if length <= 1e-9:
        return {**result, "trajectory_metric_status": "zero_length"}
    if len(t) < 3:
        return result
    acceleration = np.gradient(v, t, axis=0, edge_order=2)
    acceleration_sq = np.sum(acceleration**2, axis=1)
    result["average_acceleration_m_s3"] = float(np.trapezoid(acceleration_sq, t) / length)
    speed_sq = np.sum(v**2, axis=1)
    curvature_distance_rate = np.divide(
        np.linalg.norm(np.cross(v, acceleration), axis=1),
        speed_sq,
        out=np.zeros_like(speed_sq),
        where=speed_sq > 1e-12,
    )
    result["average_curvature_per_m"] = float(np.trapezoid(curvature_distance_rate, t) / length)
    if len(t) >= 5:
        jerk = np.gradient(acceleration, t, axis=0, edge_order=2)
        result["average_jerk_m_s5"] = float(np.trapezoid(np.sum(jerk**2, axis=1), t) / length)
        result["trajectory_metric_status"] = "measured"
    return result


def attach_trajectory_metrics(episodes, traces):
    """Add metrics up to each episode's terminal sample, excluding padded scan steps."""
    for index, row in enumerate(episodes):
        times = traces["time"][:, index]
        keep = np.r_[True, np.diff(times) > 0]
        row.update(
            trajectory_metrics(
                times[keep], traces["position"][keep, index], traces["velocity"][keep, index]
            )
        )


def summarize(episodes):
    """Count every episode and aggregate flight quality over successful episodes."""
    if not episodes:
        raise ValueError("Cannot summarize an empty episode set")
    successful = [row for row in episodes if row["event"] == "SUCCESS"]
    metrics = {}
    for name in (*FLIGHT_METRICS, "position_rmse", "path_excess_percent"):
        values = [row[name] for row in successful if row.get(name) is not None]
        metrics[name] = {"mean": float(np.mean(values)) if values else None, "count": len(values)}
    return {
        "episodes": len(episodes),
        "successes": len(successful),
        "success_rate": len(successful) / len(episodes),
        "outcomes": dict(Counter(row["event"] for row in episodes)),
        "successful_flight_metrics": metrics,
    }
