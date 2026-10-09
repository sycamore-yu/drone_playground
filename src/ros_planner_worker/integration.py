#!/usr/bin/env python3
"""Executable sensor-to-upstream-solve test; no simulator, map file, or flight claims."""

import argparse
import json

import numpy as np

from drone_playground.simulation.ros_planner import (
    RosPlanner,
    RosPlannerError,
    cloud_measurement,
    depth_measurement,
)


def sensor_returns(obstacle: bool = False, obstacle_x: float = 1.6) -> np.ndarray:
    """Ray cast an enclosing room and optional central box from a synthetic LiDAR.

    Only the nearest hit per ray is sent to the planner. The box/room geometry
    exists exclusively here in the test's sensor generator.
    """
    azimuth, elevation = np.meshgrid(
        np.linspace(-np.pi, np.pi, 360, endpoint=False), np.linspace(-0.65, 0.65, 41)
    )
    rays = np.stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ),
        axis=-1,
    )
    rays = rays.reshape(-1, 3)
    origin = np.array([0.0, 0.0, 2.0])
    boxes = [(np.array([-6.0, -6.0, 0.0]), np.array([8.0, 6.0, 5.0]))]
    if obstacle:
        boxes.append((np.array([obstacle_x, -0.5, 0.0]), np.array([obstacle_x + 0.8, 0.5, 4.5])))
    distances = np.full(len(rays), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for low, high in boxes:
            near = np.minimum((low - origin) / rays, (high - origin) / rays)
            far = np.maximum((low - origin) / rays, (high - origin) / rays)
            enter, leave = near.max(axis=1), far.min(axis=1)
            distance = np.where(enter > 0, enter, leave)
            hit = (leave >= np.maximum(enter, 0)) & (distance > 0)
            distances = np.minimum(distances, np.where(hit, distance, np.inf))
    return origin + rays[np.isfinite(distances)] * distances[np.isfinite(distances), None]


def observation(
    points,
    time=0.0,
    position=(0.0, 0.0, 2.0),
    velocity=(0.0, 0.0, 0.0),
    acceleration=(0.0, 0.0, 0.0),
):
    """Build a permitted world-frame observation from measured sensor returns."""
    return dict(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        goal=(4.0, 0.0, 2.0),
        measurement=cloud_measurement(np.asarray(points) - position, time, position=position),
    )


def check_trajectory(traj, start):
    """Check solved samples and analytic derivatives from the real C++ planner."""
    assert traj.status == "SOLVED" and len(traj.times) > 5
    assert np.linalg.norm(traj.positions[0] - start) < 0.3
    assert np.linalg.norm(traj.positions[-1] - traj.positions[0]) > 0.5
    assert np.max(np.linalg.norm(traj.velocities, axis=1)) > 0.1
    assert np.max(np.linalg.norm(traj.accelerations, axis=1)) > 0.1
    assert all(
        np.isfinite(values).all()
        for values in (traj.positions, traj.velocities, traj.accelerations)
    )
    # Numerical differentiation checks the returned native derivatives independently.
    numeric_v = np.gradient(traj.positions, traj.times, axis=0)
    numeric_a = np.gradient(traj.velocities, traj.times, axis=0)
    assert np.max(np.abs(numeric_v[2:-2] - traj.velocities[2:-2])) < 0.15
    assert np.max(np.abs(numeric_a[2:-2] - traj.accelerations[2:-2])) < 0.5
    assert traj.timings["solve_ms"] > 0
    assert traj.timings["total_ms"] >= traj.timings["solve_ms"]
    try:
        traj.sample(traj.valid_until + 0.01)
    except RosPlannerError as error:
        assert error.status == "STALE_OUTPUT"
    else:
        raise AssertionError("Expired trajectory accepted")


def run(address, planner):
    """Exercise real ROS Planner solves, replanning, reset and failure handling."""
    clear = sensor_returns()
    blocked = sensor_returns(True)
    with RosPlanner(planner, address, timeout=30.0) as client:
        trajectory = client(observation(clear), 0.0)
        check_trajectory(trajectory, np.array([0.0, 0.0, 2.0]))
        cached = client(observation(clear, 0.02), 0.02)
        assert cached is trajectory
        p, v, a = trajectory.sample(0.2)
        replanned = client(observation(clear, 0.2, p, v, a), 0.2)
        check_trajectory(replanned, p)
        assert replanned.sequence == 2
        client.reset()
        obstacle_traj = client(observation(blocked), 0.0)
        check_trajectory(obstacle_traj, np.array([0.0, 0.0, 2.0]))
        # The optimizer must actually react to sensor obstacles.
        inside = (
            (obstacle_traj.positions[:, 0] >= 1.6)
            & (obstacle_traj.positions[:, 0] <= 2.4)
            & (abs(obstacle_traj.positions[:, 1]) <= 0.5)
            & (obstacle_traj.positions[:, 2] <= 4.5)
        )
        assert not inside.any(), "Returned native trajectory intersects sensed box"
        assert np.max(abs(obstacle_traj.positions[:, 1])) > 0.5
        client.reset()
        # Ray cast the same room through a real pinhole camera; optical z is world x.
        u, v = np.meshgrid(np.arange(64), np.arange(48))
        rays = np.stack((np.ones_like(u), -(u - 31.5) / 40.0, -(v - 23.5) / 40.0), axis=-1)
        origin = np.array([0.0, 0.0, 2.0])
        with np.errstate(divide="ignore"):
            far = np.maximum(
                (np.array([-6.0, -6.0, 0.0]) - origin) / rays,
                (np.array([8.0, 6.0, 5.0]) - origin) / rays,
            )
            depth = far.min(axis=-1).astype(np.float32)
        obs = observation(clear)
        obs["measurement"] = depth_measurement(
            depth,
            0.0,
            fx=40.0,
            fy=40.0,
            cx=31.5,
            cy=23.5,
            position=(0.0, 0.0, 2.0),
            quaternion=(0.5, -0.5, 0.5, -0.5),
        )
        depth_traj = client(obs, 0.0)
        check_trajectory(depth_traj, np.array([0.0, 0.0, 2.0]))
        try:
            client(observation(clear, 0.0), 1.0, force_replan=True)
        except RosPlannerError as error:
            assert error.status == "STALE_INPUT"
        else:
            raise AssertionError("Stale measurement accepted")
        client.reset()
        invalid = observation(clear)
        invalid["measurement"].ClearField("cloud")
        invalid["measurement"].depth.width = 2
        invalid["measurement"].depth.height = 2
        try:
            client(invalid, 0.0)
        except RosPlannerError as error:
            assert error.status == "INVALID_INPUT"
        else:
            raise AssertionError("Malformed depth accepted")
        client.timeout = 0.002
        try:
            client(observation(clear), 0.0, force_replan=True)
        except RosPlannerError as error:
            assert error.status == "DEADLINE_EXCEEDED"
        else:
            raise AssertionError("Native mapping/solve unexpectedly fit a 2 ms RPC budget")
        client.timeout = 30.0
        client.reset()
        moving = client(
            observation(clear, velocity=(0.4, 0.0, 0.0), acceleration=(0.2, 0.0, 0.0)), 0.0
        )
        check_trajectory(moving, np.array([0.0, 0.0, 2.0]))
        assert np.allclose(moving.velocities[0], [0.4, 0.0, 0.0], atol=0.03)
        assert np.allclose(moving.accelerations[0], [0.2, 0.0, 0.0], atol=0.03)
        if planner == "ego":
            client.reset()
            recovery_obs = observation(
                sensor_returns(True, obstacle_x=4.0), velocity=(2.4, 0.3, 0.08)
            )
            recovery_obs["goal"] = (14.0, 0.0, 2.0)
            recovered = client(recovery_obs, 0.0)
            check_trajectory(recovered, np.array([0.0, 0.0, 2.0]))
            inside = (
                (recovered.positions[:, 0] >= 4.0)
                & (recovered.positions[:, 0] <= 4.8)
                & (abs(recovered.positions[:, 1]) <= 0.5)
                & (recovered.positions[:, 2] <= 4.5)
            )
            assert not inside.any(), "Recovered native trajectory intersects sensed box"
            client.reset()
            empty = client(observation(np.empty((0, 3))), 0.0)
            check_trajectory(empty, np.array([0.0, 0.0, 2.0]))
            client.reset()
            obs["measurement"].depth.metres[:] = [0.0] * (64 * 48)
            empty_depth = client(obs, 0.0)
            check_trajectory(empty_depth, np.array([0.0, 0.0, 2.0]))
            client.reset()
            occupied_goal = observation(sensor_returns(True, obstacle_x=7.0))
            occupied_goal["goal"] = (14.0, 0.0, 2.0)
            shortened = client(occupied_goal, 0.0)
            check_trajectory(shortened, np.array([0.0, 0.0, 2.0]))
            assert 0.5 < shortened.positions[-1, 0] < 6.8
        client.timeout = 0.002
        try:
            client.reset()
        except RosPlannerError as error:
            assert error.status == "DEADLINE_EXCEEDED"
        else:
            raise AssertionError("Native reset unexpectedly fit a 2 ms RPC budget")
        client.timeout = 30.0
        client.reset()
        return dict(
            planner=planner,
            upstream_commit=client.session.upstream_commit,
            samples=len(trajectory.times),
            duration=trajectory.valid_until,
            clear_timings=trajectory.timings,
            replan_timings=replanned.timings,
            obstacle_timings=obstacle_traj.timings,
            depth_timings=depth_traj.timings,
            obstacle_lateral_excursion=float(np.max(abs(obstacle_traj.positions[:, 1]))),
            **(
                dict(
                    empty_samples=len(empty.times),
                    empty_depth_samples=len(empty_depth.times),
                    recovery_samples=len(recovered.times),
                    shortened_goal=shortened.positions[-1].tolist(),
                )
                if planner == "ego"
                else {}
            ),
        )


def main():
    """Run the selected real-planner integration checks and save their evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--planner", choices=("ego", "super", "both"), default="both")
    args = parser.parse_args()
    for planner in ("ego", "super") if args.planner == "both" else (args.planner,):
        print(json.dumps(run(args.address, planner)), flush=True)


if __name__ == "__main__":
    main()
