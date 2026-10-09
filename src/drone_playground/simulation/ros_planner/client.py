"""Host client for the native ROS1 planners; no simulation or learning imports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import grpc
import numpy as np

from drone_playground.simulation.ros_planner import planner_pb2 as rpc
from drone_playground.simulation.ros_planner import planner_pb2_grpc as rpc_grpc


class RosPlannerError(RuntimeError):
    """A transport, planning, or trajectory-validity failure with no fallback command."""

    def __init__(self, status: str, detail: str):
        """Attach the transport or planning status to the failure detail."""
        super().__init__(f"{status}: {detail}")
        self.status = status


def _vector(values) -> rpc.Vec3:
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite vector of shape (3,)")
    return rpc.Vec3(x=values[0], y=values[1], z=values[2])


def _pose(position, quaternion) -> rpc.Pose:
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-4:
        raise ValueError("Expected a unit quaternion in wxyz order")
    return rpc.Pose(
        position=_vector(position), orientation=rpc.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    )


def _ns(seconds: float) -> int:
    if not math.isfinite(seconds) or not 0 <= seconds < 2**31:
        raise ValueError("Simulation time must be finite, nonnegative seconds below 2**31")
    return round(seconds * 1e9)


def cloud_measurement(
    points,
    time: float,
    *,
    available_time: float | None = None,
    position=(0.0, 0.0, 0.0),
    quaternion=(1.0, 0.0, 0.0, 0.0),
) -> rpc.Measurement:
    """Encode measured hit endpoints in the stated sensor frame, in metres.

    Deskew a scanning cloud before calling; the transform is a single rigid pose.
    Subtract the sensor position from world-frame endpoints and use its translation
    with identity rotation. Invalid/no-return points must be removed.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    return rpc.Measurement(
        timestamp_ns=_ns(time),
        available_ns=_ns(time if available_time is None else available_time),
        world_from_sensor=_pose(position, quaternion),
        cloud=rpc.PointCloud(points=[_vector(p) for p in points]),
    )


def depth_measurement(
    depth,
    time: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    position,
    quaternion,
    available_time: float | None = None,
) -> rpc.Measurement:
    """Encode optical +z depth in metres; zero is no return, pose maps optical to world."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("depth must be a finite nonnegative HxW array")
    return rpc.Measurement(
        timestamp_ns=_ns(time),
        available_ns=_ns(time if available_time is None else available_time),
        world_from_sensor=_pose(position, quaternion),
        depth=rpc.Depth(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            metres=depth.ravel(),
        ),
    )


@dataclass(frozen=True)
class RosTrajectory:
    """World-frame native trajectory samples, SI derivatives, and measured RPC timings."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    timings: dict[str, float]
    upstream_status: int
    sequence: int
    frame: str = "world"
    status: str = "SOLVED"

    @property
    def valid_from(self) -> float:
        """Return the first valid trajectory timestamp in simulation seconds."""
        return float(self.times[0])

    @property
    def valid_until(self) -> float:
        """Return the final valid trajectory timestamp in simulation seconds."""
        return float(self.times[-1])

    def sample(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate native PVA samples only during their declared validity interval."""
        # Compare on the wire's nanosecond clock, including rounded interval endpoints.
        if not _ns(self.valid_from) <= _ns(time) <= _ns(self.valid_until):
            raise RosPlannerError("STALE_OUTPUT", f"Trajectory is not valid at {time} s")
        return tuple(
            np.array([np.interp(time, self.times, values[:, i]) for i in range(3)])
            for values in (self.positions, self.velocities, self.accelerations)
        )


class RosPlanner:
    """One synchronous local gRPC session, with explicit deadlines and replan scheduling.

    A failed decision invalidates the cached trajectory. Timeout requires reset before
    another decision because a non-preemptible upstream solve may still be finishing.
    This host object is not thread-safe or JIT compatible; use one worker per client.
    """

    def __init__(
        self,
        planner: str,
        address: str = "127.0.0.1:50051",
        *,
        timeout: float = 5.0,
        replan_interval: float = 0.1,
        max_velocity: float = 3.0,
        max_acceleration: float = 4.0,
        sample_dt: float = 0.05,
        max_sensor_age: float = 0.25,
        robot_radius: float = 0.15,
    ):
        """Open and initialize one ROS Planner session with explicit limits."""
        if planner not in {"ego", "super"}:
            raise ValueError("planner must be 'ego' or 'super'")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive finite seconds")
        if not math.isfinite(replan_interval) or replan_interval < 0:
            raise ValueError("replan_interval must be nonnegative finite seconds")
        self.timeout = timeout
        self.replan_interval = replan_interval
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_receive_message_length", 32 * 1024 * 1024),
                ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ],
        )
        self._stub = rpc_grpc.RosPlannerStub(self._channel)
        self._closed = False
        self._needs_reset = False
        self._sequence = 0
        self._last_time = -math.inf
        self._last_plan_time = -math.inf
        self._goal = None
        self._trajectory = None
        self.last_decision = None
        try:
            self.session = self._invoke(
                self._stub.Initialize,
                rpc.InitializeRequest(
                    planner=planner,
                    max_velocity=max_velocity,
                    max_acceleration=max_acceleration,
                    sample_dt=sample_dt,
                    max_sensor_age=max_sensor_age,
                    robot_radius=robot_radius,
                ),
            )
        except Exception:
            self._channel.close()
            raise

    def _invoke(self, method, request):
        try:
            return method(request, timeout=self.timeout)
        except grpc.RpcError as error:
            self._trajectory = None
            if error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                self._needs_reset = True
            raise RosPlannerError(error.code().name, error.details()) from error

    def reset(self, seed: int = 0) -> None:
        """Reconstruct the native planner and map, allowing simulation time to restart.

        The adapter does not use the host seed; it never requests EGO's
        random polynomial initialization. The seed parameter matches the Planner API.
        """
        del seed
        self._check_open()
        self._trajectory = None
        self.session = self._invoke(self._stub.Reset, self.session)
        self._sequence = 0
        self._last_time = self._last_plan_time = -math.inf
        self._goal = None
        self._needs_reset = False
        self.last_decision = None

    def _check_open(self):
        if self._closed:
            raise RosPlannerError("CLOSED", "The native session is closed")

    def decide(
        self,
        state: rpc.State,
        goal: rpc.Goal,
        measurement: rpc.Measurement,
    ) -> RosTrajectory:
        """Always call the upstream planner; raise on transport, status, or stale output."""
        self._check_open()
        if self._needs_reset:
            raise RosPlannerError("RESET_REQUIRED", "Reset the worker after an RPC timeout")
        self._trajectory = None
        self._sequence += 1
        response = self._invoke(
            self._stub.Decide,
            rpc.DecideRequest(
                session=self.session,
                sequence=self._sequence,
                state=state,
                goal=goal,
                measurement=measurement,
            ),
        )
        self.last_decision = response
        if response.status != rpc.SOLVED:
            raise RosPlannerError(rpc.PlanningStatus.Name(response.status), response.detail)
        if (
            response.session.id != self.session.id
            or response.session.episode != self.session.episode
            or response.sequence != self._sequence
            or response.input_timestamp_ns != state.timestamp_ns
        ):
            raise RosPlannerError("STALE_OUTPUT", "Response does not match this episode/request")
        timestamps = np.array([s.timestamp_ns for s in response.samples], dtype=np.int64)
        if (
            len(timestamps) < 2
            or np.any(np.diff(timestamps) <= 0)
            or timestamps[0] != response.valid_from_ns
            or timestamps[-1] != response.valid_until_ns
            or not response.valid_from_ns <= state.timestamp_ns <= response.valid_until_ns
        ):
            raise RosPlannerError("STALE_OUTPUT", "Response trajectory has invalid time coverage")

        def vectors(name):
            return np.array(
                [
                    [getattr(s, name).x, getattr(s, name).y, getattr(s, name).z]
                    for s in response.samples
                ]
            )

        p, v, a = (vectors(name) for name in ("position", "velocity", "acceleration"))
        if not all(np.isfinite(values).all() for values in (p, v, a)):
            raise RosPlannerError("INVALID_OUTPUT", "Response contains nonfinite trajectory values")
        timings = {
            name: getattr(response.timings, name) for name in ("mapping_ms", "solve_ms", "total_ms")
        }
        self._trajectory = RosTrajectory(
            timestamps * 1e-9, p, v, a, timings, response.upstream_status, self._sequence
        )
        self._last_plan_time = state.timestamp_ns * 1e-9
        return self._trajectory

    def plan(
        self, observation: dict[str, Any], time: float, *, force_replan: bool = False
    ) -> RosTrajectory:
        """Plan from position/velocity/acceleration/quaternion/goal/measurement fields.

        Quaternion order is wxyz. ``measurement`` is made by cloud_measurement or
        depth_measurement. All vectors are single-drone, unbatched world-frame values.
        Extra simulation metadata is never serialized to the worker.
        """
        self._check_open()
        stamp = _ns(time)
        if time < self._last_time:
            self._trajectory = None
            raise RosPlannerError("STALE_INPUT", "Call reset before simulation time moves backward")
        self._last_time = time
        goal_vector = _vector(observation["goal"])
        goal_yaw = float(observation.get("goal_yaw", 0.0))
        goal_key = (goal_vector.x, goal_vector.y, goal_vector.z, goal_yaw)
        if (
            not force_replan
            and self._trajectory is not None
            and self._goal == goal_key
            and time - self._last_plan_time < self.replan_interval
        ):
            try:
                self._trajectory.sample(time)
            except RosPlannerError:
                self._trajectory = None
                raise
            return self._trajectory
        self._goal = goal_key
        return self.decide(
            rpc.State(
                timestamp_ns=stamp,
                pose=_pose(observation["position"], observation["quaternion"]),
                velocity=_vector(observation["velocity"]),
                acceleration=_vector(observation["acceleration"]),
            ),
            rpc.Goal(timestamp_ns=stamp, position=goal_vector, yaw=goal_yaw),
            observation["measurement"],
        )

    __call__ = plan

    def close(self) -> None:
        """Release the upstream planner, map, and client channel."""
        if not self._closed:
            try:
                self._invoke(self._stub.Close, self.session)
            finally:
                self._closed = True
                self._trajectory = None
                self._channel.close()

    def __enter__(self) -> RosPlanner:
        """Return the initialized Planner for a managed session."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release the Planner session and transport channel."""
        try:
            self.close()
        except RosPlannerError:
            if exc_type is None:
                raise
