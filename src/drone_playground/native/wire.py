"""Convert validated physical values to the shared protobuf schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import MotionCommand, Trajectory, Waypoint
from .geometry import ConvexPolytope, PlannerGeometry, SafeFlightCorridor, TrajectoryPreview
from .proto import algorithm_pb2 as pb


def vec3(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Expected a finite xyz vector")
    return pb.Vec3(x=value[0], y=value[1], z=value[2])


def body_state(value):
    quaternion = np.asarray(value["quaternion"], dtype=float)
    if (
        quaternion.shape != (4,)
        or not np.isfinite(quaternion).all()
        or abs(np.linalg.norm(quaternion) - 1) > 1e-4
    ):
        raise ValueError("Body orientation must be a unit xyzw quaternion")
    result = pb.BodyState(
        position=vec3(value["position"]),
        velocity=vec3(value["velocity"]),
        quaternion_xyzw=quaternion,
    )
    if "angular_velocity" in value:
        result.angular_velocity.CopyFrom(vec3(value["angular_velocity"]))
    return result


def validate_step(request, capabilities=None):
    """Validate physical input before crossing into a foreign runtime."""
    state = request.state
    if (
        not request.HasField("state")
        or not state.HasField("position")
        or not state.HasField("velocity")
    ):
        raise ValueError("Step requires position, velocity and orientation")
    body_state(
        dict(
            position=[state.position.x, state.position.y, state.position.z],
            velocity=[state.velocity.x, state.velocity.y, state.velocity.z],
            quaternion=state.quaternion_xyzw,
        )
    )
    if state.HasField("angular_velocity"):
        vec3([state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z])
    if not np.isfinite(request.solve_budget_seconds) or request.solve_budget_seconds <= 0:
        raise ValueError("Solve budget must be positive and finite")
    if request.HasField("goal"):
        decode_output(request.goal)
    upstream = request.WhichOneof("upstream")
    if upstream:
        value = decode_output(getattr(request, upstream))
        if isinstance(value, Trajectory):
            value.sample(request.header.simulation_time)
    if capabilities is not None:
        available = {"state"}
        if state.HasField("angular_velocity"):
            available.add("angular_velocity")
        if request.HasField("goal"):
            available.add("goal")
        if request.HasField("measurement"):
            available.add(request.measurement.WhichOneof("data"))
        if upstream:
            kind = (
                value.kind
                if isinstance(value, MotionCommand)
                else ("trajectory" if isinstance(value, Trajectory) else "waypoint")
            )
            accepted = set(capabilities.accepted_upstream)
            if not accepted and "reference" in capabilities.required_inputs:
                accepted = {"trajectory"}
            if kind not in accepted:
                raise ValueError("Unsupported upstream physical interface")
            available.update(("upstream", upstream))
        missing = set(capabilities.required_inputs) - available
        if missing:
            raise ValueError(f"Native algorithm requires inputs: {sorted(missing)}")
    if not request.HasField("measurement"):
        return
    m = request.measurement
    if not np.isfinite(m.time) or not 0 <= m.time <= request.header.simulation_time + 1e-9:
        raise ValueError("Measurement time must be a past/current simulation time")
    kind = m.WhichOneof("data")
    if kind == "point_cloud":
        if m.frame not in ("world", "body") or m.point_cloud.channels not in (3, 4):
            raise ValueError("Invalid point cloud frame/channel count")
        count = m.point_cloud.count * m.point_cloud.channels
    elif kind == "depth":
        if m.frame != "camera_optical" or not m.depth.height or not m.depth.width:
            raise ValueError("Invalid depth frame/shape")
        count = m.depth.height * m.depth.width
        p = m.depth.camera_position
        body_state(
            dict(
                position=[p.x, p.y, p.z],
                velocity=[0, 0, 0],
                quaternion=m.depth.camera_quaternion_xyzw,
            )
        )
    else:
        raise ValueError("Measurement requires a supported data payload")
    data = getattr(m, kind).float32_le
    if len(data) != count * 4:
        raise ValueError("Measurement byte count does not match its shape")
    values = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(values).all() or (kind == "depth" and np.any(values < 0)):
        raise ValueError("Measurement must be finite (depth nonnegative)")


def encode_output(value):
    if isinstance(value, Trajectory):
        return pb.Trajectory(
            start_time=value.start_time,
            yaw_defined=value.yaw_defined,
            segments=[
                pb.PolynomialSegment(
                    duration=duration,
                    coefficient_count=coefficients.shape[-1],
                    coefficients=coefficients.ravel(),
                )
                for duration, coefficients in zip(value.durations, value.coefficients)
            ],
        )
    if isinstance(value, Waypoint):
        return pb.Waypoint(
            positions=[vec3(position) for position in value.positions], tolerance=value.tolerance
        )
    if isinstance(value, MotionCommand):
        return pb.MotionCommand(kind=value.kind, values=value.values)
    raise TypeError("Expected Trajectory, Waypoint or MotionCommand")


def decode_output(value):
    if isinstance(value, pb.Trajectory):
        if not value.segments:
            raise ValueError("Native trajectory has no segments")
        width = max(segment.coefficient_count for segment in value.segments)
        if not 1 <= width <= 16:
            raise ValueError("Unsupported trajectory polynomial degree")
        coefficients = np.zeros((len(value.segments), 4, width))
        for index, segment in enumerate(value.segments):
            if (
                segment.coefficient_count < 1
                or len(segment.coefficients) != 4 * segment.coefficient_count
            ):
                raise ValueError("Native trajectory coefficient shape mismatch")
            coefficients[index, :, : segment.coefficient_count] = np.asarray(
                segment.coefficients
            ).reshape(4, -1)
        return Trajectory(
            value.start_time,
            [segment.duration for segment in value.segments],
            coefficients,
            yaw_defined=value.yaw_defined,
        )
    if isinstance(value, pb.Waypoint):
        return Waypoint([[point.x, point.y, point.z] for point in value.positions], value.tolerance)
    if isinstance(value, pb.MotionCommand):
        return MotionCommand(value.kind, value.values)
    raise TypeError("Unsupported native physical output")


@dataclass(frozen=True)
class Decision:
    status: str
    output: Trajectory | Waypoint | MotionCommand | None
    plan_id: str
    generated_at: float
    valid_until: float
    diagnostics: dict
    explanation: str = ""
    sampled_reference: dict | None = None
    planner_geometry: PlannerGeometry | None = None


def encode_geometry(value):
    result = pb.PlannerGeometry(
        frame=value.frame, generated_at=value.generated_at, valid_until=value.valid_until
    )
    for corridor in value.corridors:
        wire = result.corridors.add(name=corridor.name)
        for polytope in corridor.polytopes:
            poly = wire.polytopes.add()
            if polytope.vertices is not None:
                poly.vertices.extend(vec3(v) for v in polytope.vertices)
            else:
                poly.halfspaces.extend(polytope.halfspaces.ravel())
    for preview in value.trajectories:
        result.trajectories.add(name=preview.name, trajectory=encode_output(preview.trajectory))
    return result


def decode_geometry(value):
    corridors = []
    for corridor in value.corridors:
        polytopes = []
        for poly in corridor.polytopes:
            if bool(poly.vertices) == bool(poly.halfspaces):
                raise ValueError("Polytope requires exactly one representation")
            if poly.vertices:
                polytopes.append(ConvexPolytope(vertices=[[v.x, v.y, v.z] for v in poly.vertices]))
            else:
                if len(poly.halfspaces) % 4:
                    raise ValueError("Halfspace planes must have four coefficients")
                polytopes.append(
                    ConvexPolytope(halfspaces=np.asarray(poly.halfspaces).reshape(-1, 4))
                )
        corridors.append(SafeFlightCorridor(corridor.name, tuple(polytopes)))
    return PlannerGeometry(
        value.generated_at,
        value.valid_until,
        tuple(corridors),
        tuple(TrajectoryPreview(v.name, decode_output(v.trajectory)) for v in value.trajectories),
        value.frame,
    )


def decode_decision(response, time, capabilities):
    geometry = None
    if response.HasField("planner_geometry"):
        geometry = decode_geometry(response.planner_geometry)
        if geometry.generated_at > time + 1e-9:
            raise ValueError("Planner geometry has a future origin")
    decision = response.decision
    if decision.status not in (pb.VALID, pb.NO_PLAN, pb.INFEASIBLE, pb.BUDGET_EXHAUSTED):
        raise ValueError("Native decision has no supported status")
    field = decision.WhichOneof("output")
    output = None
    if decision.status == pb.VALID:
        if field is None or not decision.plan_id:
            raise ValueError("A valid decision requires an actual output and plan identity")
        if (
            not np.isfinite([decision.generated_at, decision.valid_until]).all()
            or decision.generated_at > time + 1e-9
            or decision.valid_until < time
        ):
            raise ValueError("Native decision has a future origin or expired validity")
        output = decode_output(getattr(decision, field))
        kind = output.kind if isinstance(output, MotionCommand) else field
        if kind not in capabilities.outputs:
            raise ValueError("Native output violates its advertised capabilities")
        if isinstance(output, Trajectory) and (
            output.start_time > time + 1e-9 or decision.valid_until > output.end_time + 1e-9
        ):
            raise ValueError("Native trajectory does not cover its declared valid interval")
    elif field is not None:
        raise ValueError("An unsuccessful decision cannot carry an executable output")
    diagnostics = dict(response.diagnostics)
    if not all(np.isfinite(value) for value in diagnostics.values()):
        raise ValueError("Native diagnostics must be finite")
    sampled = None
    if response.HasField("sampled_reference"):
        sample = response.sampled_reference
        if not np.isfinite(sample.time) or sample.time > time + 1e-9:
            raise ValueError("Invalid sampled reference time")
        sampled = {
            name: np.array(
                [getattr(sample, name).x, getattr(sample, name).y, getattr(sample, name).z]
            )
            for name in ("position", "velocity", "acceleration")
        }
        sampled.update(yaw=sample.yaw, time=sample.time)
        if not np.isfinite(
            np.r_[sampled["position"], sampled["velocity"], sampled["acceleration"], sample.yaw]
        ).all():
            raise ValueError("Nonfinite sampled reference")
    return Decision(
        pb.DecisionStatus.Name(decision.status).lower(),
        output,
        decision.plan_id,
        decision.generated_at,
        decision.valid_until,
        diagnostics,
        decision.explanation,
        sampled,
        geometry,
    )
