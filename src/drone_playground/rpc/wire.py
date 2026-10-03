"""Convert validated physical values to the shared protobuf schema."""

from __future__ import annotations

import numpy as np

from drone_playground.control.setpoints import (
    AttitudeSetpoint,
    ForceTorque,
    MotorRPM,
    RateSetpoint,
    StateSetpoint,
    validate_setpoint,
)
from drone_playground.planning.corridors import (
    ConvexPolytope,
    SafeFlightCorridor,
    TrajectoryPreview,
)
from drone_playground.references import Trajectory, Waypoint
from drone_playground.rpc.proto import algorithm_pb2 as pb
from drone_playground.runtime.decision import Decision


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
        vec3(
            [
                state.angular_velocity.x,
                state.angular_velocity.y,
                state.angular_velocity.z,
            ]
        )
    if not np.isfinite(request.solve_budget_seconds) or request.solve_budget_seconds <= 0:
        raise ValueError("Solve budget must be positive and finite")
    if request.HasField("goal"):
        decode_output(request.goal)
    upstream = request.WhichOneof("upstream")
    if upstream:
        value = decode_output(getattr(request, upstream))
        now = request.header.simulation_time
        if isinstance(value, Trajectory):
            value.sample(now)
        elif isinstance(value, Waypoint) and (
            value.generated_at > now + 1e-9 or (value.valid_until and value.valid_until < now)
        ):
            # An upstream goal outside its own window is stale input, not a
            # usable setpoint; the caller must resend a current decision. A zero
            # valid_until means the producer declared no horizon.
            raise ValueError("Upstream waypoint is outside its declared time window")
    if capabilities is not None:
        available = {"state"}
        if state.HasField("angular_velocity"):
            available.add("angular_velocity")
        if request.HasField("goal"):
            available.add("goal")
        if request.HasField("measurement"):
            available.add(request.measurement.WhichOneof("data"))
        if upstream:
            kind = value.kind
            accepted = set(capabilities.accepted_upstream)
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
        if m.frame not in ("world", "body") or m.point_cloud.channels not in (
            3,
            4,
        ):
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
            frame=value.frame,
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
            positions=[vec3(position) for position in value.positions],
            tolerance=value.tolerance,
            generated_at=value.generated_at,
            valid_until=value.valid_until,
        )
    validate_setpoint(value)
    if isinstance(value, StateSetpoint):
        result = pb.StateSetpoint()
        for name in ("position", "velocity", "acceleration"):
            field = getattr(value, name)
            if field is not None:
                getattr(result, name).CopyFrom(vec3(field))
        for name in ("yaw", "yaw_rate"):
            field = getattr(value, name)
            if field is not None:
                setattr(result, name, float(field))
        return result
    if isinstance(value, AttitudeSetpoint):
        return pb.AttitudeSetpoint(rpy=vec3(value.rpy), thrust=float(value.thrust))
    if isinstance(value, RateSetpoint):
        return pb.RateSetpoint(thrust=float(value.thrust), body_rates=vec3(value.body_rates))
    if isinstance(value, ForceTorque):
        return pb.ForceTorque(thrust=float(value.thrust), torque=vec3(value.torque))
    if isinstance(value, MotorRPM):
        return pb.MotorRPM(rpm=np.asarray(value.rpm))
    raise TypeError("Expected a typed Reference, Setpoint or Actuation")


def output_field(value):
    """Map physical type to its protobuf oneof field, without an action registry."""
    return "state_setpoint" if isinstance(value, StateSetpoint) else value.kind


def output_kinds():
    """Read supported transport kinds from the generated schema itself."""
    return {
        "state" if field.name == "state_setpoint" else field.name
        for field in pb.Decision.DESCRIPTOR.oneofs_by_name["output"].fields
    }


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
            frame=value.frame,
        )
    if isinstance(value, pb.Waypoint):
        return Waypoint(
            [[point.x, point.y, point.z] for point in value.positions],
            value.tolerance,
            generated_at=value.generated_at,
            valid_until=value.valid_until,
        )

    def read_vector(name):
        if not value.HasField(name):
            raise ValueError(f"Physical output requires {name}")
        field = getattr(value, name)
        return np.array([field.x, field.y, field.z])

    if isinstance(value, pb.StateSetpoint):
        result = StateSetpoint(
            **{
                name: read_vector(name)
                for name in ("position", "velocity", "acceleration")
                if value.HasField(name)
            },
            **{name: getattr(value, name) for name in ("yaw", "yaw_rate") if value.HasField(name)},
        )
    elif isinstance(value, pb.AttitudeSetpoint):
        result = AttitudeSetpoint(rpy=read_vector("rpy"), thrust=value.thrust)
    elif isinstance(value, pb.RateSetpoint):
        result = RateSetpoint(thrust=value.thrust, body_rates=read_vector("body_rates"))
    elif isinstance(value, pb.ForceTorque):
        result = ForceTorque(thrust=value.thrust, torque=read_vector("torque"))
    elif isinstance(value, pb.MotorRPM):
        result = MotorRPM(rpm=np.array(value.rpm))
    else:
        raise TypeError("Unsupported native physical output")
    validate_setpoint(result)
    return result


def encode_corridor(value):
    result = pb.SafeFlightCorridor(
        name=value.name,
        frame=value.frame,
        generated_at=value.generated_at,
        valid_until=value.valid_until,
    )
    for polytope in value.polytopes:
        poly = result.polytopes.add()
        if polytope.vertices is not None:
            poly.vertices.extend(vec3(v) for v in polytope.vertices)
        else:
            poly.halfspaces.extend(polytope.halfspaces.ravel())
    return result


def decode_corridor(value):
    polytopes = []
    for poly in value.polytopes:
        if bool(poly.vertices) == bool(poly.halfspaces):
            raise ValueError("Polytope requires exactly one representation")
        if poly.vertices:
            polytopes.append(ConvexPolytope(vertices=[[v.x, v.y, v.z] for v in poly.vertices]))
        else:
            if len(poly.halfspaces) % 4:
                raise ValueError("Halfspace planes must have four coefficients")
            polytopes.append(ConvexPolytope(halfspaces=np.asarray(poly.halfspaces).reshape(-1, 4)))
    return SafeFlightCorridor(
        value.name, tuple(polytopes), value.generated_at, value.valid_until, value.frame
    )


def encode_preview(value):
    return pb.TrajectoryPreview(
        name=value.name,
        trajectory=encode_output(value.trajectory),
        generated_at=value.generated_at,
        valid_until=value.valid_until,
        frame=value.frame,
    )


def decode_preview(value):
    return TrajectoryPreview(
        value.name,
        decode_output(value.trajectory),
        value.generated_at,
        value.valid_until,
        value.frame,
    )


def decode_decision(response, time, capabilities):
    corridors = tuple(decode_corridor(value) for value in response.corridors)
    previews = tuple(decode_preview(value) for value in response.trajectory_previews)
    if any(value.generated_at > time + 1e-9 for value in (*corridors, *previews)):
        raise ValueError("Planning inspection has a future origin")
    decision = response.decision
    if decision.status not in (
        pb.VALID,
        pb.NO_PLAN,
        pb.INFEASIBLE,
        pb.BUDGET_EXHAUSTED,
    ):
        raise ValueError("Native decision has no supported status")
    field = decision.WhichOneof("output")
    output = None
    if decision.status == pb.VALID:
        if field is None or not decision.plan_id:
            raise ValueError("A valid decision requires an actual output and plan identity")
        if (
            not np.isfinite([decision.generated_at, decision.valid_until]).all()
            or decision.generated_at < 0
            or decision.generated_at > time + 1e-9
            or decision.valid_until < time
        ):
            raise ValueError("Native decision has a future origin or expired validity")
        output = decode_output(getattr(decision, field))
        if isinstance(output, Waypoint):
            # Decision is the authoritative output envelope. Mirror it into a
            # waypoint because a later native stage receives the raw waypoint
            # without the outer Decision wrapper.
            output = Waypoint(
                output.positions,
                output.tolerance,
                generated_at=decision.generated_at,
                valid_until=decision.valid_until,
            )
        kind = output.kind
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
                [
                    getattr(sample, name).x,
                    getattr(sample, name).y,
                    getattr(sample, name).z,
                ]
            )
            for name in ("position", "velocity", "acceleration")
        }
        sampled.update(yaw=sample.yaw, time=sample.time)
        if not np.isfinite(
            np.r_[
                sampled["position"],
                sampled["velocity"],
                sampled["acceleration"],
                sample.yaw,
            ]
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
        corridors,
        previews,
    )
