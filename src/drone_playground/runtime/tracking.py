"""Apply a method's physical output with the configured execution controller."""

import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.control.controllers.trajectory import TrajectoryTracking
from drone_playground.control.setpoints import Actuation, Setpoint
from drone_playground.references import Trajectory, Waypoint
from drone_playground.runtime.decision import validate_decision


class ExternalTracking:
    """Keep episode-local tracking state; never select algorithms by their names."""

    def __init__(self, env, state, directory):
        self.env, self.controller = env, env.controller
        self.name = type(self.controller).__name__
        self.hold = np.asarray(env.controller_observation(state)["pos"])
        self.fallback = TrajectoryTracking().bind(env.low, env.high)
        self.missing = self.short_horizon = self.consumed = self.clipped = 0
        self.waypoints, self.waypoint_index = None, 0
        initialize = getattr(self.controller, "initialize", None)
        if initialize is not None:
            initialize(env, state, directory)

    def command(self, reply, state, tick):
        body = self.env.controller_observation(state)
        data = state.pipeline_state.sim_data
        mass = float(np.asarray(data.params.mass).reshape(-1)[0])
        now = tick / self.env.freq
        output = reply.output
        if output is not None:
            validate_decision(reply, now)
        if isinstance(output, Setpoint | Actuation):
            values = np.asarray(self.controller.input_values(output))
            command = np.clip(values, np.asarray(self.env.low), np.asarray(self.env.high))
            if getattr(self.controller, "input_fields", ())[:3] == ("vx", "vy", "vz"):
                command[:3] *= min(
                    1.0, self.controller.max_speed / max(np.linalg.norm(command[:3]), 1e-9)
                )
            self.clipped += int(not np.array_equal(command, values))
            self.consumed += 1
            self.hold = np.asarray(body["pos"])
            return command
        sample = reply.sampled_reference if output is not None else None
        curve = output if isinstance(output, Trajectory) else None
        if isinstance(output, Waypoint):
            if self.controller.input_kind != "waypoint":
                raise TypeError("Configured controller does not accept a waypoint reference")
            if self.waypoints is None or not np.array_equal(output.positions, self.waypoints):
                self.waypoints, self.waypoint_index = output.positions, 0
            while (
                self.waypoint_index < len(output.positions) - 1
                and np.linalg.norm(body["pos"] - output.positions[self.waypoint_index])
                <= output.tolerance
            ):
                self.waypoint_index += 1
            sample = dict(
                position=output.positions[self.waypoint_index],
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=Rotation.from_quat(body["quat"]).as_euler("xyz")[2],
            )
        elif isinstance(output, Trajectory):
            if self.controller.input_kind != "trajectory":
                raise TypeError("Configured controller does not accept a timed trajectory")
            query = now + getattr(self.controller, "lead_seconds", 0.0)
            if getattr(
                self.controller, "reference_source", "trajectory"
            ) == "trajectory" or hasattr(self.controller, "lead_seconds"):
                sample = (
                    curve.sample(query)
                    if curve.start_time - 1e-9 <= query <= curve.end_time + 1e-9
                    else None
                )
            if sample is not None and not curve.yaw_defined:
                sample["yaw"] = Rotation.from_quat(body["quat"]).as_euler("xyz")[2]
        compute_trajectory = getattr(self.controller, "compute_reference", None)
        if compute_trajectory is not None and curve is not None:
            command = compute_trajectory(body, curve, now, tick)
            if command is None:
                self.short_horizon += 1
            else:
                self.hold = np.asarray(body["pos"])
                self.consumed += 1
                return np.clip(command, np.asarray(self.env.low), np.asarray(self.env.high))
        elif sample is not None and hasattr(self.controller, "command"):
            self.consumed += 1
            self.hold = np.asarray(body["pos"])
            return self.controller.command(body, sample, mass)
        self.missing += 1
        if getattr(self.controller, "input_fields", ())[:3] == ("vx", "vy", "vz"):
            return np.r_[np.zeros(3), Rotation.from_quat(body["quat"]).as_euler("xyz")[2]]
        if self.controller.input_kind == "rates":
            return np.asarray(self.controller.hover(data))
        return self.fallback.command(
            body,
            dict(
                position=self.hold,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=Rotation.from_quat(body["quat"]).as_euler("xyz")[2],
            ),
            mass,
        )

    def close(self):
        close = getattr(self.controller, "close", None)
        if close is not None:
            close()
