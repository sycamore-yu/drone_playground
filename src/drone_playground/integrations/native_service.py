"""Algorithm-independent deployment entry for a physical-output gRPC service."""

import base64
import json
from pathlib import Path

from scipy.spatial.transform import Rotation

from drone_playground.native.client import NativeClient
from drone_playground.native.contracts import MotionCommand, Trajectory, Waypoint
from drone_playground.native.proto import algorithm_pb2 as pb
from drone_playground.native.wire import vec3


def packet_measurement(packet):
    if "depth" in packet:
        return pb.Measurement(
            time=packet["time"],
            frame="camera_optical",
            depth=pb.DepthImage(
                float32_le=base64.b64decode(packet["depth"], validate=True),
                width=packet["width"],
                height=packet["height"],
                camera_position=vec3(packet["camera_position"]),
                camera_quaternion_xyzw=packet["camera_quaternion"],
            ),
        )
    if "points" in packet:
        return pb.Measurement(
            time=packet["time"],
            frame="world",
            point_cloud=pb.PointCloud(
                float32_le=base64.b64decode(packet["points"], validate=True),
                count=packet["point_count"],
                channels=4,
            ),
        )
    return None


class NativeServicePlanner:
    def __init__(self, settings, directory):
        self.directory = Path(directory)
        deployment = settings["deployment"]
        self.client = NativeClient(
            settings["algorithm"],
            directory=directory,
            command=deployment.get("command"),
            address=deployment.get("address"),
            parameters=settings.get("parameters"),
        )
        self.output_kind = settings.get("output", "trajectory")
        if self.output_kind not in self.client.capabilities.outputs:
            self.client.close()
            raise ValueError("Service capabilities do not include the configured physical output")
        self.latencies = self.client.latencies
        self.plans, self.commands = set(), 0
        self.trajectories = set()
        self.goal = None
        self.measurement = None
        (self.directory / "service-provenance.json").write_text(
            json.dumps(self.client.provenance, indent=2)
        )

    def start(self, calibration, goal, limits=None, task_adapter=None):
        if limits:
            raise ValueError("Generic service parameters belong to method.parameters")
        self.goal = Waypoint([goal], 0.5)
        self.plans.clear()
        self.trajectories.clear()
        self.commands = 0
        self.measurement = None
        return self.client.reset(
            goal=self.goal, calibration=calibration, task=task_adapter or {}
        )

    def step(self, packet, upstream=None):
        if "goal" in packet:
            self.goal = Waypoint([packet["goal"]], 0.5)
        measurement = packet_measurement(packet)
        if measurement is not None:
            self.measurement = measurement
        decision = self.client.step(
            time=packet["time"],
            state=packet,
            measurement=self.measurement,
            goal=self.goal,
            upstream=upstream,
        )
        output, curve, reference = decision.output, None, None
        if output is not None:
            kind = output.kind if isinstance(output, MotionCommand) else (
                "trajectory" if isinstance(output, Trajectory) else "waypoint"
            )
            if kind != self.output_kind:
                raise ValueError("Native physical output differs from the configured command")
            self.plans.add(decision.plan_id)
            if isinstance(output, Trajectory):
                curve = output
                self.trajectories.add(decision.plan_id)
                reference = curve.sample(packet["time"])
                if not curve.yaw_defined:
                    reference["yaw"] = Rotation.from_quat(packet["quaternion"]).as_euler("xyz")[2]
            self.commands += 1
        return dict(
            output=output,
            plan_id=decision.plan_id,
            valid_until=decision.valid_until,
            reference=reference,
            trajectory=curve,
            commands=self.commands,
            plans=len(self.plans),
            trajectories=len(self.trajectories),
            decision_status=decision.status,
        )

    def close(self):
        self.client.close()


def create_native_planner(settings, directory, port, worker_path, env=None):
    if settings["implementation"] == "pipeline":
        from .pipeline import PipelinePlanner

        return PipelinePlanner(settings, directory, env)
    if settings["implementation"] == "native_service":
        return NativeServicePlanner(settings, directory)
    from .native_planner import NativePlanner

    return NativePlanner(settings["method"], directory, settings["container"], port, worker_path)
