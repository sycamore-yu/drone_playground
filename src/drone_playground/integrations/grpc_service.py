"""Algorithm-independent deployment entry for a physical-output gRPC service."""

import base64
import json
from pathlib import Path

from scipy.spatial.transform import Rotation

from drone_playground.references import Trajectory, Waypoint
from drone_playground.rpc.client import NativeClient
from drone_playground.rpc.proto import algorithm_pb2 as pb
from drone_playground.rpc.wire import vec3


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
    def __init__(self, settings, directory=None):
        self.settings = settings
        self.input_kind = settings.get("input")
        self.derivatives = "none"
        if directory is not None:
            self.bind(None, directory)

    def bind(self, env, directory):
        del env
        settings = self.settings
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
        if (
            self.input_kind is not None
            and self.input_kind not in self.client.capabilities.accepted_upstream
        ):
            self.client.close()
            raise ValueError("Native stage does not accept its declared upstream input")
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
        return self.client.reset(goal=self.goal, calibration=calibration, task=task_adapter or {})

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
            kind = output.kind
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
            generated_at=decision.generated_at,
            corridors=decision.corridors,
            trajectory_previews=decision.trajectory_previews,
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
    """Instantiate the configured runtime method; no algorithm-name dispatch."""
    from hydra.utils import get_method

    return get_method(settings["_target_"])(settings, directory, port, worker_path, env)


def build_service_method(settings, directory, port=None, worker_path=None, env=None):
    """Construct a typed native service method."""
    del port, worker_path, env
    return NativeServicePlanner(settings, directory)


def build_pipeline_method(settings, directory, port=None, worker_path=None, env=None):
    """Construct the scheduler from actual module constructors."""
    from drone_playground.runtime.pipeline import Pipeline

    del port, worker_path
    return Pipeline(settings, directory, env)


def build_ros_method(settings, directory, port=None, worker_path=None, env=None):
    """Freeze the external ROS implementation without changing environment code."""
    from drone_playground.integrations.ros1 import NativePlanner

    del env
    if worker_path is None:
        worker_path = NativePlanner.install_worker(
            Path(__file__).with_name("ros1_worker.py"),
            directory,
            settings["container"],
        )
    return NativePlanner(
        settings["method"],
        directory,
        settings["container"],
        port,
        worker_path,
        parameters=settings.get("parameters"),
    )
