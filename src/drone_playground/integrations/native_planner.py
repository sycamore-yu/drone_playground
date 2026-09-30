"""Deployment adapter for the two pinned ROS planners, using the common client."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

from drone_playground.native.client import NativeClient
from drone_playground.native.contracts import Waypoint

from .native_service import packet_measurement

DEFAULT_CONTAINER = "drone-playground-ros1"
RUNTIME_ROOT = "/opt/drone_playground"


def runtime_setup(method):
    if method not in ("ego", "super"):
        raise ValueError("Unknown ROS adapter")
    return "source " + RUNTIME_ROOT + "/planners/" + method + "/devel/setup.bash"


class NativePlanner:
    output_kind = "trajectory"

    @staticmethod
    def install_worker(source, directory, container):
        """Freeze the adapter AND protocol once; later edits cannot affect a run."""
        source = Path(source)
        root = source.parents[2]
        files = {"worker.py": source.read_bytes()}
        package = root / "src/drone_playground"
        for path in [
            package / "__init__.py",
            package / "contracts.py",
            *(package / "native").rglob("*.py"),
            *(package / "native/proto").glob("*.proto"),
        ]:
            files[str(Path("drone_playground") / path.relative_to(package))] = path.read_bytes()
        digest = hashlib.sha256()
        snapshot = Path(directory) / "native-service"
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, data in sorted(files.items()):
            digest.update(name.encode() + b"\0" + data)
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        sha = digest.hexdigest()
        remote = RUNTIME_ROOT + "/bridge/bundle-" + sha
        subprocess.run(["docker", "exec", container, "mkdir", "-p", remote], check=True)
        subprocess.run(["docker", "cp", str(snapshot) + "/.", container + ":" + remote], check=True)
        (Path(directory) / "ros_bridge.sha256").write_text(sha + "\n")
        return remote + "/worker.py"

    def __init__(
        self,
        method,
        directory,
        container=DEFAULT_CONTAINER,
        port=11325,
        worker_path=RUNTIME_ROOT + "/bridge/ros_bridge.py",
        parameters=None,
    ):
        self.method, self.container, self.port = method, container, int(port)
        runtime_setup(method)  # Validate before constructing any process command.
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.worker_path, self.client = worker_path, None
        self.parameters = dict(parameters or {})
        if 'limits' in self.parameters:
            raise ValueError('Kinematic limits belong to method.limits')
        self.latencies = []

    def request(self, payload, timeout=10.0):
        if payload["op"] == "start":
            if self.client is not None:
                raise RuntimeError("Use a fresh adapter for each evaluation episode")
            info = json.loads(subprocess.check_output(["docker", "inspect", self.container]))[0]
            networks = info["NetworkSettings"]["Networks"]
            addresses = [v["IPAddress"] for v in networks.values() if v.get("IPAddress")]
            if len(addresses) != 1:
                raise RuntimeError("ROS runtime requires one private container network")
            rpc_port = self.port + 20000
            if not 1024 <= rpc_port <= 65535:
                raise ValueError("Native RPC port is out of range")
            args = [
                "python3",
                self.worker_path,
                "--method",
                self.method,
                "--port",
                str(self.port),
                "--address",
                "0.0.0.0:" + str(rpc_port),
            ]
            shell = (
                runtime_setup(self.method)
                + " && export PYTHONPATH="
                + shlex.quote(str(Path(self.worker_path).parent))
                + ':"${PYTHONPATH:-}" && exec '
                + shlex.join(args)
            )
            self.client = NativeClient(
                self.method,
                command=["docker", "exec", self.container, "bash", "-c", shell],
                address=addresses[0] + ":" + str(rpc_port),
                directory=self.directory,
                parameters={**self.parameters, "limits": payload["limits"]},
            )
            self.latencies = self.client.latencies
            reply = self.client.reset(
                goal=Waypoint([payload["goal"]], 0.5),
                calibration=payload["calibration"],
                task=payload["task_adapter"],
                timeout=timeout,
            )
            if reply.get("runtime_sha256"):
                reply["runtime_sha256"] = json.loads(reply["runtime_sha256"])
            return {"ready": True, **reply}
        if payload["op"] != "step" or self.client is None:
            raise RuntimeError("Start the ROS adapter before stepping")
        result = self.client.step(
            time=payload["time"],
            state=payload,
            measurement=packet_measurement(payload),
            goal=Waypoint([payload["goal"]], 0.5) if "goal" in payload else None,
            timeout=timeout,
        )
        return dict(
            reference=result.sampled_reference,
            trajectory=result.output,
            plan_id=result.plan_id,
            generated_at=result.generated_at,
            valid_until=result.valid_until,
            planner_geometry=result.planner_geometry,
            decision_status=result.status,
            **result.diagnostics,
        )

    def start(self, calibration, goal, limits=None, task_adapter=None):
        limits = limits or {
            "max_velocity_mps": 20.0,
            "max_acceleration_mps2": 3.0,
            "planning_horizon_m": 7.5,
        }
        if any(not np.isfinite(value) or value <= 0 for value in limits.values()):
            raise ValueError("Planner limits must be finite and positive")
        reply = self.request(
            {
                "op": "start",
                "calibration": calibration,
                "goal": np.asarray(goal).tolist(),
                "limits": limits,
                "task_adapter": task_adapter or {},
                "startup_timeout_s": 90.0,
            },
            timeout=110.0,
        )
        (self.directory / "planner.launch").write_text(reply["launch_xml"])
        if reply.get("runtime_sha256"):
            (self.directory / "runtime-identity.json").write_text(
                json.dumps(reply["runtime_sha256"], indent=2) + "\n"
            )
        if reply.get("planner_yaml"):
            (self.directory / "planner.yaml").write_text(reply["planner_yaml"])
        return reply

    def step(self, packet):
        reply = self.request({"op": "step", **packet})
        reference = reply.get("reference")
        if reference is not None:
            numbers = np.r_[
                reference["position"],
                reference["velocity"],
                reference["acceleration"],
                reference["yaw"],
                reference["time"],
            ]
            age = packet["time"] - reference["time"]
            if not np.isfinite(numbers).all() or age < -1e-6 or age > 0.2:
                reply["reference"] = None
                reply["rejected_reference"] = True
        return reply

    def close(self):
        if self.client is not None:
            self.client.close()
