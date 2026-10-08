"""Run the pinned ROS1 planners behind the same protocol as other external methods."""

import hashlib
import json
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from drone_playground.integrations.rpc.client import NativeClient
from drone_playground.integrations.sensors import packet_measurement
from drone_playground.references import Waypoint

RUNTIME_ROOT = "/opt/drone_playground"


def runtime_setup(method):
    """Resolve the ROS1 runtime setup required by an external planner."""
    if method not in ("ego", "super"):
        raise ValueError("Unknown ROS adapter")
    return "source " + RUNTIME_ROOT + "/planners/" + method + "/devel/setup.bash"


class RosPlanner:
    """Own deployment and ROS lifecycle; return a typed Decision without repackaging it."""

    output_kind, input_kind, derivatives = "trajectory", None, "none"

    def __init__(self, settings, directory=None, env=None):
        self.settings = settings
        self.method, self.container = settings["method"], settings["container"]
        self.port = int(settings["port"])
        self.parameters = dict(settings.get("parameters") or {})
        if "limits" in self.parameters:
            raise ValueError("Kinematic limits belong to method.limits")
        runtime_setup(self.method)
        self.input_kind = "waypoint" if self.parameters.get("use_upstream_waypoints") else None
        self.client = None
        self.latencies = []
        if directory is not None:
            self.bind(env, directory)

    def bind(self, env, directory):
        del env
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        return self

    def _install_worker(self):
        """Copy one run-owned adapter bundle; never rewrite a running service."""
        package = Path(__file__).resolve().parents[2]
        sources = [
            package / relative
            for relative in (
                "__init__.py",
                "references.py",
                "control/__init__.py",
                "control/setpoints.py",
                "runtime/__init__.py",
                "runtime/decision.py",
                "planning/__init__.py",
                "planning/corridors.py",
                "planning/waypoints.py",
            )
        ]
        sources += list((package / "integrations").rglob("*.py"))
        sources += list((package / "integrations/rpc/proto").glob("*.proto"))
        snapshot = self.directory / "native-service"
        snapshot.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        for source in sorted(sources):
            name = str(Path("drone_playground") / source.relative_to(package))
            data = source.read_bytes()
            digest.update(name.encode() + b"\0" + data)
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        remote = RUNTIME_ROOT + "/bridge/bundle-" + digest.hexdigest()
        subprocess.run(["docker", "exec", self.container, "mkdir", "-p", remote], check=True)
        subprocess.run(
            ["docker", "cp", str(snapshot) + "/.", self.container + ":" + remote], check=True
        )
        return remote

    def start(self, calibration, goal, limits=None, task_adapter=None):
        if self.client is not None:
            raise RuntimeError("Use a fresh ROS adapter for each evaluation episode")
        limits = limits if limits is not None else self.settings["limits"]
        if any(not np.isfinite(value) or value <= 0 for value in limits.values()):
            raise ValueError("Planner limits must be finite and positive")
        info = json.loads(subprocess.check_output(["docker", "inspect", self.container]))[0]
        addresses = [
            value["IPAddress"]
            for value in info["NetworkSettings"]["Networks"].values()
            if value.get("IPAddress")
        ]
        if len(addresses) != 1:
            raise RuntimeError("ROS runtime requires one private container network")
        rpc_port = self.port + 20000
        if not 1024 <= rpc_port <= 65535:
            raise ValueError("Native RPC port is out of range")
        remote = self._install_worker()
        args = [
            "python3",
            "-m",
            "drone_playground.integrations.ros1.worker",
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
            + shlex.quote(remote)
            + ':"${PYTHONPATH:-}" && exec '
            + shlex.join(args)
        )
        self.client = NativeClient(
            self.method,
            command=["docker", "exec", self.container, "bash", "-c", shell],
            address=addresses[0] + ":" + str(rpc_port),
            directory=self.directory,
            parameters={**self.parameters, "limits": limits},
        )
        self.latencies = self.client.latencies
        reply = self.client.reset(
            goal=Waypoint([goal], 0.5),
            calibration=calibration,
            task=task_adapter or {},
            timeout=110.0,
        )
        (self.directory / "planner.launch").write_text(reply["launch_xml"])
        if reply.get("runtime_sha256"):
            identity = json.loads(reply["runtime_sha256"])
            (self.directory / "runtime-identity.json").write_text(
                json.dumps(identity, indent=2) + "\n"
            )
        if reply.get("planner_yaml"):
            (self.directory / "planner.yaml").write_text(reply["planner_yaml"])
        return reply

    def step(self, packet, upstream=None):
        if self.client is None:
            raise RuntimeError("Start the ROS adapter before stepping")
        result = self.client.step(
            time=packet["time"],
            state=packet,
            measurement=packet_measurement(packet),
            goal=Waypoint([packet["goal"]], 0.5) if "goal" in packet else None,
            upstream=upstream,
        )
        reference = result.sampled_reference
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
                result = replace(
                    result,
                    sampled_reference=None,
                    diagnostics={**result.diagnostics, "rejected_reference": True},
                )
        return result

    def close(self):
        if self.client is not None:
            self.client.close()
