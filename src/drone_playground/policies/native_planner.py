"""Process boundary for native ROS planners; the learning runtime never imports ROS."""
from __future__ import annotations

import hashlib
import json
import selectors
import subprocess
import time
from pathlib import Path

import numpy as np

DEFAULT_CONTAINER = "drone-playground-ros1"
RUNTIME_ROOT = "/opt/drone_playground"


def runtime_setup(method):
    ego = RUNTIME_ROOT + "/planners/ego/devel/setup.bash"
    if method == "ego":
        return "source " + ego
    super_setup = RUNTIME_ROOT + "/planners/super/devel/setup.bash"
    return "source " + super_setup


class NativePlanner:
    output_kind = "trajectory"

    @staticmethod
    def install_worker(source, directory, container):
        """Freeze the exact worker once per run; later code edits cannot affect episodes."""
        payload = Path(source).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        target = Path(directory) / "ros_bridge.py"
        target.write_bytes(payload)
        remote = RUNTIME_ROOT + "/bridge/bridge-" + digest + ".py"
        subprocess.run(["docker", "cp", str(target), container + ":" + remote], check=True)
        (Path(directory) / "ros_bridge.sha256").write_text(digest + "\n")
        return remote

    def __init__(self, method, directory, container=DEFAULT_CONTAINER, port=11325,
                 worker_path=RUNTIME_ROOT + "/bridge/ros_bridge.py"):
        if method not in ("ego", "super"):
            raise ValueError("Native planner must be ego or super")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / "bridge.log").open("w")
        self.process = subprocess.Popen(
            ["docker", "exec", "-i", container, "bash", "-c",
             runtime_setup(method) + ' && exec python3 '
             + worker_path + ' --method ' + method + ' --port ' + str(int(port))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.sequence = 0
        self.latencies = []

    def request(self, payload, timeout=10.0):
        self.sequence += 1
        payload = {**payload, "sequence": self.sequence}
        start = time.monotonic()
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(timeout):
            raise TimeoutError("Native planner RPC exceeded its wall-clock limit")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Native planner exited; see bridge.log")
        reply = json.loads(line)
        if reply.get("sequence") != self.sequence:
            raise RuntimeError("Native planner returned a stale episode/request")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        self.latencies.append(time.monotonic() - start)
        return reply

    def start(self, calibration, goal):
        reply = self.request({"op": "start", "calibration": calibration,
                              "goal": np.asarray(goal).tolist()}, timeout=45.0)
        (self.directory / "planner.launch").write_text(reply["launch_xml"])
        if reply.get("planner_yaml"):
            (self.directory / "planner.yaml").write_text(reply["planner_yaml"])
        return reply

    def step(self, packet):
        reply = self.request({"op": "step", **packet})
        reference = reply.get("reference")
        if reference is not None:
            numbers = np.r_[reference["position"], reference["velocity"],
                            reference["acceleration"], reference["yaw"], reference["time"]]
            age = packet["time"] - reference["time"]
            if not np.isfinite(numbers).all() or age < -1e-6 or age > 0.2:
                reply["reference"] = None
                reply["rejected_reference"] = True
        return reply

    def close(self):
        if self.process.poll() is None:
            try:
                self.request({"op": "close"}, timeout=10)
                self.process.wait(timeout=10)
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.selector.close()
        self.log.close()
