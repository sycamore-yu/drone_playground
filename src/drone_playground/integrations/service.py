"""Connect a configured external algorithm to the common physical interface."""

import json
from dataclasses import replace
from pathlib import Path

from drone_playground.integrations.rpc.client import NativeClient
from drone_playground.integrations.sensors import packet_measurement
from drone_playground.references import Trajectory, Waypoint


class NativeServicePlanner:
    """Run any language implementing algorithm.proto, without algorithm-name dispatch."""

    def __init__(self, settings, directory=None, env=None):
        self.settings = settings
        self.input_kind = settings.get("input")
        self.derivatives = "none"
        self.client = None
        if directory is not None:
            self.bind(env, directory)

    def bind(self, env, directory):
        del env
        self.directory = Path(directory)
        deployment = self.settings["deployment"]
        self.client = NativeClient(
            self.settings["algorithm"],
            directory=directory,
            command=deployment.get("command"),
            address=deployment.get("address"),
            parameters=self.settings.get("parameters"),
        )
        self.output_kind = self.settings.get("output", "trajectory")
        if self.output_kind not in self.client.capabilities.outputs:
            self.close()
            raise ValueError("Service capabilities do not include the physical output")
        if (
            self.input_kind is not None
            and self.input_kind not in self.client.capabilities.accepted_upstream
        ):
            self.close()
            raise ValueError("Service does not accept its configured upstream input")
        self.latencies = self.client.latencies
        self.plans, self.trajectories, self.commands = set(), set(), 0
        self.goal = self.measurement = None
        (self.directory / "service-provenance.json").write_text(
            json.dumps(self.client.provenance, indent=2)
        )
        return self

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
        if decision.output is not None:
            if decision.output.kind != self.output_kind:
                raise ValueError("Service output differs from its configured physical type")
            self.plans.add(decision.plan_id)
            self.commands += 1
            if isinstance(decision.output, Trajectory):
                self.trajectories.add(decision.plan_id)
        return replace(
            decision,
            diagnostics={
                **decision.diagnostics,
                "commands": self.commands,
                "plans": len(self.plans),
                "trajectories": len(self.trajectories),
            },
        )

    def close(self):
        if self.client is not None:
            self.client.close()
