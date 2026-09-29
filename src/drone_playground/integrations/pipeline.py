"""Configured physical module chain on the host, shared by all native tasks.

JAX training is deliberately a separate execution backend. This host chain
does not pretend numpy values or stateful RPCs preserve policy derivatives.
"""

import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

from drone_playground.contracts import COMMANDS
from drone_playground.native.contracts import MotionCommand, Trajectory, Waypoint


def output_kind(value):
    if isinstance(value, Trajectory):
        return "trajectory"
    if isinstance(value, Waypoint):
        return "waypoint"
    if isinstance(value, MotionCommand):
        return value.kind
    raise TypeError("Pipeline output must have a physical interface")


def output_reply(value, time, identity, valid_until):
    return dict(output=value, plan_id=identity, valid_until=valid_until,
                decision_status="valid" if value is not None else "no_plan")


class GoalWaypoints:
    input_kind, output_kind, derivatives = None, "waypoint", "none"

    def start(self, calibration, goal, limits, task):
        self.goal = Waypoint([goal], 0.5)

    def step(self, packet, upstream):
        if "goal" in packet:
            self.goal = Waypoint([packet["goal"]], 0.5)
        return output_reply(self.goal, packet["time"], "goal", packet["time"] + 1.)

    def close(self):
        pass


class MinimumJerkPlanning:
    input_kind, output_kind, derivatives = "waypoint", "trajectory", "none"

    def __init__(self, **settings):
        self.settings = settings
        self.curve, self.waypoints = None, None
        self.count = 0

    def start(self, calibration, goal, limits, task):
        self.curve, self.waypoints, self.count = None, None, 0

    def step(self, packet, upstream):
        from drone_playground.methods.planners.minimum_jerk import minimum_jerk_path

        # Preserve a generated curve while its physical input is unchanged.
        # Rebuilding from the current state at every tick would never progress
        # through the time parameterization of a rest-to-rest segment.
        if self.curve is None or packet["time"] > self.curve.end_time or not np.array_equal(
            upstream.positions, self.waypoints
        ):
            self.curve = minimum_jerk_path(upstream, packet["position"], packet["velocity"],
                                           packet["time"], **self.settings)
            self.waypoints = upstream.positions
            self.count += 1
        return output_reply(self.curve, packet["time"], str(self.count), self.curve.end_time)

    def close(self):
        pass


class FrozenNeuralCommand:
    """Reuse an actual trained physical-command policy with its recorded decoder."""

    input_kind, derivatives = None, "none"

    def __init__(self, checkpoint, env):
        import jax

        from drone_playground.methods.neural import NeuralPolicy

        self.policy, self.env = NeuralPolicy.load(checkpoint), env
        metadata = self.policy.metadata
        self.output_kind = metadata["config"]["method"]["output"]
        if self.output_kind != env.controller.input_kind or metadata["observation_size"] != env.observation_size:
            raise ValueError("Frozen neural observation/command decoder differs from this environment")
        source = metadata["config"]["env"]
        observer = instantiate(source["observation"])
        # Equal vector lengths alone do not establish equal field meanings.
        if type(observer) is not type(env.observer) or vars(observer) != vars(env.observer):
            raise ValueError("Frozen neural observation field semantics differ")
        dynamics = source["execution"]["dynamics"]
        if (source["task"]["freq"] != env.freq or dynamics["drone"] != env.drone
                or dynamics["forward"] != env.dynamics):
            raise ValueError("Frozen neural clock/model action decoder differs")
        self.provenance = dict(checkpoint=str(Path(checkpoint).resolve()),
                               sha256=metadata["sha256"], step=metadata["step"],
                               observation=source["observation"], output=self.output_kind)
        self.infer = jax.jit(self.policy.act)
        self.count = 0

    def start(self, calibration, goal, limits, task):
        self.count = 0

    def step(self, packet, upstream):
        import jax.numpy as jnp

        self.count += 1
        value = MotionCommand(self.output_kind, self.env.physical_action(
            self.infer(jnp.asarray(packet["policy_observation"]))))
        return output_reply(value, packet["time"], str(self.count), packet["time"] + self.env.dt)

    def close(self):
        pass


class PipelinePlanner:
    def __init__(self, settings, directory, env):
        from .native_service import NativeServicePlanner

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.modules, self.contracts, self.latencies = [], [], []
        self.output_kind = settings["output"]
        previous = None
        try:
            for index, spec in enumerate(settings["stages"]):
                implementation = spec["implementation"]
                if implementation == "native_service":
                    module = NativeServicePlanner(spec, self.directory / str(index))
                    self.modules.append(module)
                    module.input_kind = spec.get("input")
                    caps = module.client.capabilities
                    module.derivatives = "none"
                    if module.input_kind is not None and module.input_kind not in caps.accepted_upstream:
                        raise ValueError("Native stage does not accept the configured upstream kind")
                    if module.input_kind is None and "upstream" in caps.required_inputs:
                        raise ValueError("Native source requires an upstream physical decision")
                else:
                    params = spec.get("parameters", {})
                    if implementation == "goal":
                        module = GoalWaypoints()
                    elif implementation == "minimum_jerk":
                        module = MinimumJerkPlanning(**params)
                    elif implementation == "frozen_neural":
                        module = FrozenNeuralCommand(spec["checkpoint"], env)
                    elif implementation == "python":
                        module = instantiate(params)
                    else:
                        raise ValueError("Unknown physical module implementation: " + implementation)
                    self.modules.append(module)
                if module.input_kind != previous or module.output_kind not in COMMANDS:
                    raise ValueError(f"Pipeline contract mismatch at stage {index}: {previous} -> {module.input_kind}")
                previous = module.output_kind
                self.contracts.append(dict(stage=index, implementation=implementation,
                                           input=module.input_kind, output=module.output_kind,
                                           derivatives=module.derivatives,
                                           frequency_hz=env.freq,
                                           provenance=getattr(module, "provenance", {})))
            if not self.modules or previous != self.output_kind:
                raise ValueError("Pipeline final physical output differs from execution contract")
            (self.directory / "pipeline-contracts.json").write_text(json.dumps(self.contracts, indent=2))
        except BaseException:
            self.close()
            raise
        self.plans, self.trajectories, self.commands = set(), set(), 0

    def start(self, calibration, goal, limits=None, task_adapter=None):
        if limits:
            raise ValueError("Pipeline limits belong to each module's parameters")
        self.plans.clear()
        self.trajectories.clear()
        self.commands = 0
        for module in self.modules:
            module.start(calibration, goal, None, task_adapter or {})

    def step(self, packet):
        import time

        started = time.monotonic()
        upstream, identity = None, []
        reply = None
        try:
            for index, module in enumerate(self.modules):
                reply = module.step(packet, upstream)
                upstream = reply.get("output")
                if upstream is None:
                    break
                if output_kind(upstream) != module.output_kind:
                    raise ValueError("Module violated its declared physical output")
                if not np.isfinite(reply["valid_until"]) or reply["valid_until"] < packet["time"]:
                    raise ValueError("Module returned expired physical output")
                identity.append(f"{index}:{reply['plan_id']}")
            plan_id = "/".join(identity)
            curve, reference = None, None
            if upstream is not None:
                self.plans.add(plan_id)
                self.commands += 1
                if isinstance(upstream, Trajectory):
                    curve = upstream
                    reference = curve.sample(packet["time"])
                    if not curve.yaw_defined:
                        reference["yaw"] = Rotation.from_quat(packet["quaternion"]).as_euler("xyz")[2]
                    self.trajectories.add(plan_id)
            return dict(reply, output=upstream, reference=reference, trajectory=curve,
                        plan_id=plan_id, commands=self.commands, plans=len(self.plans),
                        trajectories=len(self.trajectories))
        finally:
            self.latencies.append(time.monotonic() - started)

    def close(self):
        errors = []
        for module in reversed(self.modules):
            try:
                module.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError("Pipeline cleanup failed: " + "; ".join(map(str, errors))) from errors[0]
