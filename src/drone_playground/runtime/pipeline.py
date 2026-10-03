"""Schedule configured host modules; algorithms live outside this module."""

import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

from drone_playground.references import Trajectory, Waypoint
from drone_playground.runtime.decision import output_kind, validate_reply_time


class Pipeline:
    def __init__(self, settings, directory, env):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.modules, self.contracts, self.latencies = [], [], []
        self.frequency = float(env.freq)
        self.periods = []
        self.output_kind = settings["output"]
        previous = None
        try:
            for index, spec in enumerate(settings["stages"]):
                component = dict(spec)
                component.pop("frequency_hz", None)
                implementation = component["_target_"]
                module = instantiate(component, _convert_="all")
                self.modules.append(module)
                if hasattr(module, "frequency_hz"):
                    module.frequency_hz = spec.get("frequency_hz", self.frequency)
                bind = getattr(module, "bind", None)
                if bind is not None:
                    bind(env, self.directory / str(index))
                if module.input_kind != previous:
                    raise ValueError(
                        f"Pipeline contract mismatch at stage {index}: {previous} -> {module.input_kind}"
                    )
                frequency = float(spec.get("frequency_hz", self.frequency))
                if not np.isfinite(frequency) or not 0 < frequency <= self.frequency:
                    raise ValueError(
                        "Module frequency must be finite, positive and no faster than execution"
                    )
                period = self.frequency / frequency
                if not np.isclose(period, round(period), rtol=0, atol=1e-8):
                    raise ValueError("Module frequency must divide the execution frequency")
                self.periods.append(round(period))
                previous = module.output_kind
                self.contracts.append(
                    dict(
                        stage=index,
                        implementation=implementation,
                        input=module.input_kind,
                        output=module.output_kind,
                        derivatives=module.derivatives,
                        frequency_hz=frequency,
                        provenance=getattr(module, "provenance", {}),
                    )
                )
            if not self.modules or previous != self.output_kind:
                raise ValueError("Pipeline final physical output differs from execution contract")
            (self.directory / "pipeline-contracts.json").write_text(
                json.dumps(self.contracts, indent=2)
            )
        except BaseException:
            self.close()
            raise
        self.plans, self.trajectories, self.commands = set(), set(), 0
        self.last_tick = None
        self.cached, self.next_ticks = [None] * len(self.modules), [0] * len(self.modules)
        self.module_calls = [0] * len(self.modules)

    def start(self, calibration, goal, limits=None, task_adapter=None):
        if limits:
            raise ValueError("Pipeline limits belong to each module's parameters")
        self.plans.clear()
        self.trajectories.clear()
        self.commands = 0
        self.last_tick = None
        self.cached, self.next_ticks = [None] * len(self.modules), [0] * len(self.modules)
        self.module_calls = [0] * len(self.modules)
        for module in self.modules:
            module.start(calibration, goal, None, task_adapter or {})

    def step(self, packet):
        import time

        started = time.monotonic()
        clock = float(packet["time"]) * self.frequency
        if not np.isfinite(clock) or clock < 0:
            raise ValueError("Pipeline clock must be finite and nonnegative")
        tick = round(clock)
        if abs(clock - tick) > 1e-4 or (self.last_tick is not None and tick <= self.last_tick):
            raise ValueError(
                "Pipeline clock must advance on execution ticks; reset starts a new episode"
            )
        self.last_tick = tick
        upstream, identity = None, []
        reply, stages = None, []
        try:
            for index, module in enumerate(self.modules):
                if tick >= self.next_ticks[index]:
                    reply = module.step(packet, upstream)
                    self.module_calls[index] += 1
                    # The cache keeps the module's own reply; the stage envelope
                    # is settled below before it is recorded.
                    self.cached[index] = reply
                    self.next_ticks[index] = tick + self.periods[index]
                else:
                    reply = self.cached[index]
                    cached_output = reply.get("output")
                    deadline = reply.get("valid_until")
                    if deadline is None and isinstance(cached_output, Waypoint):
                        # The module declares its own output envelope; read the
                        # deadline from it rather than the stage reply.
                        deadline = cached_output.valid_until
                    deadline = packet["time"] if deadline is None else deadline
                    if cached_output is not None and deadline < packet["time"]:
                        reply = {
                            **reply,
                            "output": None,
                            "decision_status": "no_plan",
                            "expired_stage": index,
                            # The envelope of the output just invalidated, so a
                            # caller still sees when the last plan was made.
                            "generated_at": getattr(
                                cached_output,
                                "generated_at",
                                reply.get("generated_at"),
                            ),
                            "valid_until": deadline,
                        }
                upstream = reply.get("output")
                if upstream is None:
                    stages.append(dict(reply, stage=index))
                    break
                if output_kind(upstream) != module.output_kind:
                    raise ValueError("Module violated its declared physical output")
                # A module that returns its own envelope expresses generation
                # time through the output itself and declares no producing tick
                # of its own; its stage reply then mirrors that output. The
                # output's envelope is the authority, so it is validated here
                # rather than trusting whatever the module dict claimed.
                if isinstance(upstream, Waypoint):
                    reply = {
                        **reply,
                        "generated_at": upstream.generated_at,
                        "valid_until": upstream.valid_until,
                    }
                validate_reply_time(reply, upstream, packet["time"])
                stages.append(dict(reply, stage=index))
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
                        reference["yaw"] = Rotation.from_quat(packet["quaternion"]).as_euler("xyz")[
                            2
                        ]
                    self.trajectories.add(plan_id)
            return dict(
                reply,
                output=upstream,
                reference=reference,
                trajectory=curve,
                stages=stages,
                plan_id=plan_id,
                commands=self.commands,
                plans=len(self.plans),
                trajectories=len(self.trajectories),
            )
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
            raise RuntimeError(
                "Pipeline cleanup failed: " + "; ".join(map(str, errors))
            ) from errors[0]
