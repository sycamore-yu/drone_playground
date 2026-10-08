"""Schedule configured host modules; algorithms live outside this module."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from hydra.utils import instantiate

from drone_playground.references import Trajectory
from drone_playground.runtime.decision import Decision, output_kind, validate_decision


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
                        f"Pipeline contract mismatch at stage {index}: "
                        f"{previous} -> {module.input_kind}"
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
                    if reply.output is not None and reply.valid_until < packet["time"]:
                        reply = replace(
                            reply,
                            output=None,
                            status="no_plan",
                            diagnostics={**reply.diagnostics, "expired_stage": index},
                        )
                if not isinstance(reply, Decision):
                    raise TypeError("A pipeline module must return Decision")
                upstream = reply.output
                stages.append(reply)
                if upstream is None:
                    break
                if output_kind(upstream) != module.output_kind:
                    raise ValueError("Module violated its declared physical output")
                validate_decision(reply, packet["time"])
                identity.append(f"{index}:{reply.plan_id}")
            plan_id = "/".join(identity)
            if upstream is not None:
                self.plans.add(plan_id)
                self.commands += 1
                if isinstance(upstream, Trajectory):
                    self.trajectories.add(plan_id)
            return replace(
                reply,
                output=upstream,
                plan_id=plan_id,
                stages=tuple(stages),
                diagnostics={
                    **reply.diagnostics,
                    "commands": self.commands,
                    "plans": len(self.plans),
                    "trajectories": len(self.trajectories),
                },
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
