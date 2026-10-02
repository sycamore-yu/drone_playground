"""Configured physical module chain on the host, shared by all native tasks.

JAX training is deliberately a separate execution backend. This host chain
does not pretend numpy values or stateful RPCs preserve policy derivatives.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

from drone_playground.actions.commands import COMMANDS, MotionCommand, Trajectory, Waypoint


def output_kind(value):
    if isinstance(value, Trajectory):
        return "trajectory"
    if isinstance(value, Waypoint):
        return "waypoint"
    if isinstance(value, MotionCommand):
        return value.kind
    raise TypeError("Pipeline output must have a physical interface")


def output_reply(value, generated_at, identity, valid_until):
    """Attach one simulation-clock validity envelope to an executable output."""
    generated_at, valid_until = float(generated_at), float(valid_until)
    if (
        not np.isfinite([generated_at, valid_until]).all()
        or generated_at < 0
        or valid_until < generated_at
    ):
        raise ValueError("Physical decision time envelope is invalid")
    if isinstance(value, Trajectory) and (
        value.start_time > generated_at + 1e-9
        or valid_until > value.end_time + 1e-9
    ):
        raise ValueError(
            "Trajectory does not cover its declared decision validity"
        )
    if isinstance(value, Waypoint):
        value = Waypoint(
            value.positions,
            value.tolerance,
            generated_at=generated_at,
            valid_until=valid_until,
        )
    return dict(
        output=value,
        plan_id=identity,
        generated_at=generated_at,
        valid_until=valid_until,
        decision_status="valid" if value is not None else "no_plan",
    )


def validate_reply_time(reply, value, now):
    """Reject stale/future module outputs before forwarding them downstream."""
    generated_at, valid_until = reply.get("generated_at"), reply.get(
        "valid_until"
    )
    if generated_at is None or valid_until is None:
        raise ValueError(
            "Physical output requires generated_at and valid_until"
        )
    generated_at, valid_until, now = (
        float(generated_at),
        float(valid_until),
        float(now),
    )
    if (
        not np.isfinite([generated_at, valid_until, now]).all()
        or generated_at < 0
        or generated_at > now + 1e-9
        or valid_until < now
    ):
        raise ValueError(
            "Module returned an invalid physical decision time envelope"
        )
    if isinstance(value, Trajectory) and (
        value.start_time > now + 1e-9 or valid_until > value.end_time + 1e-9
    ):
        raise ValueError(
            "Module trajectory does not cover its declared valid interval"
        )
    if isinstance(value, Waypoint) and (
        abs(value.generated_at - generated_at) > 1e-9
        or abs(value.valid_until - valid_until) > 1e-9
    ):
        raise ValueError("Waypoint and decision time envelopes disagree")


class GoalWaypoints:
    input_kind, output_kind, derivatives = None, "waypoint", "none"

    def start(self, calibration, goal, limits, task):
        # A task goal has no producing tick, so it stays valid until replaced.
        self.goal = Waypoint([goal], 0.5)

    def step(self, packet, upstream):
        if "goal" in packet and not np.array_equal(
            np.asarray(packet["goal"], dtype=float), self.goal.positions[0]
        ):
            # A replaced target is a new setpoint, so its producing tick is the
            # one that replaces it; an unchanged target keeps its own origin.
            self.goal = Waypoint(
                [packet["goal"]], 0.5, generated_at=float(packet["time"])
            )
        return output_reply(
            self.goal, self.goal.generated_at, "goal", packet["time"] + 1.0
        )

    def close(self):
        pass


class MinimumJerkPlanning:
    input_kind, output_kind, derivatives = "waypoint", "trajectory", "none"

    def __init__(self, **settings):
        self.settings = settings
        self.curve, self.waypoints = None, None
        self.generated_at = None
        self.count = 0

    def start(self, calibration, goal, limits, task):
        self.curve, self.waypoints, self.generated_at, self.count = (
            None,
            None,
            None,
            0,
        )

    def step(self, packet, upstream):
        from drone_playground.planning.minimum_jerk import minimum_jerk_path

        # Preserve a generated curve while its physical input is unchanged.
        # Rebuilding from the current state at every tick would never progress
        # through the time parameterization of a rest-to-rest segment.
        if (
            self.curve is None
            or packet["time"] > self.curve.end_time
            or not np.array_equal(upstream.positions, self.waypoints)
        ):
            self.curve = minimum_jerk_path(
                upstream,
                packet["position"],
                packet["velocity"],
                packet["time"],
                **self.settings,
            )
            self.waypoints = upstream.positions
            self.generated_at = float(packet["time"])
            self.count += 1
        return output_reply(
            self.curve, self.generated_at, str(self.count), self.curve.end_time
        )

    def close(self):
        pass


class FrozenNeuralCommand:
    """Reuse a frozen policy with a recorded physical output and input semantics."""

    input_kind, derivatives = None, "none"

    def __init__(self, checkpoint, env, frequency_hz=None, input_kind=None):
        import jax

        from drone_playground.networks.policies import NeuralPolicy

        self.policy, self.env = NeuralPolicy.load(checkpoint), env
        if input_kind not in (None, "trajectory"):
            raise ValueError(
                "Frozen neural upstream input must be an explicitly timed trajectory"
            )
        self.input_kind = input_kind
        self.frequency = float(
            env.freq if frequency_hz is None else frequency_hz
        )
        metadata = self.policy.metadata
        self.output_kind = metadata["config"]["method"]["output"]
        self.goal_source = metadata["config"]["method"].get(
            "goal_source", "task_goal"
        )
        if self.goal_source not in ("task_goal", "observation_reference"):
            raise ValueError("Frozen neural policy has an unknown goal source")
        self.decoder = None
        if metadata.get("physical_decoder"):
            from drone_playground.actions.decoders import PhysicalActionDecoder

            self.decoder = PhysicalActionDecoder(**metadata["physical_decoder"])
            if (
                self.decoder.kind != self.output_kind
                or self.decoder.action_size != metadata["action_size"]
            ):
                raise ValueError(
                    "Frozen neural physical decoder differs from checkpoint dimensions or type"
                )
        elif self.output_kind in ("waypoint", "trajectory"):
            raise ValueError(
                "Frozen neural geometric output requires an explicit physical decoder"
            )
        if (
            self.decoder is None
            and self.output_kind != env.controller.input_kind
        ) or (
            input_kind is None
            and metadata["observation_size"] != env.observation_size
        ):
            raise ValueError(
                "Frozen neural observation/command decoder differs from this environment"
            )
        source = metadata["config"]["env"]
        self.observer = observer = instantiate(source["observation"])
        # Equal vector lengths alone do not establish equal field meanings.
        if input_kind is None and (
            type(observer) is not type(env.observer)
            or vars(observer) != vars(env.observer)
        ):
            raise ValueError("Frozen neural observation field semantics differ")
        if (
            self.goal_source == "observation_reference"
            and observer.name != "state_reference"
        ):
            raise ValueError(
                "Frozen geometric goal requires the recorded reference observation"
            )
        dynamics = source["dynamics"]
        if (
            source["task"]["freq"] != self.frequency
            or dynamics["drone"] != env.drone
            or dynamics["forward"] != env.dynamics
        ):
            raise ValueError("Frozen neural clock/model action decoder differs")
        if input_kind == "trajectory":
            from drone_playground.environments.observations.state import TrackingObservation

            if (
                type(observer) is not TrackingObservation
                or observer.name != "state_reference"
                or observer.n_samples < 1
                or metadata["observation_size"] != observer.size
            ):
                raise ValueError(
                    "Upstream trajectory requires a recorded state_reference observation"
                )
            stride = self.frequency * observer.interval
            # The original tracking/racing encoders round fractional strides
            # differently. Require an unambiguous physical sampling clock.
            if (
                not np.isfinite(stride)
                or stride <= 0
                or not np.isclose(stride, round(stride), atol=1e-8, rtol=0)
            ):
                raise ValueError(
                    "Upstream reference interval must cover an integer number of policy ticks"
                )
            self.reference_offsets = (
                np.arange(observer.n_samples) * round(stride) / self.frequency
            )
        self.provenance = dict(
            checkpoint=str(Path(checkpoint).resolve()),
            sha256=metadata["sha256"],
            step=metadata["step"],
            observation=source["observation"],
            output=self.output_kind,
        )
        self.provenance["goal_source"] = self.goal_source
        if input_kind is not None:
            self.provenance["reference_source"] = "upstream_trajectory"
            self.provenance["reference_offsets_seconds"] = (
                self.reference_offsets.tolist()
            )
        if self.decoder is not None:
            self.provenance["physical_decoder"] = metadata["physical_decoder"]
            self.decode = jax.jit(self.decoder.decode)
        self.infer = jax.jit(self.policy.act)
        shape = jax.eval_shape(
            self.policy.act,
            jax.ShapeDtypeStruct((metadata["observation_size"],), np.float32),
        )
        if shape.shape != (metadata["action_size"],):
            raise ValueError(
                "Frozen neural parameter output differs from checkpoint dimensions"
            )
        self.count = 0

    def start(self, calibration, goal, limits, task):
        self.count = 0
        self.goal = goal

    def step(self, packet, upstream):
        import jax.numpy as jnp

        self.count += 1
        if self.input_kind == "trajectory":
            if not isinstance(upstream, Trajectory):
                raise ValueError(
                    "Frozen neural tracker requires upstream Trajectory"
                )
            times = packet["time"] + self.reference_offsets
            if (
                times[0] < upstream.start_time - 1e-9
                or times[-1] > upstream.end_time + 1e-9
            ):
                return dict(
                    output=None,
                    decision_status="no_plan",
                    reason="insufficient_reference_horizon",
                    required_reference_until=float(times[-1]),
                )
            reference = upstream.sample_many(times)
            fields = dict(
                pos="position",
                quat="quaternion",
                vel="velocity",
                ang_vel="angular_velocity",
            )
            states = SimpleNamespace(
                **{
                    field: jnp.asarray(packet[key])[None, None]
                    for field, key in fields.items()
                }
            )
            observation = self.observer(
                states, jnp.asarray(reference["position"])
            )
        else:
            observation = jnp.asarray(packet["policy_observation"])
        action = self.infer(observation)
        if "goal" in packet:
            self.goal = packet["goal"]
        if self.decoder is None:
            value = MotionCommand(
                self.output_kind, self.env.physical_action(action)
            )
        else:
            goal = (
                self.observer.reference_goal(observation)
                if self.goal_source == "observation_reference"
                else self.goal
            )
            decoded = self.decode(
                action, packet["position"], packet["velocity"], goal
            )
            value = self.decoder.message(decoded, packet["time"])
        return output_reply(
            value,
            packet["time"],
            str(self.count),
            packet["time"] + 1 / self.frequency,
        )

    def close(self):
        pass


class PipelinePlanner:

    def __init__(self, settings, directory, env):
        from drone_playground.integrations.grpc_service import NativeServicePlanner

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.modules, self.contracts, self.latencies = [], [], []
        self.frequency = float(env.freq)
        self.periods = []
        self.output_kind = settings["output"]
        previous = None
        try:
            for index, spec in enumerate(settings["stages"]):
                implementation = spec["implementation"]
                if implementation == "native_service":
                    module = NativeServicePlanner(
                        spec, self.directory / str(index)
                    )
                    self.modules.append(module)
                    module.input_kind = spec.get("input")
                    caps = module.client.capabilities
                    module.derivatives = "none"
                    if (
                        module.input_kind is not None
                        and module.input_kind not in caps.accepted_upstream
                    ):
                        raise ValueError(
                            "Native stage does not accept the configured upstream kind"
                        )
                    if (
                        module.input_kind is None
                        and "upstream" in caps.required_inputs
                    ):
                        raise ValueError(
                            "Native source requires an upstream physical decision"
                        )
                else:
                    params = spec.get("parameters", {})
                    if implementation == "goal":
                        module = GoalWaypoints()
                    elif implementation == "minimum_jerk":
                        module = MinimumJerkPlanning(**params)
                    elif implementation == "frozen_neural":
                        module = FrozenNeuralCommand(
                            spec["checkpoint"],
                            env,
                            spec.get("frequency_hz"),
                            spec.get("input"),
                        )
                    elif implementation == "python":
                        module = instantiate(params)
                    else:
                        raise ValueError(
                            "Unknown physical module implementation: "
                            + implementation
                        )
                    self.modules.append(module)
                if (
                    module.input_kind != previous
                    or module.output_kind not in COMMANDS
                ):
                    raise ValueError(
                        f"Pipeline contract mismatch at stage {index}: {previous} -> {module.input_kind}"
                    )
                frequency = float(spec.get("frequency_hz", self.frequency))
                if (
                    not np.isfinite(frequency)
                    or not 0 < frequency <= self.frequency
                ):
                    raise ValueError(
                        "Module frequency must be finite, positive and no faster than execution"
                    )
                period = self.frequency / frequency
                if not np.isclose(period, round(period), rtol=0, atol=1e-8):
                    raise ValueError(
                        "Module frequency must divide the execution frequency"
                    )
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
                raise ValueError(
                    "Pipeline final physical output differs from execution contract"
                )
            (self.directory / "pipeline-contracts.json").write_text(
                json.dumps(self.contracts, indent=2)
            )
        except BaseException:
            self.close()
            raise
        self.plans, self.trajectories, self.commands = set(), set(), 0
        self.last_tick = None
        self.cached, self.next_ticks = [None] * len(self.modules), [0] * len(
            self.modules
        )
        self.module_calls = [0] * len(self.modules)

    def start(self, calibration, goal, limits=None, task_adapter=None):
        if limits:
            raise ValueError(
                "Pipeline limits belong to each module's parameters"
            )
        self.plans.clear()
        self.trajectories.clear()
        self.commands = 0
        self.last_tick = None
        self.cached, self.next_ticks = [None] * len(self.modules), [0] * len(
            self.modules
        )
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
        if abs(clock - tick) > 1e-4 or (
            self.last_tick is not None and tick <= self.last_tick
        ):
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
                    raise ValueError(
                        "Module violated its declared physical output"
                    )
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
                        reference["yaw"] = Rotation.from_quat(
                            packet["quaternion"]
                        ).as_euler("xyz")[2]
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
