"""A runtime decision keeps executable output separate from inspection evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

import numpy as np

from drone_playground.control.setpoints import Actuation, Setpoint
from drone_playground.planning.corridors import SafeFlightCorridor, TrajectoryPreview
from drone_playground.references import Reference, Trajectory, Waypoint


@dataclass(frozen=True)
class Decision:
    """One method output, identified and bounded by the simulation clock."""

    status: str
    output: Reference | Setpoint | Actuation | None
    plan_id: str
    generated_at: float
    valid_until: float
    diagnostics: dict
    explanation: str = ""
    sampled_reference: dict | None = None
    corridors: tuple[SafeFlightCorridor, ...] = ()
    trajectory_previews: tuple[TrajectoryPreview, ...] = ()
    stages: tuple[Decision, ...] = ()


def output_kind(value):
    """Classify an execution method output by its reference or setpoint type."""
    if isinstance(value, (*get_args(Reference), *get_args(Setpoint), *get_args(Actuation))):
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
        value.start_time > generated_at + 1e-9 or valid_until > value.end_time + 1e-9
    ):
        raise ValueError("Trajectory does not cover its declared decision validity")
    if isinstance(value, Waypoint):
        value = Waypoint(
            value.positions,
            value.tolerance,
            generated_at=generated_at,
            valid_until=valid_until,
        )
    return Decision(
        output=value,
        plan_id=identity,
        generated_at=generated_at,
        valid_until=valid_until,
        status="valid" if value is not None else "no_plan",
        diagnostics={},
    )


def validate_decision(reply, now):
    """Reject stale/future module outputs before forwarding them downstream."""
    value = reply.output
    generated_at, valid_until = reply.generated_at, reply.valid_until
    if generated_at is None or valid_until is None:
        raise ValueError("Physical output requires generated_at and valid_until")
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
        raise ValueError("Module returned an invalid physical decision time envelope")
    if isinstance(value, Trajectory) and (
        value.start_time > now + 1e-9 or valid_until > value.end_time + 1e-9
    ):
        raise ValueError("Module trajectory does not cover its declared valid interval")
    if isinstance(value, Waypoint) and (
        abs(value.generated_at - generated_at) > 1e-9 or abs(value.valid_until - valid_until) > 1e-9
    ):
        raise ValueError("Waypoint and decision time envelopes disagree")
