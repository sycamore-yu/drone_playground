"""Native outputs and observed states for replaying the downstream control chain."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

from drone_playground.control.setpoints import (
    AttitudeSetpoint,
    ForceTorque,
    MotorRPM,
    RateSetpoint,
    StateSetpoint,
)
from drone_playground.planning.corridors import SafeFlightCorridor, TrajectoryPreview
from drone_playground.references import Trajectory, Waypoint
from drone_playground.runtime.decision import Decision

_OUTPUTS = {
    kind.__name__: kind
    for kind in (
        Trajectory,
        Waypoint,
        StateSetpoint,
        AttitudeSetpoint,
        RateSetpoint,
        ForceTorque,
        MotorRPM,
        SafeFlightCorridor,
        TrajectoryPreview,
    )
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class NativeDecisionRecorder:
    """Record decisions before physics, deduplicating complete physical outputs.

    A produced command is not proof it was applied: join its tick to the separate
    post-transition trace. A null command means the controller did not return one.
    """

    def __init__(self, directory: Path, command_contract: dict):
        if not {"kind", "fields", "units", "frame"} <= command_contract.keys():
            raise ValueError(
                "Decision archive requires the execution controller's physical contract"
            )
        self.directory = Path(directory)
        self.command_contract = dict(command_contract)
        self.outputs = {}
        self.frames = 0
        self.last_time = -math.inf

    def __enter__(self):
        """Open the NativeDecisionRecorder context and return its recording handle."""
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = gzip.open(self.directory / "frames.jsonl.gz", "xt", encoding="utf-8")
        return self

    def _pack(self, value):
        if isinstance(value, Decision):
            return {
                "runtime_decision": {
                    field.name: self._pack(getattr(value, field.name)) for field in fields(value)
                }
            }
        if isinstance(value, tuple(_OUTPUTS.values())):
            data = dict(type=type(value).__name__, fields=self._pack(asdict(value)))
            digest = hashlib.sha256(_json(data).encode()).hexdigest()
            self.outputs.setdefault(digest, data)
            return {"physical_output_sha256": digest}
        if isinstance(value, dict):
            return {key: self._pack(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._pack(item) for item in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def record(self, tick, time, state, reply, command):
        if tick != self.frames or not math.isfinite(time) or time < 0 or time <= self.last_time:
            raise ValueError("Decision ticks must be contiguous with increasing simulation time")
        row = self._pack(dict(tick=tick, time=time, state=state, reply=reply, command=command))
        self.stream.write(_json(row) + "\n")
        self.stream.flush()
        self.frames += 1
        self.last_time = time

    def __exit__(self, error_type, error, traceback):
        """Close the NativeDecisionRecorder context and release its owned resources."""
        self.stream.close()
        with gzip.open(self.directory / "outputs.json.gz", "xt", encoding="utf-8") as handle:
            handle.write(_json(self.outputs))
        index = dict(
            schema_version=2,
            frames=self.frames,
            unique_outputs=len(self.outputs),
            command_kind=self.command_contract["kind"],
            command_contract=self.command_contract,
            interrupted=error_type is not None,
            error=None if error is None else repr(error),
            timing=(
                "pre-transition; produced commands require matching post-transition "
                "frames to prove application"
            ),
            files={
                name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
                for name in ("frames.jsonl.gz", "outputs.json.gz")
            },
        )
        temporary = self.directory / "index.json.tmp"
        temporary.write_text(json.dumps(index, indent=2) + "\n")
        temporary.replace(self.directory / "index.json")


def load_native_decisions(directory: Path):
    """Verify both files, then yield typed replies and pre-transition states."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if index["schema_version"] != 2:
        raise ValueError("Unsupported native decision archive schema")
    content = {}
    for name in ("frames.jsonl.gz", "outputs.json.gz"):
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != index["files"][name]:
            raise ValueError("Native decision archive digest mismatch: " + name)
        content[name] = gzip.decompress(data)
    stored = json.loads(content["outputs.json.gz"])
    outputs = {}
    for digest, value in stored.items():
        if hashlib.sha256(_json(value).encode()).hexdigest() != digest:
            raise ValueError("Native physical output digest mismatch")
        outputs[digest] = _OUTPUTS[value["type"]](**value["fields"])

    def unpack(value):
        if isinstance(value, dict):
            if set(value) == {"runtime_decision"}:
                data = {key: unpack(item) for key, item in value["runtime_decision"].items()}
                for key in ("stages", "corridors", "trajectory_previews"):
                    data[key] = tuple(data[key])
                return Decision(**data)
            if set(value) == {"physical_output_sha256"}:
                return outputs[value["physical_output_sha256"]]
            return {key: unpack(item) for key, item in value.items()}
        if isinstance(value, list):
            return [unpack(item) for item in value]
        return value

    lines = content["frames.jsonl.gz"].splitlines()
    if len(lines) != index["frames"] or len(outputs) != index["unique_outputs"]:
        raise ValueError("Native decision archive counts differ from its index")
    last_time = -math.inf
    for tick, line in enumerate(lines):
        row = unpack(json.loads(line))
        if row["tick"] != tick or not math.isfinite(row["time"]) or row["time"] <= last_time:
            raise ValueError("Native decision archive clock is not contiguous")
        last_time = row["time"]
        yield row
