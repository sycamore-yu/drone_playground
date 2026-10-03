"""Python client for a local native algorithm process, independent of its language."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
import time as clock
import uuid
from pathlib import Path

import grpc
import numpy as np

from drone_playground.rpc.proto import algorithm_pb2 as pb
from drone_playground.rpc.wire import (
    body_state,
    decode_decision,
    encode_output,
    output_field,
    output_kinds,
    validate_step,
)


class NativeClient(contextlib.AbstractContextManager):
    def __init__(
        self,
        algorithm,
        *,
        command=None,
        address=None,
        parameters=None,
        directory=None,
        startup_timeout=30.0,
    ):
        if command is None and address is None:
            raise ValueError("Specify an owned process command or an existing service address")
        self._temporary = tempfile.TemporaryDirectory(prefix="drone-native-")
        self.directory = Path(directory or self._temporary.name)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.process = self.channel = self._log = None
        self._session = uuid.uuid4().hex
        self._episode = ""
        self._sequence = 0
        self._time = 0.0
        self._faulted = False
        self._closed = False
        self._initialized = False
        self.latencies = []
        self.address = address or "unix:" + str(Path(self._temporary.name) / "service.sock")
        try:
            if command is not None:
                self._log = (self.directory / "service.log").open("w")
                self.process = subprocess.Popen(
                    [str(arg).replace("{address}", self.address) for arg in command],
                    stdout=self._log,
                    stderr=self._log,
                    start_new_session=True,
                )
            self.channel = grpc.insecure_channel(
                self.address,
                options=[
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ],
            )
            ready = grpc.channel_ready_future(self.channel)
            deadline = clock.monotonic() + startup_timeout
            while not ready.done():
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        f"Native service exited during startup; see {self.directory / 'service.log'}"
                    )
                if clock.monotonic() >= deadline:
                    raise TimeoutError("Native service did not become ready")
                clock.sleep(0.02)
            request = pb.InitializeRequest(header=self._header(), algorithm=algorithm)
            request.parameters.update(parameters or {})
            reply = self._rpc("Initialize", request, pb.InitializeResponse, startup_timeout)
            self._initialized = True
            self.capabilities = reply.capabilities
            if self.capabilities.algorithm != algorithm or not self.capabilities.outputs:
                raise ValueError("Native service returned the wrong algorithm capabilities")
            if not set(self.capabilities.outputs) <= output_kinds():
                raise ValueError("Native service advertises unsupported physical outputs")
            self.provenance = dict(reply.provenance)
        except BaseException:
            self.close()
            raise

    def _header(self, time=None):
        self._sequence += 1
        return pb.Header(
            protocol_version=2,
            session_id=self._session,
            episode_id=self._episode,
            sequence=self._sequence,
            simulation_time=self._time if time is None else time,
        )

    def _rpc(self, name, request, response_class, timeout):
        if self._closed:
            raise RuntimeError("Native client is closed")
        call = self.channel.unary_unary(
            "/drone.native.v2.Algorithm/" + name,
            request_serializer=type(request).SerializeToString,
            response_deserializer=response_class.FromString,
        )
        started = clock.monotonic()
        try:
            reply = call(request, timeout=timeout)
            if reply.header != request.header:
                raise RuntimeError(
                    "Native response has a stale session/episode/request/time identity"
                )
            return reply
        except Exception:
            self._faulted = True
            raise
        finally:
            self.latencies.append(clock.monotonic() - started)

    def reset(
        self,
        *,
        state=None,
        goal=None,
        calibration=None,
        task=None,
        time=0.0,
        timeout=110.0,
    ):
        if self._faulted:
            raise RuntimeError("Native session is faulted; close it and create a new instance")
        if not np.isfinite(time) or time < 0:
            raise ValueError("Reset time must be finite and nonnegative")
        request = pb.ResetRequest(header=self._header(time))
        request.header.episode_id = uuid.uuid4().hex
        if state is not None:
            request.state.CopyFrom(body_state(state))
        if goal is not None:
            request.goal.CopyFrom(encode_output(goal))
        request.calibration.update(calibration or {})
        request.task.update(task or {})
        response = self._rpc("Reset", request, pb.Acknowledgement, timeout)
        self._episode = request.header.episode_id
        self._time = time
        return dict(response.artifacts)

    def step(
        self,
        *,
        time,
        state,
        measurement=None,
        goal=None,
        upstream=None,
        timeout=10.0,
        solve_budget_seconds=None,
    ):
        if self._faulted:
            raise RuntimeError("Native session is faulted; close it and create a new instance")
        if not self._episode:
            raise RuntimeError("Native client must reset an episode before step")
        if not np.isfinite(time) or time < self._time:
            raise ValueError("Simulation time must be finite and monotone")
        budget = timeout if solve_budget_seconds is None else solve_budget_seconds
        if not np.isfinite([budget, timeout]).all() or min(budget, timeout) <= 0:
            raise ValueError("Native solve/RPC budgets must be finite and positive")
        request = pb.StepRequest(
            header=self._header(time),
            state=body_state(state),
            solve_budget_seconds=budget,
        )
        available = {"state"}
        if request.state.HasField("angular_velocity"):
            available.add("angular_velocity")
        if measurement is not None:
            request.measurement.CopyFrom(measurement)
            available.add(measurement.WhichOneof("data"))
        if goal is not None:
            request.goal.CopyFrom(encode_output(goal))
            available.add("goal")
        if upstream is not None:
            kind = upstream.kind
            accepted = set(self.capabilities.accepted_upstream)
            if kind not in accepted:
                raise ValueError("Native service does not accept this upstream physical interface")
            field = output_field(upstream)
            getattr(request, field).CopyFrom(encode_output(upstream))
            available.update(("upstream", field))
        missing = set(self.capabilities.required_inputs) - available
        if missing:
            raise ValueError(f"Native algorithm requires inputs: {sorted(missing)}")
        validate_step(request)
        response = self._rpc("Step", request, pb.StepResponse, timeout)
        try:
            decision = decode_decision(response, time, self.capabilities)
        except Exception:
            self._faulted = True
            raise
        self._time = time
        return decision

    def close(self):
        if self._closed:
            return
        try:
            if self.channel is not None:
                with contextlib.suppress(Exception):
                    if self._initialized:
                        self._rpc(
                            "Close",
                            pb.CloseRequest(header=self._header()),
                            pb.Acknowledgement,
                            15.0,
                        )
                self.channel.close()
            if self.process is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        finally:
            self._closed = True
            if self._log is not None:
                self._log.close()
            self._temporary.cleanup()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
