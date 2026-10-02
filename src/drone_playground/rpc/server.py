"""Small Python service adapter for existing runtimes such as ROS.

Pure C++ algorithms use ros_integrations/sdk/include/drone_native/algorithm.h. Both services
implement the same wire protocol and own one serialized algorithm instance.
"""

from __future__ import annotations

import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np

from drone_playground.rpc.proto import algorithm_pb2 as pb
from drone_playground.rpc.wire import decode_decision, validate_step


class Service:

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.session = self.episode = ""
        self.sequence = 0
        self.time = 0.0
        self.last_activity = time.monotonic()
        self.capabilities = None

    def call(self, method, request, context):
        with self.lock:
            self.last_activity = time.monotonic()
            h = request.header
            if (
                h.protocol_version != 1
                or not h.session_id
                or h.sequence <= self.sequence
                or not np.isfinite(h.simulation_time)
                or h.simulation_time < 0
            ):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Invalid protocol/header/sequence",
                )
            if method != "Initialize" and self.session != h.session_id:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "Initialize a matching session",
                )
            if method == "Initialize" and self.session:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "Session already initialized",
                )
            if method == "Reset" and (
                not h.episode_id or h.episode_id == self.episode
            ):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Reset requires a fresh episode",
                )
            if method == "Step" and (
                not self.episode
                or h.episode_id != self.episode
                or h.simulation_time < self.time
            ):
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "Reset/matching monotone episode required",
                )
            self.sequence = h.sequence
            try:
                if method == "Initialize":
                    response = self.algorithm.initialize(request)
                    self.capabilities = response.capabilities
                    self.session = h.session_id
                elif method == "Reset":
                    response = pb.Acknowledgement(
                        artifacts=self.algorithm.reset(request)
                    )
                    self.episode, self.time = h.episode_id, h.simulation_time
                elif method == "Step":
                    validate_step(request, self.capabilities)
                    if not context.is_active():
                        context.abort(
                            grpc.StatusCode.CANCELLED, "Request cancelled"
                        )
                    started = time.monotonic()
                    response = self.algorithm.step(request)
                    response.diagnostics["algorithm_seconds"] = (
                        time.monotonic() - started
                    )
                    if (
                        response.diagnostics["algorithm_seconds"]
                        > request.solve_budget_seconds
                    ):
                        response.decision.Clear()
                        response.decision.status = pb.BUDGET_EXHAUSTED
                        response.decision.explanation = (
                            "Algorithm exceeded its wall-clock solve budget"
                        )
                        response.ClearField("sampled_reference")
                        response.ClearField("planner_geometry")
                    # Validate algorithm output before it crosses the service boundary.
                    # This mirrors the client-side decoder and prevents a foreign
                    # runtime from emitting future, stale, or horizon-inconsistent plans.
                    decode_decision(
                        response, h.simulation_time, self.capabilities
                    )
                    self.time = h.simulation_time
                    if not context.is_active():
                        self.episode = ""
                        context.abort(
                            grpc.StatusCode.CANCELLED,
                            "Late decision; reset required",
                        )
                else:
                    self.algorithm.close()
                    response = pb.Acknowledgement()
                    self.closed.set()
                response.header.CopyFrom(h)
                return response
            except (ValueError, KeyError) as exc:
                self.episode = ""
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception as exc:
                self.episode = ""
                context.abort(grpc.StatusCode.INTERNAL, str(exc))
            finally:
                self.last_activity = time.monotonic()


def serve(algorithm, address, idle_timeout=120.0):
    service = Service(algorithm)
    server = grpc.server(
        ThreadPoolExecutor(max_workers=2),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    methods = {}
    for name, request in (
        ("Initialize", pb.InitializeRequest),
        ("Reset", pb.ResetRequest),
        ("Step", pb.StepRequest),
        ("Close", pb.CloseRequest),
    ):
        methods[name] = grpc.unary_unary_rpc_method_handler(
            lambda request, context, name=name: service.call(
                name, request, context
            ),
            request_deserializer=request.FromString,
            response_serializer=lambda response: response.SerializeToString(),
        )
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "drone.native.v1.Algorithm", methods
            ),
        )
    )
    if not server.add_insecure_port(address):
        raise RuntimeError("Cannot bind native service address")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: service.closed.set())
    server.start()
    try:
        while not service.closed.wait(0.2):
            # A dead owner must not leave ROS/C++ processes resident indefinitely.
            if (
                time.monotonic() - service.last_activity > idle_timeout
                and service.lock.acquire(False)
            ):
                service.lock.release()
                break
    finally:
        server.stop(2).wait()
        with service.lock:
            algorithm.close()
