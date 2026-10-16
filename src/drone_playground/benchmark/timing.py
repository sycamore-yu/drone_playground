"""Warm single-world computation timings, separated from simulated flight time."""

import math
import time

import jax
import numpy as np

from drone_playground.simulation.ros_planner import RosPlannerError


def profile(env, method, actor=None, parameters=None, *, samples=20):
    """Time synchronized decision calls after warm-up; exclude physics and sensor synthesis."""
    if env.num_envs != 1:
        raise ValueError("Latency measurement requires one world")
    if type(samples) is not int or samples < 1:
        raise ValueError("Latency measurement requires a positive sample count")
    state = env.reset(jax.random.PRNGKey(0))
    memory = method.initialize(env, 0, actor)
    if method.execution == "jax":
        if method.trainable:

            def command(state, memory, parameters):
                raw, memory, _ = actor.apply(parameters, env.observe(state), memory)
                return env.action_setpoint(state, method.deterministic(raw)), memory
        else:

            def command(state, memory, parameters):
                return method.controller(state.physics, env.task.target(state.time), memory)

        command = jax.jit(command)
    else:
        command = method.host_command
    simulation = jax.jit(env.step_setpoint)
    warmup = 1
    if getattr(method, "planner", None) is not None:
        latency = env.sensor.config.latency
        warmup += math.ceil((latency + 1 / env.sensor.config.frequency_hz) / env.dt)
    observations = []
    error = None
    for index in range(samples + warmup):
        begin = time.perf_counter()
        decision = None
        if method.execution == "jax":
            control, memory = jax.block_until_ready(command(state, memory, parameters))
        else:
            try:
                control, memory, _, decision = command(env, state, memory)
            except RosPlannerError as failure:
                error = str(failure)
                break
            jax.block_until_ready(control)
        decision_ms = (time.perf_counter() - begin) * 1000
        transport_ms = 0.0
        if decision is not None:
            transport_ms = max(0.0, decision["rpc_roundtrip_ms"] - decision["total_ms"])
        begin = time.perf_counter()
        state = jax.block_until_ready(simulation(state, control))
        simulation_ms = (time.perf_counter() - begin) * 1000
        if index >= warmup:
            observations.append((decision_ms - transport_ms, simulation_ms, transport_ms))
    values = np.asarray(observations)
    return {
        name: {
            "mean_ms": float(values[:, i].mean()) if observations else None,
            "p95_ms": float(np.percentile(values[:, i], 95)) if observations else None,
            "count": len(observations),
        }
        for i, name in enumerate(
            ("computation", "simulation_and_sensor_synthesis", "transport_and_rpc_overhead")
        )
    } | {
        "complete": len(observations) == samples,
        "error": error,
        "protocol": {
            "worlds": 1,
            "warmup_calls": warmup,
            "synchronization": "block_until_ready",
            "computation_scope": "perception encoding/mapping, policy/planner, setpoint control",
            "flight_controller_ticks": "inside simulation",
            "devices": [str(device) for device in jax.devices()],
        },
    }
