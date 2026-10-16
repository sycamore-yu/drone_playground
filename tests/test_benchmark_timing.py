"""Decision timing excludes startup acquisition and separates native RPC overhead."""

from types import SimpleNamespace

import pytest

from drone_playground.benchmark import timing
from drone_playground.simulation.ros_planner import RosPlannerError


def test_native_probe_warms_sensor_and_planner_before_measurement(monkeypatch):
    """The first sensor/planner calls cannot contaminate the measured samples."""
    monkeypatch.setattr(timing.jax, "jit", lambda function: function)
    monkeypatch.setattr(timing.jax, "block_until_ready", lambda value: value)
    clock = iter(value for i in range(5) for value in (i, i + 0.010, i + 0.020, i + 0.024))
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    env = SimpleNamespace(
        num_envs=1,
        dt=0.02,
        sensor=SimpleNamespace(config=SimpleNamespace(latency=0, frequency_hz=50)),
        reset=lambda key: 0,
        step_setpoint=lambda state, control: state + 1,
    )
    calls = []

    def command(env, state, memory):
        calls.append(state)
        decision = None if state == 0 else {"rpc_roundtrip_ms": 5, "total_ms": 2}
        return 0, memory, None, decision

    method = SimpleNamespace(
        execution="host", planner=object(), initialize=lambda *args: None, host_command=command
    )
    result = timing.profile(env, method, samples=3)
    assert calls == [0, 1, 2, 3, 4]
    assert result["protocol"]["warmup_calls"] == 2
    assert result["computation"]["mean_ms"] == pytest.approx(7)
    assert result["simulation_and_sensor_synthesis"]["mean_ms"] == pytest.approx(4)
    assert result["transport_and_rpc_overhead"]["mean_ms"] == pytest.approx(3)
    assert result["computation"]["count"] == 3 and result["complete"]


def test_native_probe_failure_is_unavailable_timing(monkeypatch):
    """A failed probe preserves its reason and does not replace flight outcomes."""
    monkeypatch.setattr(timing.jax, "jit", lambda function: function)
    env = SimpleNamespace(num_envs=1, reset=lambda key: 0, step_setpoint=lambda *args: 0)

    def fail(*args):
        raise RosPlannerError("NO_SOLUTION", "probe failed")

    method = SimpleNamespace(execution="host", initialize=lambda *args: None, host_command=fail)
    result = timing.profile(env, method)
    assert not result["complete"] and "probe failed" in result["error"]
    assert result["computation"] == {"mean_ms": None, "p95_ms": None, "count": 0}
