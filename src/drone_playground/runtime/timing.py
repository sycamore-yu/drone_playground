"""Synchronized decision latency and the sensor schedule actually executed."""

import time

import jax
import numpy as np


def decision_statistics(seconds, dt, *, warmup=1):
    """Summarize measured decision latency relative to the control interval."""
    values = np.asarray(seconds, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0) or dt <= 0:
        raise ValueError(
            "Decision latency needs finite, nonnegative durations and a positive deadline"
        )
    stable = values[warmup:]
    return dict(
        decision_p50_ms=float(np.median(stable) * 1000) if len(stable) else None,
        decision_p95_ms=float(np.quantile(stable, 0.95) * 1000) if len(stable) else None,
        deadline_miss_fraction=float(np.mean(stable > dt)) if len(stable) else None,
        decision_samples=len(stable),
        warmup_decisions=min(warmup, len(values)),
        decision_deadline_ms=dt * 1000,
        timing_protocol=(
            "observation available to executable command; synchronized completion; physics excluded"
        ),
    )


def measure_decision(decide, dt, *, samples=20):
    """Measure host-side method decision latency over repeated samples."""
    durations = []
    for _ in range(samples + 1):
        start = time.perf_counter()
        jax.block_until_ready(decide())
        durations.append(time.perf_counter() - start)
    return decision_statistics(durations, dt)


def measure_policy(env, make_policy, params, observation, state=None):
    """Measure a learned policy's inference time using the active environment."""

    def decide(parameters, obs, current):
        action = make_policy(parameters, deterministic=True)(obs, jax.random.PRNGKey(0))[0]
        if hasattr(env, "policy_command"):
            return env.policy_command(current, action)
        return getattr(env, "physical_action", lambda action: action)(action)

    decision = jax.jit(decide)
    return measure_decision(lambda: decision(params, observation, state), env.dt)


def sensor_schedule(sensor, policy_hz):
    """Derive the periodic sensor schedule for the policy control rate."""
    if sensor is None:
        return None
    period = sensor.period_steps(policy_hz) if hasattr(sensor, "period_steps") else 1
    return dict(
        requested_source_hz=float(sensor.source_rate_hz),
        nominal_capture_hz=float(getattr(sensor, "capture_rate_hz", sensor.source_rate_hz)),
        effective_capture_hz=policy_hz / period,
        period_steps=period,
        capture_clock="simulation seconds",
        availability_latency_s=0.0,
    )
