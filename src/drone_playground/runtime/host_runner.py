"""Synchronous host orchestration for external decisions and compiled physics."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class HostStep:
    before: object
    after: object
    command: object
    diagnostics: dict
    environment_seconds: float


def run_steps(
    initial,
    length: int,
    decide: Callable,
    advance: Callable,
    after_step: Callable | None = None,
):
    """Yield every physical transition, including the terminal transition.

    Resource creation/cleanup stays with the owning method context. A method's
    completion callback may stop iteration; task outcomes remain task-owned.
    """
    if length < 1:
        raise ValueError("A host rollout requires at least one control interval")
    state = initial
    for tick in range(length):
        started = time.perf_counter()
        command, diagnostics = decide(state, tick)
        jax.block_until_ready(command)
        decision_seconds = time.perf_counter() - started
        decision_seconds -= diagnostics.get("observation_seconds", 0.0)
        decision_seconds -= diagnostics.get("recording_seconds", 0.0)
        diagnostics = {**diagnostics, "decision_seconds": decision_seconds}
        start = time.perf_counter()
        candidate = advance(state, jnp.asarray(command, jnp.float32))
        jax.block_until_ready(candidate.obs)
        record = HostStep(state, candidate, command, diagnostics, time.perf_counter() - start)
        finished = bool(after_step(record)) if after_step is not None else False
        yield tick, record, finished
        state = candidate
        if bool(state.done) or finished:
            break
