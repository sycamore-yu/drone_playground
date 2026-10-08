"""Shared control-interval execution on the native physical clock."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _select(mask, new, old):
    return jax.tree.map(
        lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (n.ndim - mask.ndim)), n, o),
        new,
        old,
    )


@dataclass(frozen=True)
class ActionTransition:
    """Own physical ticks, held commands, event accumulation and optional freezing.

    Callbacks retain the backend's native state and task event rules. A backend
    can advance an unchecked interval in bulk without changing its integrator or
    derivative. All observed intervals use the same scan below.
    """

    apply_command: Callable
    advance: Callable
    substeps: int
    physics_dt: float

    def __post_init__(self):
        """Validate and prepare the ActionTransition instance after initialization."""
        if self.substeps < 1 or self.physics_dt <= 0:
            raise ValueError("Execution requires positive physical steps and timestep")

    @property
    def dt(self):
        """Return the duration of one control interval in seconds."""
        return self.substeps * self.physics_dt

    def _scan(
        self,
        state,
        control_at,
        event=None,
        *,
        timestamp=0.0,
        outcome=0,
        memory=(),
        freeze=False,
        sample_from_start=False,
        record=None,
    ):
        origin = state
        timestamp = jnp.asarray(timestamp, jnp.float32)
        outcome = jnp.asarray(outcome, jnp.int32)
        minimum = jnp.full_like(timestamp, jnp.inf)

        def tick(carry, index):
            current, clock, result, event_memory, minimum = carry
            elapsed = (index + 1) * self.physics_dt
            source = origin if sample_from_start else current
            dt = elapsed if sample_from_start else self.physics_dt
            candidate = self.advance(source, control_at(source, index), dt)
            now = timestamp + elapsed
            if event is None:
                current, clock = candidate, now
            else:
                finite, occurred, clearance, proposed_memory = event(
                    current, candidate, now, index, event_memory
                )
                active = (result == 0) if freeze else jnp.ones_like(result, bool)
                current = _select(active & finite, candidate, current)
                event_memory = _select(active, proposed_memory, event_memory)
                clock = jnp.where(active, now, clock)
                result = jnp.where(result == 0, occurred, result)
                minimum = jnp.where(active & finite, jnp.minimum(minimum, clearance), minimum)
            observation = None if record is None else record(current)
            return (current, clock, result, event_memory, minimum), observation

        result, history = jax.lax.scan(
            tick, (state, timestamp, outcome, memory, minimum), jnp.arange(self.substeps)
        )
        current, clock, outcome, memory, minimum = result
        return (
            current,
            clock,
            outcome,
            memory,
            jnp.where(jnp.isfinite(minimum), minimum, 0.0),
        ), history

    @staticmethod
    def _probe(probe):
        if probe is None:
            return None

        def observe(previous, current, time, index, memory):
            del previous, time
            clearance, collision = probe(current, index)
            return jnp.array(True), collision.astype(jnp.int32), clearance, memory

        return observe

    def step(self, state, physical, probe=None):
        """Hold one decoded command; optionally retain every substep collision."""
        controlled = self.apply_command(state, physical)
        if probe is None:
            return self.advance(state, controlled, self.dt)
        result, _ = self._scan(state, lambda current, index: controlled, self._probe(probe))
        return result[0], result[4], result[2] != 0

    def step_schedule(self, state, commands, probe=None, *, record=None):
        """Apply each due command on its actuator tick, retaining optional trace data."""
        if commands.shape[0] != self.substeps:
            raise ValueError("Command schedule must cover exactly one control interval")
        result, history = self._scan(
            state,
            lambda current, index: self.apply_command(current, commands[index]),
            self._probe(probe),
            record=record,
        )
        if record is not None:
            return result[0], history
        return (result[0], result[4], result[2] != 0) if probe else result[0]

    def checked(
        self,
        state,
        commands,
        event,
        *,
        timestamp,
        outcome,
        memory=(),
        freeze=True,
        sample_from_start=False,
    ):
        """Execute a scheduled interval using the task's first-event predicate.

        The event returns finite-state, outcome, clearance and task memory.
        ``sample_from_start`` retains an existing discrete-time model's dense
        sampling rule; it does not turn that model into a different integrator.
        """
        if commands.shape[0] != self.substeps:
            raise ValueError("Command schedule must cover exactly one control interval")
        result, _ = self._scan(
            state,
            lambda current, index: self.apply_command(current, commands[index]),
            lambda old, new, time, index, mem: event(old, new, time, mem),
            timestamp=timestamp,
            outcome=outcome,
            memory=memory,
            freeze=freeze,
            sample_from_start=sample_from_start,
        )
        return result


def delayed_schedule(command, previous, delay_ticks, substeps):
    """Select the old command until its replacement reaches the physical clock."""
    delay_ticks = jnp.asarray(delay_ticks)
    ticks = jnp.arange(substeps).reshape((substeps,) + (1,) * delay_ticks.ndim)
    return jnp.where((ticks < delay_ticks)[..., None], previous, command)
