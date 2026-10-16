# ADR-0009: Delay buffers belong to environment state

**Status:** Accepted.
**Approval:** Originally recorded as 2026-10-09; exact time was not retained.
**Implementation:** `CommandState` and `ObservationState` hold per-world delivery history.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Context

Commands and measurements can arrive after production. Their histories must
advance independently per world, support partial resets, and survive training
continuation. Saving only the latest action or measurement is insufficient.

## Decision

Keep command buffers in environment execution state and measurement buffers in
`ObservationState`. Shared controllers, sensor configuration objects, and host
globals do not own mutable delivery queues. Existing modules perform acquisition,
enqueueing, and delivery.

A world uses only delivered data. Before a new command arrives it keeps the last
valid command. Measurements retain acquisition time, validity masks, and acquisition
pose. Partial resets clear only selected worlds. Checkpoints preserve buffers and
related random state. Action values retain pathwise derivatives through buffering;
delivery indices follow the declared schedule.

Use physical ticks or timestamps to match the actual clock relationship. Do not
add a generic messaging framework.

## Consequences

Record sample rate, delivery delay, and motor response separately. Delay uses
simulation time. `simulation.action_delay_s` and `sensor.config.latency` accept
fixed seconds or a per-episode uniform `[low, high]` interval. Capacity follows the
maximum delay. Before the first measurement arrives, the method sees invalid data
and the initial command remains active. Planners wait while physics and task time
continue. Per-message jitter, reordering, and retransmission are outside this scope.

Targeted checks cover zero-delay behavior, fixed delivery, partial resets, exact
continuation, world independence, and delayed action gradients. The implementation
is in `environment.py`, `observation.py`, and `sensors.py`.
