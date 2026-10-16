# ADR-0028: Paired benchmark initial conditions

**Status:** Implemented; focused CPU tests passed on 2026-10-16.
**Scope:** Flight-case construction, not training or checkpoint selection.

## Context

The earlier Benchmark used a vectorized initial-state batch for JAX methods
but reseeded a one-world environment for each ROS planner episode. Consequently,
two methods could be evaluated on different initial positions, velocities, and
sensor histories even with the same `benchmark.seed_base`.

## Decision

For each scene, construct the initial batch once using the declared suite seed.
Vectorized methods evaluate the full batch. Host planners evaluate each world
serially by slicing that same initial batch, without resetting or resampling it.
The single-world simulator metadata is restored from its environment template;
physical arrays come from the shared cohort. Each result records a stable
`case_id` of the form `S01:0000`, together with the cohort seed and world index.

The scene, initial position and velocity, physical randomization, observation
buffers, and obstacle motion are consequently fixed for a case independently of
which learning policy, controller, or planner runs it. Method-specific stochastic
behavior remains an execution choice rather than an initial-condition change.

Play and Benchmark use the same Simulation rollout interface. Play selects a
scenario for inspection and replay, without the expensive latency or complexity
sweeps. Benchmark runs the declared multi-episode suite and reports all outcomes,
including failures. Neither changes the training state or chooses a checkpoint.

## Verification

`tests/test_benchmark_pairing.py` checks leaf shapes and actual initial physical
states for vectorized and serial Tracking/Navigation environments, and checks a
complete host/JAX short flight beginning from identical initial positions and
velocities. `tests/test_play_vs_benchmark.py` checks that Play skips Benchmark-only
measurements. Command Builder coverage and native planner compatibility remain
separate integration checks; short flights do not establish convergence.

This decision preserves the existing FlightBench-style metrics and the
AvoidBench-inspired collision-free reference path and Traversability metrics.
