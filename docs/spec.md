# Scope and release acceptance

The platform separates Learning, Simulation, and Benchmark under
[ADR-0026](adr/0026-training-benchmark-release-separation.md). The current release
uses Crazyflow forward physics, local BPTT/SHAC/PPO, state/depth/LiDAR Actors,
existing controllers, and native EGO/SUPER. Additional backends and scene
families require separate design and acceptance work.

## Task and method contracts

Tracking follows its configured reference. Racing visits gates in the declared
order. Static and Dynamic OA navigate to the goal with existing collision,
boundary, timeout, and arrival conditions. Navigation keeps its 0.5 m arrival
radius and 0.07 m collision sphere. Aggregate benchmark quality thresholds are
experiment-specific and are not hardcoded in the framework.

The environment owns task rewards, physical state, geometry, sensor acquisition,
and termination. Actor input contains only method-visible observations. Optional
privileged Critic input is training-only. Action contracts specify physical units,
frame, bounds, and control level. Delayed commands and measurements retain
per-world state and timestamps. Native methods use measured sensor data rather
than privileged map geometry.

OA training uses independent procedural layouts for each world and episode.
Static OA and Dynamic OA have separate policies and training distributions.
The unchanged S01/S02/S03/S06 and D01/D02/D03/D06 maps are fixed benchmark suites.

## Runtime release checks

Every finite combination displayed by the Command Builder must execute its
actual method. Training checks include updates, complete checkpoint persistence,
fresh Actor loading, and evaluation. Play and Benchmark checks include native
worker/solver execution where selected, complete episode counts, and reports.
A parsed command or composed Hydra configuration is only a preflight check.
Explicit checkpoint paths are additionally checked for loading and contract
validation; arbitrary user-entered paths are not a finite acceptance matrix.

Run appropriate unit/integration tests and installed-wheel checks when public
interfaces or packaging change. Record exact source/configuration identities and
actual limitations. Retain failed attempts. Reduced-duration checks establish
runtime compatibility only.

## Convergence evidence

The Command Builder exposes 42 training recipes: PPO plus BPTT and SHAC with
three supported backward models across six task/sensor groups. Independent
initialization seeds, sample budgets, and convergence criteria belong to a
separate experiment-specific protocol; the 230-command runtime compatibility
check is not a convergence study.

Declare environment count, rollout steps, iteration budget, evaluation cohort,
and quality criterion before execution. Report each scene and seed, not only an
aggregate reward or success count. Continuation checks evaluate the actual final
checkpoint and preservation of previously successful episodes. Incomplete or
regressed runs remain incomplete or failed under their declared protocol.

## Measurement and evidence

[Benchmark](benchmark.md) defines FlightBench motion metrics, static path excess,
Traversability, computation timing, denominators, and unavailable values.
[Validation](validation.md) records current execution evidence. Historical
archives retain their original criteria and cannot substitute for current-source
acceptance. [ADR-0017](adr/0017-eight-acceptance-gates.md) retains the historical
gate organization; ADR-0026 replaces its former generic quality thresholds and
training selection workflow.
