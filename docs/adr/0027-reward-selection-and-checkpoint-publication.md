# ADR-0027: Reward selection and checkpoint publication

**Status:** Accepted.
**Reward-selection approval:** 2026-10-16T06:45:58Z.
**Publication and measurement approval:** 2026-10-16T07:07:08Z.
**Implementation:** Implemented in the working tree; focused tests and GPU mechanism checks passed. Current packaging and full release checks remain separate.
**Last reviewed:** 2026-10-16T07:44:24+00:00 against `main@7d59162` and the working tree.

## Context

[ADR-0026](0026-training-benchmark-release-separation.md) removed screening and
first-pass selection. The user subsequently requested optional reward Best,
consistent checkpoint publication, and shared live/final Autoresearch measurements.

## Decision

`learning.best.source` selects independent fixed-distribution `evaluation`,
completed `training` episodes, or `null` to disable selection. Prefer fixed
reward evaluation when training difficulty changes. Learning uses Simulation;
final Benchmark remains separate and is not called by Learning.

Intervals count environment interactions. Measure after a completed update
crosses an interval and at the final budget update. Fixed evaluation uses
deterministic actions, separate environment and random state, fixed procedural
seed and difficulty, and complete Task-objective episode returns. It does not
advance training state. Training-source selection carries partial episodes across
updates and compares only completed returns. Strict improvement replaces Best;
ties preserve the earlier checkpoint. Training ends at its budget or an explicit
interruption. This decision adds no patience chain or automatic early stopping.

Commit both Orbax payloads and metadata before publication. One atomic Latest
pointer publishes a snapshot containing its Best reference; `best_model` follows
that reference. Readers use committed metadata rather than mutable run summaries.
Retention follows publication and preserves frozen Actors, including a Best whose
training payload was pruned. When no Best is recorded, Best falls back to Latest
and reports that choice. Failed publication leaves the previous pair readable.

Autoresearch uses one projection of native cumulative episode measurements for
live and final records. Weight collection windows by their episode counts;
unmeasured returns remain null. Keep fixed-evaluation probes and their episode
records identifiable. Preserve historical records with their original meaning.

Study matrices, temporary queues and phase-specific scripts stay in ignored
local storage. Public source contains reusable training and evaluation behavior.

## Implementation and evidence

`learning/selection.py` measures reward; `learning/runner.py` connects updates and
saving. `learning/checkpoint.py` publishes snapshots; `simulation/policy.py`
resolves committed metadata. `tools/autoresearch.py` projects native measurements.

The focused batch covering reward selection, Orbax publication, Autoresearch,
Command Builder composition, and real-physics continuation passed 24 tests.
GPU PPO and BPTT checks retained update one's Best after update two scored lower.
Training-source SHAC and depth/LiDAR OA mechanism checks also completed. These
short checks do not establish convergence or current package acceptance.

Episode accumulators change the current full TrainingState template. Do not
promise resumption of older templates; retain their Actors and evidence without
introducing a legacy loader or migration path.

This supersedes ADR-0026's unconditional absence of within-run reward evaluation
and Best-to-Latest equivalence. Its ownership decisions remain in force.
