# ADR-0026: Separate training, simulation, and benchmarking

**Status:** Accepted for the decisions below; remaining choices are listed explicitly.
**Approval recorded:** 2026-10-16T04:30:29Z, from the user's instructions in this session.
**Implementation:** Implemented in the working tree. All 230 Command Builder runtime checks pass. Formal convergence is tracked in local experiment protocols.
**Last reviewed:** 2026-10-16T06:24:13Z against `main@7d59162` and the working tree.

Optional reward Best selection and its publication rules are now governed by
[ADR-0027](0027-reward-selection-and-checkpoint-publication.md), approved after
the original lifecycle decision below.

## Decision

Organize the package into Learning, Simulation, and Benchmark. Learning owns
parameter updates and resumable training state. Simulation owns environments,
methods, physical execution, and replay. Benchmark owns evaluation cases,
measurement protocols, metrics, and reports. Both Learning and Benchmark use
Simulation. Training does not invoke Benchmark.

Use Autoresearch Gym as the public training orchestration entry. Reuse the
existing Trainer and external adapter. Candidate comparison stays outside the
training loop; final release measurements use a separately declared test cohort.

Use **BPTT** as the name of the local differentiable policy-training algorithm
formerly called `apg`. Update public commands, configuration, source names,
callers, and tests together. This naming decision leaves the update equations
unchanged and preserves the original names of cited upstream methods.

Follow the Isaac Lab + RSL-RL training lifecycle: collect samples, update
parameters, log, and save numbered checkpoints at intervals and on completion.
Remove the independent `checkpoint_eval` command and the training-time
screening, full-validation, patience, progress-guard, and first-pass selection
workflow. Remove the former optional Validation workflow as well. ADR-0027
subsequently adds optional reward evaluation within Learning.

Align checkpoint selectors with the Isaac Lab **develop** workflow requested by
the user: `latest`, `best`, or an explicit checkpoint. The develop documentation
defines `best` as the library-recorded best or final checkpoint, falling back to
latest when unavailable. RSL-RL's own runner does not add independent validation.
Autoresearch candidate ranking remains separate from within-run checkpoint
references. The earlier inference that the user approved removing `best` is
withdrawn: it was based on the older Isaac Lab `main` entrypoint, not the requested
develop workflow. The develop RSL-RL entrypoint supplies no preferred checkpoint
pattern. This originally made both selectors resolve to the latest compatible
numbered checkpoint. ADR-0027 subsequently adds recorded reward Best selection;
fallback remains applicable when no Best is recorded. Report the resolved
checkpoint and whether a `best` request used this fallback.

Keep Orbax as the sole current checkpoint format. Remove ZIP save, load, and
resume paths. Retain historical files and their original provenance locally;
historical execution remains associated with its source revision. This change
does not authorize deletion or conversion of stored results.

Navigation training uses independently randomized obstacle layouts. The fixed
navigation maps belong exclusively to Benchmark, organized as **Static Obstacle
Avoidance (Static OA)** and **Dynamic Obstacle Avoidance (Dynamic OA)**. Remove
fixed-map training recipes and their Command Builder choices. Preserve the
mapping from historical scene IDs to benchmark cases and preserve geometry.

Release acceptance has two parts: execute every displayed Command Builder
combination end to end, and demonstrate convergence for an explicitly selected
representative set. Configuration composition alone is not execution evidence.
Keep runtime checks and convergence results separately identifiable.

Convergence coverage and independent seeds belong to each local experimental
protocol. Phase-specific trial matrices and scripts are not public project code.

Remove the framework's hardcoded aggregate pass thresholds (Tracking 95% with
RMSE at most 0.25 m, Racing 90%, and Navigation 23/25). Preserve the per-episode
success and failure conditions and report all measured outcomes. Each convergence
experiment declares its quality criteria before execution. This decision was
approved in the reply to Q9, recorded at 2026-10-16T04:47:15Z. It changes the
quality-gate requirement in [ADR-0017](0017-eight-acceptance-gates.md).

Expand the builder to Train, Play, and Benchmark using existing algorithms,
backward models, sensors, controllers, and native planners. Additional learning
libraries, physics backends, and new scene families are deferred to separate
acceptance work. This scope was confirmed in the user's reply to Q5.

Publish English documentation and code comments. Remove host-specific operating
notes and transient experiment material from the public source surface while
preserving local evidence. Keep upstream method names and attribution where they
identify an actual published method.

## Upstream evidence

- [RSL-RL runner, revision a87ef38](https://github.com/leggedrobotics/rsl_rl/blob/a87ef38fde4e2a554bec1565e14b868a64de5851/rsl_rl/runners/on_policy_runner.py#L56-L144): fixed iteration loop, periodic and final saves; no validation or best-selection phase.
- [Isaac Lab develop training, revision d7339b7](https://github.com/isaac-sim/IsaacLab/blob/d7339b767df1b07249eb4a51a0383df1bb6ae55c/source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py): constructs the runner and calls `learn`, without independent validation.
- [Isaac Lab develop checkpoint selectors](https://github.com/isaac-sim/IsaacLab/blob/d7339b767df1b07249eb4a51a0383df1bb6ae55c/source/isaaclab_rl/isaaclab_rl/entrypoints/common.py#L653-L714): `best` prefers a backend-recorded best/final checkpoint and otherwise resolves like `latest`.
- [Isaac Lab Command Builder](https://isaac-sim.github.io/IsaacLab/develop/source/setup/environments.html): exposes supported task, learning-library, physics, rendering, and preset combinations.
- [Isaac Lab develop evaluation guide](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html#evaluate-a-policy): defines `latest`, `best`, pretrained, and explicit-path selection.
- [Isaac Lab benchmark success tracking](https://github.com/isaac-sim/IsaacLab/blob/d7339b767df1b07249eb4a51a0383df1bb6ae55c/source/isaaclab/isaaclab/benchmark/entrypoints/early_stop.py): configurable convergence measurement, with early stopping enabled only by an explicit flag. Its defaults do not define this project's task quality.
- [DiffAero training runner](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/utils/runner.py#L89-L180): records training-episode success rates and saves `best` when a new maximum coincides with a save interval; no fixed aggregate release pass threshold.

## Earlier decisions affected

This decision replaces the training selection and validation requirements in
[ADR-0015](0015-three-stage-learning-acceptance.md),
[ADR-0018](0018-training-patience-and-frozen-selection.md), and
[ADR-0024](0024-navigation-ppo-progress-regression-screening.md).
Their historical results keep their original interpretation.

It changes Benchmark ownership in [ADR-0001](0001-simulation-learning-boundary.md),
removes current ZIP compatibility from [ADR-0016](0016-results-layout-orbax-tensorboard.md),
and makes [ADR-0025](0025-autoresearch-gym.md) the public training workflow.
It retains the separation of randomized training geometry and fixed benchmark
geometry in [ADR-0019](0019-procedural-training-fixed-navigation-benchmark.md),
while removing its training-time checkpoint-validation requirement.

## Choices still under discussion

- Training budgets and quality criteria belong to local convergence protocols.
  Approved coverage includes the supported differentiable backward models and
  three independent seeds. Temporary schedules remain outside Git.

These choices are not approved by this ADR.

## Benchmark measurement decision

The user approved this measurement scope in the reply to Q6; the approval was
recorded at 2026-10-16T04:38:30Z. Success rate counts
every requested episode. Aggregate speed, acceleration, jerk, and curvature over
successful episodes, report the contributing count, and retain failed-episode
records and trajectories separately. Preserve existing task success criteria.

Use the definitions in [FlightBench v3](https://arxiv.org/html/2406.05687v3#S3.SS3):
average speed is path length divided by flight time; squared acceleration and
jerk integrals are normalized by path length; average curvature is weighted by
distance along the trajectory. Derivatives describe realized world-frame motion.
Record the sampling interval and differentiation method. Zero-length paths and
insufficient samples produce an explicitly unavailable metric.

Measure computation time after warm-up, with device work synchronized. Include
perception processing, planning or policy execution, and control. Report
simulation stepping and communication separately. Preserve timing sample counts
and the execution conditions used for comparison.

Add static-scene path excess relative to a collision-free reference path, following
[AvoidBench](https://arxiv.org/pdf/2301.07430). Record reference-path resolution
and clearance; straight-line distance through obstacles is not that reference.
Use Traversability for the initial environment-complexity measurement, with fixed
spatial and directional samples. Dynamic measurements also declare the sampled
times. Relative Gap Size and new scene families remain outside this release scope.

## Implementation checks

Verify bounded training and interruption/resume with reward selection disabled;
round-trip Orbax state and execute its Actor through Play; verify stable
checkpoint selection by step; test benchmark formulas on analytic trajectories;
run the generated command matrix, and verify the installed package outside the
checkout. Runtime evidence is recorded in [Validation](../validation.md): 42
Train, 142 Play, and 46 Benchmark commands passed; exact resumption, analytic
metrics, real native execution, and installed-wheel checks also passed.
Convergence acceptance remains separate and incomplete.

## Independent OA training distributions

Approval recorded: `2026-10-16T05:09:28Z`. Static OA samples random static
obstacles; Dynamic OA samples random moving obstacles. Fixed S/D maps remain
exclusive to Benchmark. Separate procedural distributions are implemented and
covered by runtime checks. Convergence is a separate experimental claim.
Last reviewed at `2026-10-16T06:24:13Z` against the working tree based on `7d59162`.
