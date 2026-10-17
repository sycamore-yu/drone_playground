# ADR-0034: Trainable release defaults and task-matched sampling

Status: Accepted.
Approval: User instruction on 2026-10-17 in the release-training session.
Recorded: 2026-10-17T02:21:01Z.
Implementation: Implemented in `feat/release-trainable-defaults`. Sixty-seven targeted tests pass, including real PPO/BPTT/SHAC updates with both sensors and matched-motion replay. Full-size Depth PPO updates pass on the target RTX 4090. Launch results and convergence scores are recorded in the run directory, not inferred from tests.

## Goal and scope

The release needs a usable default recipe for each of the 42 supported Train
combinations. The task is to demonstrate training convergence from random
initialization. Reuse verified parameter choices and already passing results to
avoid repeated tuning. Do not run a separate historical-baseline research study.
An older trained Actor does not establish that the current training recipe converges.

Retain completed results and replay records. Stop starting jobs with known
inadequate sampling or success checks. Preserve a currently useful training run
rather than restarting it just to change the orchestration source snapshot.

## Decisions

### Training distribution

Use the existing procedural scene implementation with a `matched` layout preset.
Training does not load the fixed S/D benchmark geometry. The preset:

- samples short, medium and long start-goal distances, retaining long flights
  from the start and increasing their proportion toward the end of training;
- uses a different bent interior route in each world instead of an always-clear
  exterior lane;
- covers forest obstacle counts and sizes, wide obstacles and variable interior
  openings, using the published scene geometry only to measure parameter ranges;
- samples both existing prescribed-motion equations with new scales, phases and
  periods, allowing moving obstacles to cross the static-clear route.

The route construction proves a static collision-free connection for sampled
starts. Dynamic obstacles require timing and avoidance. A construction route is
not supplied to the policy as a privileged observation.

The ordinary training distance mixture changes from 40/30/30 percent to
10/20/70 percent for short/medium/long flights. Nominal ranges are 10-30 m,
30-70 m and 80-98 m; the finite map limits the longest straight-line distance to
96 m. Half of the long checkpoint-check cases use that full 96 m flight.

### Convergence checks within Learning

Keep `learning`, `simulation`, and `benchmark` as the public ownership boundary.
Do not restore a top-level Validation module, command or dataset type.

An optional within-Learning check uses 24 stratified cases. All quick cases must
succeed before running a 96-case confirmation with a separate fixed seed.
Only the confirmation may satisfy a configured convergence target. The 24 and
96 cases both balance distance and geometry families. Ordinary training can
still disable checks and run to a fixed interaction budget.

Success-based checkpoint selection is an optional metric of the existing Best
selector. Confirmed results take priority over small-check scores. Latest is
saved as before. The final fixed-map Benchmark runs separately and records all
100 requested cases per static/dynamic group. It is not invoked by Learning.
A passed quick check, a completed process, and a passed final Benchmark are
recorded as different facts.

### Release code boundary

Reusable scene sampling, optional check/selection logic, recipe YAMLs and tests
belong in the release. Use existing components; do not add a second trainer or a
validation framework.

Machine-specific scheduling, CPU affinity, experiment queues, old-result import,
trial status and automatic candidate promotion belong to the run directory
under `results/`. These are operational tools, not package dependencies. Retain
original provenance without copying historical weights into new training jobs.

### Resource use

Run one memory-fitting GPU training job and several CPU jobs concurrently.
The user permits CPU utilization of 50 percent and above. Use bounded worker
slots with explicit CPU affinity and memory admission. Measure actual load;
allocated cores do not prove measured utilization. Preserve unrelated jobs.
State tasks, CPU-suitable OA jobs, frozen evaluation and replay work can occupy
CPU slots while the GPU trains perception policies.

## Parameter evidence

The local research note `docs/research/ppo-training-findings-20261016.md` is the
recipe reference. Its useful findings include the Racing PPO reference optimizer
settings, the finite-horizon goal-reaching PPO reward, gamma and lambda equal to
one, a 512-step rollout, and the observed benefits of removing Depth PPO entropy
bonus. Historical BPTT/SHAC coefficients are retained where applicable.

Those historical scores used their recorded map and device settings. The new
sampling preset must earn its own convergence result; selecting known useful
coefficients does not by itself prove the new run has passed.

## Training iteration budget (2026-10-17)

Use the same budget terminology as Isaac Lab / RSL-RL. The public Learning
configuration exposes `learning.max_iterations` as the number of optimizer
iterations, separate from the per-environment rollout horizon and number of
parallel worlds. Sample-interaction totals remain recorded as measurements and
can be derived for external orchestration; they do not set the native budget.
No separate `max_interactions` setting is retained. Frozen pre-migration runs
keep their original source and result metadata, and cannot be resumed under a
changed training-configuration contract without an explicit migration.
The release OA PPO iteration cap is raised from the former 64 (Depth) or 256
(LiDAR) updates to 1,000, consistent with the other OA algorithms. This is a
search budget for convergence, not evidence that the PPO policy converges. The
optional success-rate confirmation can stop a run earlier.

## Earlier decisions affected

This updates the convergence scope in ADR-0026 from a representative subset to
all 42 supported Train combinations. It keeps the removal of independent
Validation and extends ADR-0027's optional Best selection with a success metric
and confirmation check. It replaces the temporary proposal to add a historical
baseline-reproduction study.

Runtime tests, default-recipe quality, and final flight quality remain separate
records. A failed final result is retained and cannot promote a recipe.
