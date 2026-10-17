# Validation evidence

The original 230-command compatibility sweep was recorded against the working
tree based on `main@7d59162` on 2026-10-16. The present release candidate also
changes reward-based Best selection, Dashboard projection, and paired benchmark
initialization. The targeted checks for those changes are listed below.
Formal training convergence is outside this release verification run.

## Runtime compatibility

| Mode | Commands | Observed result |
|---|---:|---|
| Train | 42 | Two real GPU updates, Orbax save, fresh-process Actor loading, and complete evaluation |
| Play | 142 | Complete episode reports, including latest/best selectors and real native methods |
| Benchmark | 46 | Complete requested scene sets with every episode recorded |

All 230 commands also composed through Hydra. A Chrome session exercised all
230 UI selections and matched the generated commands. The original runtime records have
no process, method, or numerical failures, and each timing probe recorded twenty
samples. Native checks used the real EGO/SUPER workers and acados solver.

Training checks use two worlds, a two-step horizon, and two updates: eight
interactions. Execution checks use one episode per scene. Tracking uses its
normal twenty-second trajectory; Racing and OA checks use a 0.2-second episode
limit. These checks establish execution compatibility. Timeouts are recorded as
flight outcomes; they do not establish navigation success or convergence.
Parallel smoke timings are not representative performance measurements.

The checks found and resolved three concrete issues:

- OA with LOTF retained acceleration-specific loss options despite using a
  body-rate action. The builder now selects the Task objective and clears those
  incompatible options. All eight affected training commands passed again.
- Float32 sensor timestamps could round a few nanoseconds beyond the host clock,
  causing SUPER to reject an available observation. The client normalizes only
  representational roundoff. Tests retain rejection of genuinely future data;
  all seven affected native commands passed their real-worker reruns.
- The first smoke protocol shortened Tracking's period from twenty seconds to
  0.2 seconds and therefore changed reference velocity. This caused acados solve
  failures. All sixteen Tracking execution commands were rerun at the original
  period. The controller and solver were unchanged. Attitude MPC completed a
  twenty-second episode with position RMSE 0.05657 m.

Every original attempt remains local. Commands, source/configuration hashes,
logs, outcomes, and the final manifest are under
`results/release-check-20261016/`; `runtime-summary.json` maps all 230 commands to
their passing attempt. Checks performed during the refactor retain their actual
source hash. Changed behavior received targeted reruns.

## State, metrics, and packaging

Public Autoresearch resumption from update one to update two produced exactly
the same 148 TrainingState leaves as uninterrupted training, including optimizer,
RNG, environment, and policy state. Signal handling and preservation of the last
finite update also have focused regression tests.

Analytic trajectory tests verify speed, acceleration, jerk, and curvature.
Geometry tests verify obstacle detours, blocked endpoints, and repeatable
Traversability. The actual S01 reference-path check measured 98.93709 m between
the nominal endpoints with a 0.5 m grid and 0.07 m body radius. A frozen depth
policy exported a complete RScope replay with sensor data and scene assets.

The full CPU regression initially reported 367 passed, eleven failed, and one
skipped. All eleven failures were repaired and their focused reruns passed.
Additional timing and native-clock tests passed. Final collection has 385 tests;
the entire CPU suite was not repeated after those focused fixes. Optional Brax
is absent from the primary environment and its adapter tests remain skipped.

The earlier Ruff and formatting checks passed. The original wheel and source
distribution built, and five installed-wheel checks passed outside the checkout,
covering imports, ten scenes, three Actor kinds, and twelve Hydra task/algorithm
configurations. Those installed-wheel checks predate reward-based Best selection.

## Current release candidate checks (2026-10-16)

The current source separates Learning, Simulation, and Benchmark. The public
`train` command uses Autoresearch Gym and supports optional reward Best without
the former `checkpoint_eval` entry or multi-stage screening. Checkpoint state
uses Orbax; live and completed training measurements share the same event
projection. The dashboard distinguishes completed-episode collection returns,
fixed-policy probes, environment steps, and outer-loop candidate scores.

- **Focused tests:** 33 passed for the builder, checkpoint, Autoresearch, reward
  selection, geometry, timing, and paired benchmark behavior. All 230 builder
  commands compose under Hydra. Three legacy CLI fixture failures were
  resolved by updating the mocked Trainer to return its documented completed-
  episode statistics; their focused rerun passed three tests.
- **Paired initial states:** three focused benchmark tests passed. They verify
  static JAX/host flight state and Tracking/Navigation initial-state matching.
  One additional Play smoke test verifies that flight inspection omits
  benchmark-only complexity and latency sweeps.
- **Real training:** an eight-interaction PPO Autoresearch run completed two
  updates, saved and selected its Orbax Actor, and finished the external
  benchmark. The short task duration is a mechanism check, not an indication
  of task quality.
- **Dashboard:** JavaScript syntax and the live HTTP page passed local checks.
  The final upstream status writer was separately checked for retained native
  `num_envs`, horizon, algorithm, policy-update counters, and null returns.
- **Packaging:** the current source built a wheel and source distribution using
  isolated PEP 517 dependencies. All five current-source installed-wheel checks
  passed outside the checkout.
- **Public content:** 236 text files passed the English/private-path/local-link
  audit. All 441 historical private artifacts remained on disk with matching
  hashes. An additional prose audit examined all 44 public Markdown/HTML files
  for recurrent defensive patterns and generic filler; it found no generic
  filler and retained scope qualifications relevant to historical evidence.

The full CPU suite was started but stopped after recording three failures in
outdated CLI mock data; the corresponding cases subsequently passed. It was not
completed under the current concurrent compute load. The original 230-command
runtime sweep has not been repeated in full after the newest changes, and the
full CPU regression has not been repeated to completion after the focused fixes.

## Dashboard training media (2026-10-16)

The Autoresearch training adapter now exports optional, one-world pose previews
for the MuJoCo `Live Frame` and `Sampled Rollout` modes. A separate renderer
process writes JPEG frames and a trajectory manifest into the session's `live/`
directory. Rendering is disabled until the Dashboard user selects a visual mode.
The world-camera preview shows flight motion and scene geometry, not depth or
LiDAR sensor measurements.

The renderer produced procedural obstacle geometry and loaded fixed Racing and
S01 MJCF scenes. A short Autoresearch PPO run with two worlds and 16 total
environment interactions published a two-frame sampled training rollout. Its
frame, manifest, and visual index were readable through the active Dashboard
HTTP server. Session `tracking_ppo_native_s219831_6st85j6x` retains that example.
All three learning algorithms passed focused preview and checkpoint-continuation
checks. This short run is an integration check, not convergence evidence.

## Extended scene and preset integration (2026-10-16)

The optional MJCF suites add twelve independently generated Forest, Maze,
Dense, and Dynamic Dense maps, with Easy/Medium/Hard variants. All twelve
compile with MuJoCo, have collision-free task endpoints, and regenerate
identically from their tracked generator. Dynamic geometry changes with time,
and Maze walls obstruct the direct nominal route. The `nominal`, `dynamics`,
and `sim2real` Hydra groups construct usable Crazyflow environments with the
documented model perturbations, delays, and gust parameters.

The current capability-based Builder contains **550 unique commands**
(42 Train, 380 Play, 128 Benchmark). All 550 configurations compose through
Hydra. The original 230-command core suite retains its historical runtime
evidence. The native Controller and Planner subset has a separate end-to-end
smoke acceptance recorded below; policy execution and training use their own
protocols.

The final training-media update retains a browser-playable animation of the
actual last one-world rollout fragment even when the Dashboard was Off. It
does not recover older unrecorded rollouts or substitute a new evaluation
trajectory. The updated scene, preset, preview, TrainingState continuation, and
Command Builder checks passed 23 focused tests; this is not a convergence run.

## Mainline Racing control and browser integration (2026-10-16)

Two controllers were selectively integrated from the Racing feature branch:
native acados contouring MPC and experimental JAX known-landmark perception-
aware sampling MPC. The original RPG Time-Optimal + Attitude MPC adapter,
original RPG ROS1/ACADO PAMPC, upstream checkout tools and numerical RPG solver
are excluded from `main`. The unmerged research branch retains that work.

The Racing feature branch's fixed-Level0 eight-episode Benchmark reported
**8/8 full gate-course successes**, with **14.682 s** mean completion time
and **0.045 s** sample standard deviation. CPU timing was **7.31 ms** mean and
**8.51 ms** P95 over 20 decisions. The experimental JAX perception MPC ran
short Play/Benchmark episodes, but its 32-sample, four-stage CPU setup measured
**83.46 ms** mean per decision, exceeding the 50 Hz (20 ms) control period.
Original simulation results remain in the ignored `results/` directory of
the Racing research worktree, not in the public Git tree. No training
convergence or cross-track performance is inferred from these tests.

After selection into a clean `main` worktree, the native MPCC, perception
objective and Builder suite passed **8 focused tests**, including a real acados
short Crazyflow flight. The 550-configuration Builder Hydra test passed.
Replay, Autoresearch projection, training media, paired benchmark and Play
tests passed **18 focused tests**. The MuJoCo WASM viewer built successfully
with `npm ci` and Vite. Its mainline build was tested independently in headless
Chrome against the recorded MPCC flight: 250 frames, eleven native Crazyflie
meshes, and eight Mocap bodies loaded. Browser Play advanced from frame 100 to
118 without JavaScript errors. A short JAX perception-aware Crazyflow Play also
completed at its 0.16-second test deadline without a method exception. A running
Dashboard from an older worktree is not automatically replaced by a Git commit.

The 550-command catalog is not intended to run every policy and training
configuration as one runtime sweep. Native method compatibility is checked
separately from Learning experiments; the current 172 Controller/Planner
Play/Benchmark commands are covered by the native smoke acceptance below.

## Full-State NMPC and controller compatibility (2026-10-16)

The new RPS98-inspired, locally authored 10-state acados Full-State NMPC uses
Crazyflow's native BodyRate physical input. It accepts timestamped reference
position, velocity and acceleration. Its pure reference-orientation calculation
and Task configuration are tested separately from acados/physical execution.
The six-case NMPC test suite passed, including a compiled acados/Crazyflow
tracking flight and consumption of a native position/velocity/acceleration
`Trajectory` within Navigation. A real short Tracking Benchmark finished with
20 synchronized decision-time probes: **7.97 ms mean** and **8.92 ms P95**
on the tested CPU. A 0.4-second Racing Play also executed with no solver
exception, but naturally did not complete the course during that short window.
These checks demonstrate interface feasibility, not tracking convergence or
multi-scene navigation capability. The subsequent native command smoke includes
EGO/SUPER Planner + Full-State NMPC execution across the declared OA scene set.

The revised Command Builder categorizes methods by required reference, rather
than by an unrestricted task × solver Cartesian product. It generates **550
distinct compatible commands** (42 Train, 380 Play, 128 Benchmark), and all 550
pass Hydra composition, including Full-State NMPC with EGO and SUPER planner
configurations, Racing's timed spline, and Tracking. Native Controller and
Planner execution is covered separately below. The reference-compatibility
decision and RPG Quadrotor Control adaptation status are recorded in
[ADR-0033](adr/0033-controller-compatibility-and-reference-tracking.md).

## Native Controller and Planner command execution (2026-10-16)

The Command Builder contains **172** non-policy Play/Benchmark entries:
8 Play Controllers, 120 Play Planner + Controllers, 8 Benchmark Controllers and
36 Benchmark Planner + Controllers. The source command catalog and per-case
results are stored locally in ignored `results/command-builder-native-20261016/`.
Its `live.html` lists each command with the terminal event, report, log, native
RScope record and MuJoCo WASM browser replay where actual frames are available.

The smoke runner passes the exact generated Hydra arguments to the installed
`python -m drone_playground.cli` rather than reinstalling the worktree's CUDA
environment through the equivalent `pixi run` entry. It overrides only the
flight budget/capture conditions: 0.14-second Task duration, one episode per
scene, CPU device, one environment, replay enabled and one replay per scene.
Benchmark scene suites are retained intact. Independent ROS Worker ports
50051/50052 (Play) and 50151/50152 (Benchmark) avoid concurrent planner
sessions. The machine-readable manifests keep command, source revision,
execution status, task outcomes, wall time and replay references.

All **172/172** native Controller and Planner commands completed their bounded
CLI run and produced a replay. The two initial Tracking + Attitude MPC cases
failed with the old full-SQP solver under the artificial 0.14 s Task setting.
This setting also increased the Figure-8 reference angular frequency from
0.314 to 44.88 rad/s, so the failure cannot be attributed to SQP alone.
They were rerun using SQP-RTI and produced actual short Crazyflow recordings.
The successful smoke cases include **256** browser replay records across the
scene suites. Most OA episodes terminate at the deliberately short smoke
timeout; this is an execution result, not a navigation score. The protocol
checks command compatibility, physical execution and replay export. It does
not replace the full Benchmark protocol for obstacle-avoidance success, policy
convergence, complete Racing completion or real-time control claims.

## Complete physical tasks (2026-10-17)

The 172 native-method commands require a separate **full-episode run** to
validate flight behavior. The previous 0.14-second tests are never reused as
full-flight outcomes. `tools/validate_full_episodes.py` derives each horizon
from the actual Task YAML: 20 s Tracking, 60 s Racing and 300 s Navigation.
The same supported Hydra command is used with one independent Episode per
scene, CPU execution, native RScope recording and sampled MuJoCo WASM playback.
The complete Benchmark suite retains every scene; statistical 25/100-episode
cohorts remain a separate, more expensive publication protocol.

The first full Tracking + Sampling MPC flight completed 20.0 s (10,000 physics
ticks) with `SUCCESS`, position RMSE **0.0318 m**, and a recording spanning the
whole episode. The first full Racing + MPCC flight completed all ordered gates
at **14.722 s**, also with a complete native/browser replay.

A separate check of the unchanged original LSY full-SQP controller on the
untruncated 20 s Figure-8 reference also completed successfully, with position
RMSE **0.0510 m**; the current SQP-RTI variant completed the same 20 s test
with RMSE **0.0596 m**. Consequently, the short-smoke solver failure was
confounded by the 0.14 s reference rescaling. It is not evidence that the
published full-SQP formulation was intrinsically invalid. The live full-flight
acceptance uses the frozen SQP-RTI revision for repeatability.

A full EGO/S01 navigation run exposed a previously invisible clock-interface
bug: at 0.4 s the 30 Hz Depth frame clock was represented as 0.40000003576 s,
slightly ahead of the exact 50 Hz control timestamp. The ROS Worker correctly
rejected that request as `STALE_INPUT`. The Python adapter now reconciles up
to two float32 representational ULPs without accepting real future sensor
timestamps. Its regression tests and a real EGO session with successive
planning solves at 0, 0.4, 0.8 and 1.2 s passed. The original failing trace
is preserved. The next full run reached 32.0 s, then identified a second issue:
a valid native trajectory ended before Sampling MPC's one-second fixed future
window, causing `STALE_OUTPUT`. The host tracker now pads only this prediction
tail with a stationary endpoint reference. An expired trajectory at the current
control time still fails. Both failure records are preserved for comparison
with the subsequent full-episode rerun.

The corrected EGO/S01 full-flight run reached the actual navigation goal at
**32.424 s**, with `SUCCESS` and native/browser replays covering the whole
flight. Other scenes and controllers are being evaluated under the same
complete-episode protocol without reducing the 300 s task limit.

The active full-flight catalog, resumable process status, terminal task events,
report links and replay URLs are in the locally ignored directory
`results/command-builder-native-full-20261017/` (`index.html`, `catalog.json`,
`manifest.json`, `runs/`). Every genuine `TIMEOUT` is checked against its full
Task horizon; wall-clock watchdog expiry is labeled `process_timeout` rather
than presented as a simulated task outcome. Until all commands have physically
terminated, the manifest—rather than the prior 172/172 short-smoke result—is
the source of truth for full-mission coverage.

### Repeated early failures and browser replay coverage audit

The first full-flight run reached case 66 with a large cluster of early
`METHOD_FAILURE` outcomes: EGO + Attitude MPC and Full-State NMPC stopped at
1.2 s; SUPER + Attitude MPC stopped at 0.3 s. Their planners previously
returned `SOLVED` trajectories. The common cause was strict comparison of
the exact host `t=now` controller stage against a float32-quantized ROS
trajectory first timestamp. Sampling MPC does not exhibit the same stage-zero
problem because its prediction samples begin at `t+dt`. The host now reconciles
only float32 clock representation error; real future or expired trajectories
still fail the validity check. Targeted regression cases cover 0.3, 1.2 and
32.0 s and reject a genuinely future plan.

`tools/audit_full_episode_replays.py` independently compares each physical
Episode's actual termination time with the first/last saved WASM frames. In the
original batch **66/66** recorded cases had a faithful browser recording of
their actual physical Episode. A 1.2 s recording ending in `METHOD_FAILURE`
is therefore **an early flight termination, not lost video footage**. The
audit distinguished **32 shared stage-zero boundary failures** and **six
separate native `NO_SOLUTION` responses**; the latter are real planner-level
failures, not resolved by timestamp alignment. Failed and superseded records
are retained, with historical attempts listed in the audit JSON.

After fixing the common interface, representative full-flight rechecks:

- EGO + Full-State NMPC / S06 (case 63): stage-zero `METHOD_FAILURE` at
  1.2 s disappeared, but the vehicle actually crossed the task boundary at
  **10.804 s** (`OUT_OF_BOUNDS`, not success).
- SUPER + Attitude MPC / S01 (case 47): completed the actual task at
  **33.804 s**, `SUCCESS`, versus the original 0.3 s failure.
- EGO + Attitude MPC / S01 (case 34): continued until **34.400 s**, when the
  native EGO planner returned `NO_SOLUTION`; no stage-zero exception remained.

Both the original failure and corrected replay remain available. The batch
was intentionally paused after repeated deterministic failures, rather than
continuing to burn CPU on the same defect. A resumable, fail-fast retry pass
will validate the remaining affected commands against their full Task horizons.

The full-episode acceptance was resumed on a fixed snapshot with the corrected
stage-zero comparison. Its local `live-audit-refresh.sh` updates the inspection
index and replay audit without interrupting the flight process. The new
`failure_detail` ledger and browser viewer distinguish native `NO_SOLUTION`,
stage-zero representation bugs, ROS session errors and physical termination.
Auditing explicitly skips actively running episodes while the replay serializer
has not yet published the final JSON: an in-progress recording is not a missing
or truncated replay.

Independent physical boundary inspection of the first **12 OUT_OF_BOUNDS**
episodes found nine reaching the Navigation height floor (0.5 m) and three
reaching the ceiling (6.0 m). The displayed first native plan point at each
recorded frame remained inside the nominal flight bounds, but sometimes came
close to them while the actual aircraft crossed the boundary. This does **not**
certify the entire upstream trajectory as collision-free or within the task
volume. These are true planner/tracker safety outcomes, not bugs in the RScope
serializer or a reason to relax Task bounds. The local
`results/command-builder-native-full-20261017/boundary-diagnostics.json`
contains each terminal pose and sampled planning-reference point. No privileged
global obstacle map, endpoint clipping or synthetic continuation was introduced
to improve the baseline score.

## Recorded Task Preview in the Environments browser (2026-10-17)

The [Command Builder](command-builder.html) now includes Isaac Lab-style task
selection, a selectable Task Preview backed by our native MuJoCo WASM player,
and a searchable list of already recorded combinations. The generated 550
commands remain the same; visualization does not create new simulation or
Benchmark results. `tools/build_task_preview_catalog.py` maps the original
full-episode Command Builder catalog, manifest and Episode CSV to a separate
ignored static JSON index. The currently captured example contains **172
native-method combinations and 256 recorded Episodes**: 119 `SUCCESS`, 66
`METHOD_FAILURE`, 67 `OUT_OF_BOUNDS`, and four `COLLISION`. These are counts
of existing individual Episode results, not confidence intervals or a
multi-seed algorithm ranking.

Each preview is attributed to the exact selected Planner/Controller/task and
recorded scene, with truthful terminal time and event. Benchmark suites expose
their own scene selector and mixed outcomes. Train/Policy selections that lack
a native policy recording display an explicit **task illustration**, never
misattributed checkpoint footage. A recording with early method failure ends
on the original physical terminal frame instead of looping or extrapolating.
EGO + Sampling MPC / S06 is an illustrative case: its record runs from 0 to
6.400 s (321 original browser frames), with the final ROS planner decision
`NO_SOLUTION`. It is neither `COLLISION` nor truncated video footage.

The browser regression opens a real recorded Tracking flight in autoplay,
selects the matching native S06 case and verifies the terminal `NO_SOLUTION`
label, switches Benchmark episodes, and selects Racing MPCC from the recorded
gallery while confirming the generated CLI method changes with the preview.
The updated viewer maintains one WebAssembly instance throughout selections,
loads each recorded MJCF/pose dataset by a same-origin message channel, and
supports optional translucent obstacle rendering in the embedded preview for
unobstructed aircraft inspection. Native geometry, terminal task outcome,
scene motion and original saved RScope state remain unchanged.
All assets remain local to the same SSH-forwarded HTTP origin. Recorded
flight previews never execute the physics solver or mutate source experiments.

## Convergence

Formal experiments retain their declared protocol and measured outcomes locally.
The public repository contains reusable training and evaluation behavior, not
private run matrices or phase-specific orchestration. Runtime checks and a
completed interaction budget do not establish convergence. Check task quality
and paired retention of previously successful episodes before making that claim.

## Local evidence preservation

Publication cleanup removed 441 development/evidence files from Git tracking
and verified that every local file remained present with the same SHA-256.
This includes five historical native-worker reports found inside the source
tree. Checkpoints, results, replays, and untracked research drafts were preserved.
Previous operating documents were snapshotted before rewriting. All 230 intended
public text files pass the language/private-path scan, and all public document
links resolve to retained files. Source, tests, licenses, and current English
documentation form the public repository. The cleanup and refactor are committed
as separate local commits; no push or history rewrite was performed.
