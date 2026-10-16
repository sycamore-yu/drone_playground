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

The optional MJCF scene extensions originally yielded 486 unique commands
(42 Train, 334 Play, 110 Benchmark). The mainline Racing control integration
adds both MPCC and JAX perception-aware MPC to Play and Benchmark, bringing
the builder to **490 unique commands** (42 Train, 336 Play, 112 Benchmark).
All 490 configurations compose through Hydra. The original
230-command core suite retains its prior runtime evidence. Expanded commands
require additional real method execution before claiming the same coverage;
configuration composition alone is not runtime validation.

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
short Crazyflow flight. The 490-configuration Builder Hydra test passed.
Replay, Autoresearch projection, training media, paired benchmark and Play
tests passed **18 focused tests**. The MuJoCo WASM viewer built successfully
with `npm ci` and Vite. Its mainline build was tested independently in headless
Chrome against the recorded MPCC flight: 250 frames, eleven native Crazyflie
meshes, and eight Mocap bodies loaded. Browser Play advanced from frame 100 to
118 without JavaScript errors. A short JAX perception-aware Crazyflow Play also
completed at its 0.16-second test deadline without a method exception. A running
Dashboard from an older worktree is not automatically replaced by a Git commit.

The earlier 490-command candidate and complete CPU test suite were not run
end-to-end after that selective integration. The current 550-command revision
also requires a full runtime sweep before release acceptance. These remain
separate from configuration composition and representative method checks.

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
multi-scene navigation capability. EGO/SUPER ROS worker execution with the new
NMPC has not yet been rerun.

The revised Command Builder categorizes methods by required reference, rather
than by an unrestricted task × solver Cartesian product. It now generates
**550 distinct compatible commands** (42 Train, 380 Play, 128 Benchmark); all
550 passed Hydra composition, including Full-State NMPC with EGO and SUPER
planner configurations, Racing's timed spline, and Tracking. The corresponding
full runtime command sweep has not been completed. The reference-compatibility
release proposal and RPG Quadrotor Control adaptation limitations are recorded
in [ADR-0033](adr/0033-controller-compatibility-and-reference-tracking.md).

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
