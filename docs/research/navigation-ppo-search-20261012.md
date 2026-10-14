# Navigation PPO: matched controls and convergence acceptance

## Current branch and evidence

Checked at `2026-10-14T03:03:09Z`. The integration branch is `v0.2` at
`3345f03`; reward ablations (`c60ab63`) and Scratch Matrix (`6480bc9`) are
already integrated. Their clean worktrees were removed after checking that
neither held results and archiving every temporary evidence file with SHA-256
verification. The manifest is at
`new_drone_playground/tmp/branch-organization-20261014/manifest.json`.
SANDO and the independent Brax branch remain separate.

Paper branch merge `051e455` incorporates the integration snapshot, its canonical
ADR numbering and snapshot sensors. Fan raw history consumes hits at their
acquisition pose rather than removed rolling-scan fields. The merge passed
55 targeted paper/sensor/search tests and two Fan snapshot history tests.
Paper adaptations remain unmerged into `v0.2`; convergence is not established.

The former `navigation_ppo_search_v01/study.json` and `paper_ppo_v04/study.json`
are historical observations. Actual run records identify Depth update1190 and
Fan margin2 update3359 as interrupted; no old trainers were active at audit.
Preserved screens show Depth 1/48 and Fan margin2 8/48, with all remaining
outcomes colliding. VisFly shared is 0/48 with collisions and boundary exits.
Historical Depth 17/48 used shorter training starts and is not a matched >=80m
control. The original four-architecture search did not finish.

Old Fan/VisFly controls also used different screening frequencies, patience and
interaction budgets from their proposed treatments. Their scores cannot establish
single-variable improvements. New runs use fresh matched controls and the current
sensor contract. Historical results and checkpoints remain at their original paths.

## Active experiment

The result root is `new_drone_playground/results/navigation_ppo_convergence_v02`.
Depth and LiDAR seed-zero >=80m controls started from random parameters at
`051e455`. Their actual source hashes and configurations are in each `run.json`
and `config.yaml`. Changes after this commit repair scheduler selection and
interrupted budget-screen recovery; the sensor, reward, networks and PPO update
used by these controls are unchanged.

Both long-start controls finished by `2026-10-14T03:51:26Z`:

| Control | Stop update | Best checkpoint | S01 | S02 | S03 | D01 | D02 | D03 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Depth seed0 | 3600, early stopped | 2160 | 8/8 | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| LiDAR seed0 | 2880, early stopped | 2880, tied zero | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

Depth's final screen fell to 0/48 (29 collisions, 19 boundary exits); its
16/48 checkpoint remains retained. LiDAR's final screen has the same outcome
counts. These are diagnostics, not full validation or frozen acceptance.
Mixed-start trials now run independently from random initialization. Their saved
recipes are checked against each control, excluding only output location.

The user approved a training-start-distribution ablation on 14 October 2026,
recorded at `2026-10-14T03:03:09Z`. It uses the existing collision-free mixed
sampling by setting `min_start_goal_distance_m=0`. About 35% of sampling draws
use the near-goal region and the rest use the broad course; collision rejection
and the existing >4m separation still apply. No curriculum framework is added.
Formal scenes, long-distance benchmark starts, success definitions and thresholds
remain fixed. This parameter decision does not approve a new observation or model.

| Family | Matched control | Single-variable candidates |
| --- | --- | --- |
| Depth CNN/GRU | Depth A, >=80m starts, actor LR 2e-4 | Mixed starts; world rotation input; fixed heading; LR 5e-5; existing 12-feature goal input |
| LiDAR PointNet/GRU | LiDAR D, >=80m starts, actor LR 1e-4 | Mixed starts; world rotation input; fixed heading; LR 2.5e-5; existing 12-feature goal input; batch64; privileged Critic; velocity window1; horizon128; task-only reward |
| Fan | Margin2 equations, fresh current-sensor control | Actor LR 2.5e-5; initial log std -1.5 |
| VisFly | Shared visual encoder, fresh current-sensor control | Initial log std -1; thrust range 0–2 times weight |

There are 28 seed-zero slots after the regression-guided extension below.
Original Depth/LiDAR controls run first. The two new Depth mixed-start
observation/value controls have priority after a measured mixed-start regression.
Their combination waits for both controls to finish. Subsequent allocation uses
the parent score, failure events and idle sensor modality. All candidates require
a finished matched control; the earlier 24-slot history remains preserved.
A frozen-benchmark pass promotes the same recipe to independent seeds1 and2.
Once a sensor modality has three accepted seeds of one recipe, no competing
recipe or replication for that modality is launched. Interrupted trials remain
visible and do not prevent independent candidates from running; their saved
state is preserved for explicit contract-checked CLI resumption.

The user requested additional LiDAR convergence candidates and higher GPU
utilization on 14 October 2026. These candidates reuse existing options:

| Candidate | Change from `lidar_base` | Question tested |
| --- | --- | --- |
| `lidar_batch64` | 16 to 64 environments | Does a larger sample batch improve PPO learning and GPU throughput? |
| `lidar_privileged` | Sensor Critic to existing privileged Critic | Does true state/geometry improve value estimation while Actor input stays fixed? |
| `lidar_velocity1` | Causal velocity window 30 to 1 | Does an immediate velocity target improve credit assignment? |
| `lidar_horizon128` | Rollout 32 to 128 steps | Does longer recurrent context/bootstrap spacing help? This also increases samples per update. |
| `lidar_task_only` | Named Liu cost weight 0.5 to 0 | Does the existing Task reward suffice without the adapted differentiable cost? |

The Critic choice needs two mutually exclusive existing flags, sensor=false
and privileged=true; it is one categorical treatment. The privileged input
already contains goal-relative position, time, clearance and ideal rays.
No network interface, dynamics or new reward formula is introduced.
Task-only retains Task-owned progress, clearance, altitude and terminal terms;
it does not remove collision termination or change benchmark success.

Batch and horizon treatments keep evaluation/pruning/sample budgets equal in
environment interactions. They consequently perform fewer optimizer updates.
The batch64 and horizon128 candidates both collect 2,048 samples per update,
versus 512 for the control, enabling a secondary comparison of independent
worlds and temporal context. They do not isolate these effects from update count.
Only the explicit candidate factor and the derived update cadence differ.
All candidates start from random weights, keep >=80m procedural training starts,
and retain the fixed six-scene benchmark. Combinations require subsequent,
separately identified runs after individual evidence is available.

Depth mixed-start training reached 26/48 at update2880 but ended its budget at
update4320 with 0/48; its best checkpoint is retained. The independent
`depth_mixed_lowlr` follow-up inherits that mixed-start recipe and changes only
Actor LR from 2e-4 to 5e-5, starting from random weights with the same interaction
budget. Its parent is `depth_mixed`, not the long-start baseline. This targets
the observed regression and frequent KL skips without silently resuming the
finished run or combining observation changes. The orientation-only trial
continues independently.

The frame diagnostic follows a reproduced ambiguity: the original navigation
state encodes body-frame velocity, target direction and up, while acceleration
is decoded in world coordinates. `heading=target` changes vehicle yaw without
providing world yaw to the Actor. In empty returns, two rotated positions with
matching body-relative targets can therefore have identical instantaneous policy
inputs but need different world actions. The existing `heading=fixed` option
is a narrow test of this issue; it also changes where the camera points and
does not prove that the ambiguity caused convergence failure. This test changes
only heading; it retains >=80m training starts and is separate from mixed starts.

The user approved keeping world-frame actions and adding Actor orientation,
recorded at `2026-10-14T03:53:53Z`; see [ADR-0023](../adr/0023-navigation-world-rotation-observation.md).
The `world_rotation` candidates append the body-to-world rotation matrix through
the existing observation configuration. They retain target heading and >=80m
training starts, so each differs from its original control by only this input.

CPU replay of a copied Depth update2160 and LiDAR update2880 training state
reproduced behavior-policy GRU memory exactly. Replaying the same 32-step rollout
after one PPO update changed the next normalized action by mean/max
0.00157/0.00684 (Depth) and 0.00127/0.00949 (LiDAR). Older boundary memory stayed
fixed, so this does not rule out accumulated drift. It provides no current basis
for changing recurrent-state handling. Raw diagnostics and scripts are in
`new_drone_playground/tmp/navigation-ppo-review-20261014/memory-drift`.

## Screening and budgets

Every 122,880 transitions, evaluate a rotating static/dynamic pair, eight
episodes each. Every third checkpoint, screen all six primary scenes.
Any successful probe triggers a full screen at that checkpoint. The budget
boundary also forces a full screen; interrupted boundary evaluations must be
retried on resume without another training update or substitution by a probe.

Only complete six-scene screens affect ranking and patience. The allocation
score is `min(scene success rates) + 0.25 * mean(scene success rates)`.
Three non-improving full screens trigger early stopping after the minimum
budget, with improvement threshold 0.005. Report comparisons at matched
interactions; a maximum observed score alone is not an ablation conclusion.

| Family | Minimum transitions before pruning | Maximum transitions per pilot |
| --- | ---: | ---: |
| Depth/LiDAR | 737,280 | 2,211,840 |
| Fan | 4,423,680 | 8,847,360 |
| VisFly | 1,474,560 | 4,423,680 |

The initial run used two GPU training processes. The 14 October expansion
measures warmed throughput with up to four concurrent training processes,
including the ongoing Depth trials; concurrency is an execution setting,
not a training recipe change. Compare actual end-to-end samples/s and warmed
update time, excluding first compilation from the latter. GPU busy percentage
alone is not evidence of higher throughput. Admission checks RAM,
VRAM and I/O pressure, without adding per-service CPU, memory or swap quotas.
The scheduler preserves the best complete screening checkpoint and the recent
resumable states through the existing Results policy. Missing scores stay unknown.

### Feedback allocation, 14 October

The user approved the acceleration work on 14 October. The feature-branch
scheduler policy `navigation-feedback-20261014` replaces the fixed candidate
priority list. It gives an idle modality first access to an available slot,
preserves accepted-recipe seed confirmation, and ranks the existing bounded
candidate pool using its parent's measured best/latest complete screen and
collision/out-of-bounds outcomes. A regressing parent prioritizes its lower-LR
candidate; collision failures prioritize goal/task-reward/horizon diagnostics.
This is a local, explicit proposal policy; no model sampler or new backend runs.

Each launch writes its parameters, command, output, source commit and selection
evidence under `--optim-directory` before `optim_agent.Study.ask(params)`. The
manager calls `tell` once when real results finish, including recovery after a
manager restart. Intentional stops are `pruned` with the latest valid full-screen
metric; the best retained checkpoint remains separately recorded. Partial probes
and absent evaluations never become zero-valued objective observations.
Reserved-but-unlaunched proposals reuse their exact saved command and must
resolve to the same config hash. Launcher timeouts are recorded separately and
never overwrite worker-owned `run.json`; an ambiguous launch without a worker
record is held for reconciliation. Pending pruning requests are retried after
manager restart until the signal is acknowledged.

Native LiDAR pilots now have an additional scheduler stop: after at least
1,105,920 interactions, stop if two complete screens have failed to improve by
0.005 since the 737,280-interaction minimum. A new improvement retains the
original 2,211,840 maximum. Depth, Fan and VisFly retain their previous budgets.
This applies to running LiDAR workers without rewriting their saved configs:
`study_pruned.json` records the policy and evidence, SIGINT requests a resumable
checkpoint, and the worker's original interrupted run record is preserved.
The scheduler records it as pruned after the process exits. Comparisons against
older pilots must use matched interaction counts and identify the budget policy.

A controlled comparison used the same v2 native-LiDAR checkpoint for 96 repeated
updates at each concurrency, with warmed workers and explicit synchronization:

| Concurrent workers | Transitions/s | Mean four-update job seconds |
| --- | ---: | ---: |
| 2 | 1274.85 | 3.20 |
| 3 | 1256.08 | 4.86 |
| 4 | 1248.27 | 6.51 |

The selected limit is two: choose the smallest concurrency within 5% of the best
measured throughput. This measures one fixed LiDAR training workload, not full
mixed-sensor training or convergence. Existing excess workers drain normally;
the manager does not terminate useful jobs to meet the lower limit. All live
trainers were temporarily suspended and resumed during measurement. Frozen
checkpoint files were hash-verified unchanged. An initial measurement attempt
failed its worker-warmup check, was discarded, and was repeated with a barrier.
Raw evidence is in the primary checkout's
`tmp/navigation-ppo-review-20261014/concurrency-fixed-v2.json`.

### Train/screen consistency audit

The 14 October audit instantiated the saved Depth mixed, Depth lower-LR and
LiDAR privileged-Critic environments and compared their training and S01
screening contracts. Action conversion, timestep, physics frequency, sensor
specification, observation options and Task contracts matched. For each run,
best-versus-final checkpoint configurations matched, as did all six screens'
geometry, Task contracts, sensors, episode counts, seed partition and training
seed. Screening uses the current actor parameters directly and the same
environment observation and transition functions; PPO samples actions during
training, while screening uses `tanh(raw)` deterministically.

The deliberate distribution differences remain substantial. In this diagnostic
eight-world reset sample, mixed starts had distances 8.67–84.61m, long-start
training 82.13–94.13m, and fixed S01 starts 95.79–96.24m. These are sampled ranges,
not distribution bounds. Procedural training resamples geometry; each screening
scene uses one fixed map and eight small start/velocity/yaw perturbations.
Therefore the eight outcomes within a scene are correlated, and jumps by several
successes do not establish stable generalization. This audit found no contract
drift. Actual episode-table seed/world/geometry identifiers also matched between
best and final weights for all 48 screening episodes in each audited run.

A subsequent frozen-policy diagnostic evaluated Depth mixed checkpoints 2880 and
4320 on the same eight procedural episodes, seed base 1,000,000, deterministic
actions, and unchanged training distribution. Success dropped from 7/8 to 4/8
(the final failures were three out-of-bounds and one collision), alongside the
fixed-screen drop from 26/48 to 0/48. This supports in-distribution policy
regression as well as a generalization gap; eight diagnostic episodes do not
quantify the contribution of either. No optimization updates or formal acceptance
results were changed. Reports are under the primary checkout's
`tmp/navigation-ppo-review-20261014/training-distribution-probe/`.
Evidence: primary checkout
`tmp/navigation-ppo-review-20261014/screening-contract-audit.json`.

### Paired regression diagnosis, 14 October

Audited at `2026-10-14T08:34:07Z`, feature source `8ca73c8`, using the retained
Depth mixed-start checkpoints at updates 2880 and 4320. These are frozen-policy
diagnostics, not additional optimization or acceptance results. `7/8` means
seven successes in eight fresh procedural episodes, one per evaluation world;
this is re-evaluation on the training distribution, not eight saved training
trajectories. The training run itself used sixteen parallel environments.
`26/48` means the sum over six fixed maps, eight perturbed episodes per map.
Neither denominator counts training seeds, updates or independent fixed maps.

| Fixed scene | Update 2880 | Update 4320 | Final failure events |
| --- | ---: | ---: | --- |
| S01 | 8/8 | 0/8 | 8 below the lower height bound |
| S02 | 8/8 | 0/8 | 8 below the lower height bound |
| S03 | 0/8 | 0/8 | 8 collisions |
| D01 | 8/8 | 0/8 | 8 below the lower height bound |
| D02 | 2/8 | 0/8 | 8 below the lower height bound |
| D03 | 0/8 | 0/8 | 3 below the lower height bound, 5 collisions |

All 26 previously successful episodes became lower-height violations. All 35
out-of-bounds events in the final screen crossed z=0.5m; none crossed a lateral
or upper-height bound. This establishes loss of altitude control as the direct
failure mechanism accounting for the observed screen-score regression. It does
not identify the unique optimizer-level cause of that policy change.

The diagnostic runner uses the existing actor, public method execution,
environment transition and episode objective. It asserts that the instrumented
original policies reproduce every recorded episode event. Added instrumentation
changes XLA fusion: fixed-scene flight times differ by up to 0.521 seconds, so
the traces are not claimed to be bitwise identical. Paired procedural starts and
sampled scene-state arrays were additionally fingerprinted. Checkpoint/config
digests, scripts, traces and aggregate assertions are retained under the primary
checkout's `tmp/navigation-ppo-review-20261014/regression-cause/`.

| Diagnostic intervention | Procedural cohort | Fixed S01 cohort |
| --- | ---: | ---: |
| Original update2880, deterministic | 7/8 | 8/8 |
| Original update4320, deterministic | 4/8 | 0/8, all lower-height violations |
| Update4320 with sampled actions | 4/8 | 0/8, all lower-height violations |
| Update4320 horizontal actions, update2880 vertical action | 7/8 | 0/8, all timeouts |

The vertical-action intervention maintains a separate GRU for each policy while
both observe the same visited states. It restores the lost procedural successes
and removes all eight S01 height violations. In S01, however, this hybrid settles
about 2.94m from the goal until the 300s deadline. Resetting only the newer
policy's GRU once at 45s still gives eight timeouts. Switching the horizontal
actions to the earlier policy at 45s instead gives eight collisions near the far
wall. These interventions do not provide a deployable repair or establish an
independent terminal-control defect in the unmodified late policy: combining
policies changes visited states and control coordination. They show that fixing
the observed height symptom alone does not establish task acceptance.

Reward did penalize these failures. Immediately before the 35 lower-bound
terminations, the final Critic predicted values from -91.63 to -33.57 (mean
-62.75), whereas the next terminal reward was -260.77 to -260.59 (mean -260.69).
The total includes the base failure cost 20 and configured out-of-bounds cost 240.
On the paired deterministic episodes, mean discounted return also deteriorated:
procedural -22.46 to -68.82; S01 -12.91 to -167.36. Thus these probes do not support
the claim that the late policy improved its actual objective by exploiting a
missing failure penalty. A sampled-action replay also failed on all S01 cases;
one action-noise seed does not estimate the full stochastic-policy value.

The leading upstream concern is inaccurate value/advantage estimation with
limited state information. The original ten-feature observation has no explicit
absolute height or goal distance; the goal-vector magnitude is capped beyond 3m,
and the Critic is feedforward. At 10Hz, rollout horizon 32 covers 3.2s, while direct
GAE residual weights decay as `(0.99 * 0.95)^k`; later effects depend heavily on
the learned bootstrap. These code facts and the observed prediction errors
motivate separate goal-input and privileged-Critic controls; they are not causal
proof that either change will prevent regression. Actor KL skipping is also not
a sufficient explanation: the long-start lower-LR run still fell from 20/48 to 0
with a mean post-peak skip fraction of 5.05%, versus 56.25% for Depth mixed.

Sparse training logs agree with deterioration in the mixed run: the logged
completed-episode success fraction falls from 60.3% in the 600-update window before
the peak to 52.8% afterward, and the out-of-bounds event fraction rises
from 7.1% to 26.8%. These are summaries of every tenth logged update, not exact
whole-run episode rates. LiDAR's persistent collision-dominated zero scores
remain a separate failure mode; this Depth diagnosis is not evidence of a shared
root cause or convergence for LiDAR/Fan/VisFly.

### Experiment response to the regression diagnosis

On 14 October the user requested revised experiments and continued work toward
convergence. Policy `navigation-regression-controls-20261014` adds four pilots
using existing options. The matched parent specifies the full training recipe;
each child adds one experimental factor, with fresh weights and training seed0.

| Candidate | Matched parent | Only added factor |
| --- | --- | --- |
| `depth_mixed_goal` | `depth_mixed` | Explicit normalized height error and remaining goal distance |
| `depth_mixed_privileged` | `depth_mixed` | Privileged Critic replacing the sensor Critic |
| `depth_mixed_goal_privileged` | `depth_mixed_goal` | Privileged Critic replacing the sensor Critic |
| `lidar_goal_privileged` | `lidar_privileged` | Explicit normalized height error and remaining goal distance |

The three Depth cases retain the approved mixed start distribution. The LiDAR
case retains its parent's >=80m starts. Goal input is available to both the
Actor and the configured Critic; privileged information remains Critic-only.
The standalone long-start Depth goal trial reached4/48 and ended0/48, so it
does not answer the mixed-start question. The new combination tests interaction
between observation and value estimation after both Depth main effects have
been measured. The first two controls temporarily precede idle-modality
allocation; independent accepted-seed confirmations retain higher priority.

At `2026-10-14T10:31:59Z`, the just-started `visfly_lowstd_s0` trial received a
graceful checkpointed stop request to release one slot. This is resource
reallocation, not a measured failure: it had no complete six-scene screen and
its optimizer value must remain null. The worker's original `run.json` is
preserved; the scheduler records the reason in `study_pruned.json`. Fan keeps
its existing minimum budget of4,423,680 transitions before its native stopping
rule can stop the zero-score run. Concurrency remains two.

Depth pilots retain the2,211,840-transition ceiling and existing early stopping;
LiDAR retains its existing measured pruning rule. All six-scene screening seeds,
best-checkpoint retention,25-episode full validation, independent frozen
benchmark and three-training-seed requirement are unchanged. An unmeasured
canceled proposal is recorded as pruned with no objective, not as zero success.
No action, reward, physics, network or checkpoint contract changes in this step.

Runtime check at `2026-10-14T04:55:10Z`, scheduler source `600246a`:
Depth world-rotation plus LiDAR world-rotation, batch64 and privileged-Critic
workers all performed real GPU PPO updates with finite losses and gradients.
Saved configurations match their declared factors; batch64 additionally changes
only the derived evaluation/checkpoint/pruning update counts. The four-worker
window after initial compilation and Depth's full screen measured about
1,445 training transitions/s, versus about 1,360 in the earlier two-worker
window. These 60-second windows contain different recipes and checkpoint work;
they establish useful concurrent execution, not a controlled 6% speedup claim.
An intervening window with Depth evaluating measured 1,190 transitions/s.
GPU busy was 99–100%, observed VRAM about 11.1 GiB of 24 GiB, and available
host RAM about 41 GiB. The manager remains at max-parallel4; no higher-count
throughput claim is made. Measurements and resolved-config comparisons are in
`new_drone_playground/tmp/navigation-ppo-review-20261014/gpu-load-*.json` and
`lidar-expansion-live-config-diff.json`. The queue/config tests passed 30 cases;
Standards and Spec code reviews each found no actionable proved defect.
These checks do not establish LiDAR convergence. Depth orientation's best
complete screen is 1/48 at update2400; its update3600 screen is 0/48.

New workers also preserve each full screen's per-scene report, score and patience
in the existing TensorBoard event stream before checkpoint retention can remove
its directory. The prior writer recorded only the pass flag and overwrote its
single latest-evaluation index; pruned historical reports cannot be reconstructed
from those flags. The already-running controls keep their original writer;
available reports were copied read-only to `tmp/navigation-ppo-review-20261014/`
for this audit. This recording fix does not change optimization or acceptance.

## Acceptance and execution

A candidate must pass full validation: each S01/S02/S03/D01/D02/D03 scene at
least23/25 safe arrivals. The same selected checkpoint then runs a separate
frozen benchmark using independent seeds. Stable acceptance requires three
independent training seeds of the same recipe. Training reward, probes,
matched-control tests and budget exhaustion are not acceptance. Results from
different checkpoints, recipes or seeds are not spliced into a passing policy.

From the experiment worktree, the supported Pixi task is:

```bash
pixi run navigation-ppo-search status --root /absolute/path/to/results/navigation_ppo_convergence_v02
pixi run navigation-ppo-search run --root /absolute/path/to/results/navigation_ppo_convergence_v02 --max-parallel 2 --optim-directory /absolute/path/to/.optim-agent-runs/navigation_ppo_convergence_v02
```

The manager interpreter must provide `optim_agent` and `omegaconf`; `--python`
can separately select the existing training interpreter. These are operator-tool
requirements, not new runtime dependencies of the drone package. On this host,
the existing `tmp/optim-agent-venv` is reused with the shared Pixi site-packages
on its manager-only `PYTHONPATH`; worker `PYTHONPATH` remains the feature `src`.

The task accepts `--python` to reuse an existing Pixi interpreter across worktrees;
the launcher explicitly sets this worktree's `src` as its import path. Commands,
resolved configurations, source identity and per-scene outcomes are stored per
run; `study.json` is the scheduler snapshot. The finite queue ends with acceptance
or an explicit negative/incomplete result, not an unbounded parameter sweep.
