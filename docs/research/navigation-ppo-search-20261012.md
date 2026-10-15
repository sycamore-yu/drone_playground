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

Trace clarification at `2026-10-14T12:20:24Z`: representative S01 hybrid episode0
ends at `[95.2832,1.0376,2.5728]`, with goal `[98,0,3]`. Its remaining distance
is2.9394m:2.9082m horizontally and0.4272m vertically. During the final100
decisions the newer policy's mean normalized horizontal action is approximately
`[0.000117,0.000000666]`; the executed older vertical action is approximately
`0.00000147`. Under the saved +/-6m/s² action contract these requests are near
zero. At the same state, the newer policy's vertical request is about+0.97m/s²,
but the intervention suppresses it; the older horizontal request is about
`[2.03,-0.48]`m/s² and is also not executed. Thus the observed hybrid has a
near-hover closed-loop equilibrium even though its components disagree on how
to proceed. This is evidence about the mixed controller, not proof that the
unaltered newer policy lacks horizontal terminal control. Both GRUs see the
live hybrid states; no prerecorded action time series is substituted.

There is no configured2.94m stopping radius: the success radius is0.5m.
The old goal feature equals a body-frame version of
`delta / distance * min(3.0,distance)`. Below3m it is still a nonzero goal-error
vector, so the encoding itself does not command a stop at3m. The actual
stationary policy output is directly observable; a causal claim that the3m
feature transition creates that equilibrium needs a separate matched experiment.
The3m/s reference was introduced in implementation commit `b377e13` and is
also used by the default Task velocity cost and named perception cost. It is an
implementation assumption, not a user-imposed mission speed. The mission is
safe arrival as quickly as physically feasible; reference/reward changes must
remain separately identified experiments.

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

Runtime confirmation at `2026-10-14T10:58:48Z`: `depth_mixed_goal_s0` scored8/48
at update720 (D01:8), then0/48 at update1440 (43 collisions,5 out-of-bounds).
`depth_mixed_privileged_s0`, started at `2026-10-14T10:46:55Z`, scored8/48 at
update240 (S01:8),17/48 at update480 (S01:8,S02:8,D01:1), then0/48 at update720
(27 out-of-bounds,21 collisions). Both remain running within the original
budgets; neither individual factor has demonstrated stable convergence.
The early successes cannot be added across methods or checkpoints. Resolved
configuration diffs against `depth_mixed_s0` contain only the declared factor
and output path. Evidence is in
`tmp/navigation-ppo-replan-20261014/{first,critic}-worker-contract.json` in the
primary checkout. The original mixed control's full-screen counts at updates
720/1440/2160/2880/3600/4320 were0/0/14/26/0/0, recovered from worker stdout
and cross-checked against retained reports. Its later first improvement is
why these new pilots are not stopped solely on their first regression.

Fan's native patience starts counting at its minimum, so it would continue
through two further full screens after update2160. At `2026-10-14T10:46:05Z`
the optimizer instead requested a measured prune: twelve full screens were
all0/48 and4,485,120 training transitions had already been observed. The worker
saved update2209 (4,524,032 transitions); its original status is `interrupted`,
while the optimizer records `pruned` with the measured zero objective. This
released the Critic control's slot. The stop reason and signal timestamp are
preserved in `fan_base_s0/study_pruned.json`.

A separate CPU check of the actual Fan reward function found a task-incentive
conflict: at0.6m planar goal distance, zero speed/acceleration and safe clearance,
the current weights give+1.083049 per decision, although the0.5m goal sphere
ends the episode. With gamma0.995,1000 such decisions yield a discounted215.17
before any later terminal contribution. This is an analytical safe-hover
counterexample, not a simulated trajectory or proof of the held-out collision
cause. It motivates checking arrival incentives before further Fan learning-rate
or exploration sweeps. No reward change has been applied; the exact terms and
assumptions are saved in `tmp/navigation-ppo-replan-20261014/fan-hover-reward-audit.json`.

Historical comparison audit at `2026-10-14T10:57:46Z`: the accepted seed0
`scratch_depth_apg`, `scratch_safe_depth_shac`, `scratch_safe_lidar_apg` and
`scratch_safe_lidar_shac` configurations rotate through
S01/S02/S03/D01/D02/D03/S06/D06 during training. Their environment batches are
128 for Depth and64 for LiDAR. Their recorded training budgets are respectively
2,621,440;3,276,800;1,310,720;3,932,160 transitions. The current PPO pilots use
`train/procedural`, no fixed training-scene list and16 environments. Earlier
75-update acceptance runs additionally load APG initialization policies; their
reported new interactions exclude shared initialization and discarded trials.
These historical passes are therefore not algorithm-only controls for the
current procedural-to-fixed-map PPO question. This audit reads preserved
configuration/run records only, without loading historical v1 policies or
changing the current training/benchmark split. Exact paths and extracted values:
`tmp/navigation-ppo-replan-20261014/historical-comparison-configs.json`.

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

### Speed-free start-distance controls (2026-10-14)

User-authorized distance controls are implemented in `f9eb5c0`, outside the
integrated `v0.2` code. `--suite distance` launches separate Depth/LiDAR seed0
runs in order 20m, 80m, 10m, 30m, 50m. Within each sensor family only
`simulation.procedural.start_goal_distance_range_m=[d,d]` differs. This samples
actual 3D goal distance with collision-free starts and matched obstacle RNG.
Changing the old `min_start_goal_distance_m` to20 did not give20m starts.

All controls use relative goal position, world rotation input, privileged critic,
world acceleration commands, and Task `minimum_time_navigation`. The common
baseline removes the3m/s reference and velocity-tracking learning loss; elapsed
time, progress, clearance, action smoothness and safety penalties remain. Comparing
these runs with old recipes cannot isolate distance or reward effects. The matched
20m-versus80m controls provide that comparison within the new baseline.

Each pilot has2,211,840 transitions (4320 updates). Zero-success patience starts
only at that hard budget; external zero-only pruning is disabled. Uncalibrated
progress thresholds stay disabled. Every240 updates, an additional8-episode
procedural probe uses fixed seed_base1100000, separately saved under
`checkpoints/model_XXXXXX/training_distribution`. Actual initial distances are
recorded alongside goal progress, path length, height and failure events. Fixed
long-distance probes still rotate two scenes; a full six-scene6x8 screen runs
every720 updates or after a promising probe. The original23/25 and frozen
three-training-seed acceptance requirements remain unchanged.

The old Fan low-LR and VisFly hover-thrust jobs were paused at resume checkpoints
u786/u268, without pruning or altering historical configuration/results. New Depth
and LiDAR20m jobs have actually started. At the first Depthu240 checkpoint the
procedural probe was6/8 and the full screen15/48 (S01/S02/S03:0 each;
D01:8, D02:6, D03:1). LiDARu480 procedural probe was4/8; its rotating fixed
two-scene probe remained0/16. These are early results, not stable convergence.

Atu720 the matched procedural probe became Depth3/8 and LiDAR5/8, while both
complete fixed six-scene screens were0/48. Depth had40 collisions and8 S03
timeouts; LiDAR had48 collisions. Depthu240's15/48 checkpoint remains preserved.
The early drop therefore persists in the speed-free recipe and differs from the
historical lower-height collapse: no out-of-bounds event occurred in these full
screens. These observations establish a loss of measured long-distance capability,
without isolating its optimizer or representation cause.

Actual procedural recordings are in the primary worktree's
`results/navigation_distance_v01/inspection`: static, dynamic and dense scenes
with167/58/208 active obstacles, including0/37/135 moving obstacles. Their
recorded geometry and mocap motion were checked through the native RScope decoder.
These three recordings use historical Depthu2880 for scene inspection. A separate
recent Fan u720/D01 single-policy recording reaches about0.97m from the goal,
then times out at300s with near-zero final speed; this is distinct from the
earlier2.94m hybrid-policy diagnosis. Export now tolerates one float32 timestamp
ULP and skips submicrometre display capsules; actual recorded poses are preserved.

### Existing acceptance protocol

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

## Follow-up: completion retention, hover and lighter static scenes

Checked at `2026-10-14T14:41:37Z`. The analyzed speed-free baseline is
`f9eb5c0`; implementation fixes and the independent static-density cohort are
`60eb27b`, still outside integrated `v0.2`. Evidence below is from actual
frozen-policy rollouts and Task/physics counterfactuals, not reward-based acceptance.
Primary-worktree evidence: `tmp/navigation-stability-20261014/`.

### Measured capability loss

Depthu240 andu2160 were replayed from identical initial state hashes with
separate whole policies. No horizontal/vertical checkpoint splicing was used.

| Matched diagnostic | u240 | u2160 |
|---|---:|---:|
| D01, deterministic,8 episodes | 8/8 | 0/8 |
| D01, sampled actions,8 episodes | 5/8 | 0/8 |
| 20m procedural, deterministic,8 episodes | 6/8 | 6/8 |
| D01 realized-return Critic RMSE | 10.42 | 73.18 |

Late D01 failures are all collisions, mean flight5.54s, with height2.91–3.99m;
this is distinct from the historical lower-height collapse. Initial world
acceleration changed from about `[0.335,-0.119,-0.071]` to
`[0.926,-0.685,-0.147]`m/s² in the first matched episode. The policy became
more aggressive and its measured long-distance avoidance capability disappeared.
The sampled-action control also failed, so the late failure is not explained
solely by deterministic evaluation.

Equal short-distance totals conceal a lost capability: only5 of the6 previously
successful procedural episodes remain successful; world4 changes success to
collision, while world0 gains a success. Report the paired success-set retention
alongside completion count, not just6/8 versus6/8.

The late Critic predicts initial D01 value−69.07; sampled realized return is
−118.32. On1.6s prefixes it assigns higher bootstrapped returns to the ultimately
colliding late policy in all8 pairs than to the successful old deterministic
policy. At3.2s that ordering holds in only2/8 pairs, so it is evidence of a
short-prefix calibration problem, not proof of one universal value-ranking error.
Current PPO horizon32 at10Hz gives3.2s rollouts; gamma0.99/GAE lambda0.95 gives
an approximately1.6s exponential trace scale. Long avoidance therefore depends
strongly on value estimates beyond these short windows.

Realized-return calibration is conditional on the recorded policy, and differs
from stochastic, timeout-bootstrapped training targets. These diagnostics support
distribution coverage, conservative actor updates and value calibration as next
controlled experiments; they do not isolate a unique optimizer defect. LiDAR
also loses long-distance completion with much lower KL-skip rates, and both
policies have zero measured tanh saturation, ruling out either as a sufficient
common explanation. Both20m pilots have now completed their4320-update budgets
without acceptance.

### Two separate near-goal findings

The observed Fan u720/D01 policy settles near0.97m and times out. At recorded
poses, setting unrecorded acceleration/jerk costs to zero gives a reward upper
bound of−0.2101 per decision; those actual costs are nonpositive. The observed
hover cannot be described as collecting a positive immediate reward there.

Real-physics controllers initialized at those recorded final positions give:

| Initial distance | Hold for remaining10s: discounted return | Approach and terminate: discounted return |
|---|---:|---:|
| about0.97m | −16.57 | +4.48; arrival in1.52s |
| 0.51m | +81.98 | +4.18; arrival in0.37s |

Both counterfactual controllers share freshly initialized attitude, velocity and
controller history. This establishes reachability from the position and an
objective conflict; it is not a replay of the exact unavailable internal
controller state. The0.97m behavior is a suboptimal policy equilibrium. Its
unique learning cause is not yet isolated.

The0.51m counterexample independently proves a Fan reward defect: positive
`alive` and `hover` reward continues outside the0.5m success sphere, while
arrival terminates that reward stream without a separate arrival bonus. Correcting
this terminal incentive requires a named reward experiment. It is not a complete
explanation of the observed0.97m equilibrium and does not apply to the new
minimum-time Depth/LiDAR profile, which has elapsed-time cost and arrival reward.

Evidence files: `paired-summary.json`, `paired-old-stochastic-summary.json`,
`bootstrap-ranking.json`, `training-metrics-summary.json`,
`fan-reward-diagnosis.json`, `fan-boundary-diagnosis.json`, `findings.json`,
`diagnosis.png`, and the underlying full traces.

### Implemented retention fix and approved density reduction

The old rolling checkpoint policy retains fixed-scene best, but can delete a
procedural completion peak when all fixed scores are zero. Early LiDARu7205/8
was lost this way; its append-only event evidence remains, and deleted weights
cannot be reconstructed from it. Lateru36006/8 survives because it also scored
1/48 on fixed scenes; incidental overlap does not ensure capability retention.
`60eb27b` additionally
retains the best completed training-distribution checkpoint in explicitly opted-in
cohorts. It stays separate from formal selection. This is evidence preservation,
not an actor rollback or a claim that future updates are stable.

The generator already samples easy/medium/hard uniformly and static/dynamic with
equal probability; it is not an ordered curriculum. Original static counts are
41/81/162 with±8 jitter; dynamic50/100/200 with±8 jitter and about65% moving.
User-approved static-world counts are now multiplied by0.7, giving ranges
23–34,51–62,107–119. Dynamic worlds, obstacle sizes and sampled RNG remain
unchanged. Default scale1 preserves historical geometry identity.

Independent `--suite static-density` uses the two20m recipes and changes only
`simulation.procedural.static_obstacle_scale=0.7`; it is recorded through
optim-agent in primary `results/navigation_static_density_v01` and
`.optim-agent-runs/navigation_static_density_v01`. Original80m jobs were paused
with recovery checkpointsu241/u480;80/10/30/50m comparisons remain pending.
New lighter-static Depth/LiDAR pilots are running. Early shorter-distance gains
have not produced formal long-distance acceptance.

The three real density previews contain23/58/111 static obstacles and use the
existing Depthu240 actor only to inspect the new geometry. All installed-viewer
and native-decoder checks passed after trimming each replay directory to
`episode.mj_unroll` and `rscope_meta.pkl`. The branch's temporary viewing entry
`tmp/rscope-inspection-navigation-distance-v01/` lists the three lighter static
grades, a dynamic flight and the Fan hover. The complete preview index remains
at primary `results/navigation_distance_v01/inspection/README.md`.

## Forgetting research and matched distance curriculum, 2026-10-15

The user requested a literature/blog/community review, an independent curriculum
comparison, and an inventory based on all extant experiment records. This is a
training-sampling experiment, not an approved replacement of PPO or Crazyflow.
The integrated decision index remains the `v0.2` ADR directory; this feature is
not merged there.

Florensa et al.'s [Reverse Curriculum paper](https://proceedings.mlr.press/v78/florensa17a.html)
and [author blog](https://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/)
retain earlier start states and collect fresh trajectories from them to avoid
forgetting. The implementation here is a smaller fixed distance schedule, not
their adaptive SoID/Brownian algorithm. CLEAR's
[replay and cloning method](https://arxiv.org/html/1811.11682v2) uses V-trace for
off-policy corrections; inserting historical trajectories into the existing
PPO loss would not reproduce CLEAR. Teacher KL/cloning is a possible later
independent intervention, not part of the present curriculum comparison.

Two recipes are compared separately for Depth and LiDAR:

| Training budget fraction | Stationary control: 20/50/80 m | Curriculum: 20/50/80 m |
| --- | --- | --- |
| [0, 1/3) | .4 / .3 / .3 | .8 / .1 / .1 |
| [1/3, 2/3) | .4 / .3 / .3 | .2 / .6 / .2 |
| [2/3, 1] | .4 / .3 / .3 | .2 / .2 / .6 |

Each pair changes only `simulation.procedural.start_sampling` and output path.
Both use the same 2,211,840-interaction ceiling, current minimum-time Task
reward, world acceleration and heading observation, static scale .7, optimizers
and evaluation seeds. The stage is passed explicitly from saved host
interaction counts; it changes only future episode resets and never restarts
optimizer state or live-world recurrent memory. Resuming with a different
curriculum budget is rejected. The sampler uses the existing collision-free
candidate and clear-corridor fallback construction at exact Euclidean distance.

Training-distribution evaluation omits sampling progress and therefore always
uses the stationary mixture for both recipes. Six fixed long-distance scenes,
25 episodes per scene, at least23/25 per scene and three independent training
seeds remain the formal acceptance rule. Curriculum phase and actual reset
counts for each distance are logged. Equal nominal mean weights across three
interaction phases do not guarantee equal reset counts: episode durations
depend on the policy. Diagnostic comparisons must disclose observed counts.

The four seed0 cases are `depth_distance_mixture`, `lidar_distance_mixture`,
`depth_distance_curriculum`, `lidar_distance_curriculum` in the existing finite
`navigation_stability_fixes_v01` queue, before pending value-clipping controls.
Current workers continue. Seed1/2 promotion still requires accepted seed0;
short-horizon integration checks do not constitute convergence evidence.

The full read-only inventory and literature/action matrix are in the primary
checkout's `results/navigation_research_audit_20261015/{README.md,runs.csv,runs.json}`.
It separates actual updates from queued configurations, from-scratch from
initialized policies, six-scene48/150 reports from historical eight-scene64/200
reports, and extant reports from scores surviving only in scheduler history.
Historical configuration defaults are not reconstructed and old weights are
not loaded. A score such as initial VisFly1/48 must not be rounded down to the
later variants'0/48; all remain below acceptance.


## Follow-up: matched stability repairs and research, 2026-10-15

Matched static0.7/mixed-start seed0 runs finished: Depth best21/48 at u2880, final2/48; LiDAR best11/48 at u3360, final0/48. In identical S01 initial conditions, sampled whole-actor rollouts changed8/8->0/8 and realized-return Critic RMSE12.43693->55.12588. These reproduce regression without the deterministic-mean/stochastic-behavior explanation and do not identify a unique optimizer cause.

The new primary cohort `results/navigation_stability_fixes_v01/` declares12 pilot cases: each original sensor separately tests quarter Actor LR,128-step rollout,lambda1,value clipping off, and lambda1 against its completed128-step control; two Fan arms share speed-penalty removal,static0.7 and mixed starts and differ only in the arrival reward contract. Each original-sensor case retains2,211,840 interactions; each Fan arm8,847,360. Seed1/2 promotions require actually accepted seed0. Fixed six scenes,23/25 per scene and independent frozen testing are unchanged. optim-agent uses explicit ask/tell and the finite scheduler runs up to4 GPU trainers with a CPU-only manager.

The new named Task profile `fan2025_arrival` is implemented at c6f3da5, with JIT-safe speed-penalty disabling at1e1dac6. It negates the positive alive occupancy term, removes the ongoing hover bonus and pays5 once on actual3D SUCCESS. Historical fan2025 equations remain executable and unchanged for their original positive weights. Published Fan hovering aims to maintain flight at the goal until a new target, explaining why that continuing objective conflicts with immediate arrival termination here. No policy-invariance or global minimum-time optimality claim is made for this adaptation.

The real-physics D01 comparisons all passed across8 worlds: at0.51m, old hold81.98097 vs arrive4.18081 becomes new hold-33.49931 vs arrive3.23695; at the observed approximately0.97m pose, new hold-37.06775 vs arrive-2.37915 in world0. Approach uses equal freshly initialized velocity,attitude/controller history and succeeds in0.368s/1.516s. This proves reward ordering and reachability, not learned PPO convergence or the unique cause of the original0.97m hover.

On identical frozen sampled u4320/S01 trajectories, target RMSE to complete realized returns is45.97076 for32/lambda.95,44.59601 for32/lambda1,41.38159 for128/lambda.95 and34.43946 for128/lambda1. Whole-trajectory lambda1 is near numerical zero. This isolates target bias and surviving rollout-end bootstrap error; the network parameters were not trained by this comparison. lambda1 is a task-specific bias/variance hypothesis, not the general recommendation of the large-scale PPO study. Post-training CPU calibration samples each best/latest whole policy in S01/D01 and separates TIMEOUT because its saved training contract still bootstraps final values.

The quarter-LR Depth training probe went2/8->0/8->4/8->1/8 at u240/480/720/960. The4/8 policy retains only1/2 of the earliest successful identities: lower Actor LR alone has not established stability. Paired success sets, plus copied full recovery snapshots, prevent a higher total from silently replacing lost abilities in diagnostic selection. The archive originally waited for a full six-scene screen; that missed the u240 recovery weights before rolling cleanup. The missing weights are explicitly recorded as unavailable, not reconstructed. Every subsequent positive training probe now archives its whole recovery checkpoint before full screening. These diagnostic recovery snapshots are not formally accepted exported Actors.

Twelve paper/geometry/JIT checks passed; the added disabled-speed JIT check passed separately;6/8->6/8 with5/6 retention passed the schedule self-check. No packaging or full-platform rerun was needed. Code remains unmerged into v0.2.

[Complete research sources, experiment design and current evidence](/home/tong/tongworkspace/simulation_dev/new_drone_playground/results/navigation_stability_fixes_v01/README.md), [live study](/home/tong/tongworkspace/simulation_dev/new_drone_playground/results/navigation_stability_fixes_v01/study.json), [paired identities](/home/tong/tongworkspace/simulation_dev/new_drone_playground/results/navigation_stability_fixes_v01/paired-retention.json), [archive and recorded gap](/home/tong/tongworkspace/simulation_dev/new_drone_playground/results/navigation_stability_fixes_v01/capability-archive.json).
