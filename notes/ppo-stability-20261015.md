# PPO stability experiment design

Objective: the user attachment `6bfd2216-f69e-4121-97b8-8f1181e64d2a/pasted-text-1.txt`,
including stable scratch Depth/LiDAR PPO and the original 18-cell BPTT/SHAC/PPO matrix.
This is a working experiment design, not evidence of convergence or a new architecture.
The sections below retain their chronological protocol changes. The current user-selected
protocol is screening + one full validation + independent frozen testing, with different
algorithm rewards allowed; see `docs/acceptance.md` and `docs/status.md`.

## Identity and protection

The isolated branch starts at `351d855fc5a122f0356e91f669b48c9b6bdc95bb`, the
source recorded by the two collapsing distance-curriculum runs. The originally
named rl-brax-decoupling branch remains at `b0eed1e` and is not the execution source.
The paper branch has an existing untracked outputs directory. Neither that directory
nor the running old calibration scheduler is modified. New machine evidence belongs
under `artifacts/verification/ppo-stability-20261015/` in this worktree.

The original curriculum runs, checkpoints and their CSVs remain read-only under
`/home/tong/tongworkspace/simulation_dev/new_drone_playground/results/navigation_stability_fixes_v01/`.

## Diagnostic questions and discriminating interventions

1. Does retaining behavior-policy GRU state after an update cause immediate loss?
   Compare normal continuation with recomputation of the just-collected sequence
   under the updated actor, starting at the identical recorded boundary memory.
   Reduced action disagreement and retained successes would support a contribution.
   Equal losses would reject this limited intervention; it does not reconstruct
   history preceding the rollout boundary.
2. Does critic drift cause the first failure interval? Freeze only critic parameters
   and optimizer state after the identical checkpoint, preserving actor updates,
   RNG, environment and curriculum. Improved retention supports a contribution but
   does not prove critic bias is the unique cause or that freezing is a final recipe.
3. Do cumulative actor updates destroy the learned solution despite the existing
   local KL gate? Scale only actor LR by 0.25 from the same optimizer state. Compare
   measured action drift and paired successes at the same interactions. Historical
   scratch low-LR experiments failed; this continuation isolates the local effect.

First use LiDAR u2640, with 240 updates (122,880 interactions) and 60-update
evaluations. This contains the recorded 29/48 to 0/48 interval, before the next
curriculum phase. Evaluate S01/D01/D02/D03, which contain all 29 original successes;
S02/S03 must be included before any promotion. The source replay must reproduce
the individual episode outcomes before accepting any intervention result. The
control must show loss before interpreting a treatment as preventing that loss.
The check uses real rollout log-probability and recurrent-memory equality, not a
mock environment. A diagnostic improvement requires retaining at least 80% of
source successes and exceeding control retention at the final diagnostic update;
it does not count as formal convergence. Failed treatments remain in the report.

Depth uses u2880 as its independent replication source. Its later 1,440-update
failure interval crosses a curriculum transition and cannot be assigned the same
cause merely because the outcome looks similar.

## Final protocol fixed before selecting a new recipe

Use the original task/sensor/algorithm cells and C1-C6: three independent training
seeds, every main Navigation scene at least 23/25, three consecutive full validation
passes, then a separate frozen benchmark. No peak across models or different
recipes may be combined. Existing Tracking/Racing evidence requires provenance,
contract and raw-outcome checks before reuse. The integrated v0.2 specification
requires random initialization for all 18 cells, so old initialized Racing runs
cannot fill this matrix. The new user instruction explicitly restores three
consecutive passes and adds continued-learning stability; it is stricter than
ADR-0017's single complete validation pass. Preserve the historical protocol
labels rather than retroactively marking those old runs as passing the new test.

For the selected recipe, continue actual learning for at least 900,000 additional
interactions after the first three consecutive validation passes. This exceeds the
860,160-interaction peak-to-final-loss interval in the LiDAR anchor. Evaluate the
fixed validation cohort every 122,880 interactions rounded up to a whole update,
and at the end. Every check must still satisfy the original per-scene threshold;
at least 95% of previously successful validation cases must remain successful.
Keep the final model, all regression cases and a complete recoverable state.
Saving/rolling back to the first model is not a substitute for this test.

Training seeds remain 0/1/2 for formal comparison. Historical development cases
are used only for diagnosis. New final benchmark cases use a separately recorded
seed base at least 3,000,000, fixed before running, and cannot supply training data
or teachers. If a failed final benchmark informs a change, record that cohort as
development information and allocate a fresh final cohort.

For each sensor, algorithms share actual forward physics, controls, actor inputs
and backbone, task reward, initialization and sampling schedule. Critic, surrogate
loss, optimizer and differentiable rollout mechanics remain algorithm-specific.
Reward changes are common task changes, not a license to lower physical success
criteria. Pretraining and teacher methods must disclose provenance and cannot be
reported as pure scratch PPO without the corresponding evidence.

The experiment unit is an independently initialized training run. Report all three
seeds separately; evaluation episodes are not additional training replicates.

## References and existing interfaces

- PPO: https://arxiv.org/abs/1707.06347
- PPO implementation details: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Controlled implementation choices: https://arxiv.org/abs/2006.05990
- SHAC: https://arxiv.org/abs/2204.07137
- RL evaluation uncertainty: https://arxiv.org/abs/2108.13264

Reuse Trainer/TrainingState, checkpoint loaders, evaluate/rollout and the original
acceptance collector. The bounded script in tmp is a diagnostic harness, not a
replacement public training lifecycle. Changes to real training behavior must
reach the existing Trainer, CLI, configuration and appropriate tests.

## Paired results and next intervention

All four LiDAR continuations completed at u2880 from the same full-state SHA256.
At u2640/2700/2760/2820/2880, successes out of 32 were control 29/0/8/0/0,
memory refresh 29/3/0/2/0, frozen critic 29/0/0/0/0, quarter actor LR
29/11/3/0/0. None retained a source success at the end. These interventions
failed the diagnostic criterion; they are not retained in the candidate recipe.
`artifacts/verification/ppo-stability-20261015/recompute_audit.py` reconstructs
individual losses, new successes, scene outcomes and raw input hashes.

The stochastic calibration D01, seed 1000003, world 4 has identical initial
state hashes. Its u2640 success at 183.644 s yields -44.5679 under gamma .995;
u4320 collision at 95.960 s yields -17.1618. This proves a realized reward-order
counterexample, not an ordering of expected policy returns or a unique cause.

The next scratch intervention changes the task objective as a coherent bundle:
bounded goal-reaching reward, undiscounted return, finite-horizon deadline and
remaining-time observation. With progress measured per 20 m, success/failure
pay +1/-1 and all time/safety/action costs sum to at most .5. A success must
therefore exceed any failure from the same start by at least 1.475. For the
recorded D01 pair the analytic difference is at least 1.78366. The existing
NPZ lacks full position/action traces for exact rescoring, so this is a bound,
not an executed replay. Shaping is not claimed to preserve the old objective.

Candidate configs are fully resolved under the verification directory. Keep
the original seed, architecture, optimizer, physics, sampling and 2,211,840
interaction budget. Remove superseded legacy reward coefficients explicitly.
Run independent Depth/LiDAR pilots through the existing CLI and treat every
automatic pilot validation/benchmark as development information. This joint
intervention cannot identify which individual objective change is responsible
for any improvement; component ablations require a positive result first.

The goal-reward Depth pilot still lost its early D03 success: u720 had 4/8,
u960 had 0/8. This rejects an early claim that reward ordering alone solved
the collapse. Both 16-world controls continue. The next matched scratch
comparison uses 64 worlds with the same horizon, optimizer settings and
interaction budget; checkpoint/evaluation update intervals are divided by
four. This changes batch size and the number of optimizer steps per interaction,
so any benefit cannot be uniquely attributed to scene diversity.

## Existing matrix evidence and Racing comparison

The existing C1-C6 collector rechecked all three Tracking families from
scratch_v02 (PPO variant 0, APG/SHAC variant 1). All nine seed runs pass;
the partial manifest reports 3 passed cells, 14 missing, 1 incomplete.
Historical Racing/APG has three passing seeds but lacked the other two
algorithms' matching evidence in that manifest. It also used different rewards:
old APG progress 0/failure 0, convergent PPO progress 1/failure 0, convergent
SHAC progress 2/failure 40. New Racing comparisons use the last common task
reward, with explicit source configs and algorithm-specific optimizer settings.

The shared-reward PPO seed 0 was stopped cleanly at u2907 after five consecutive
100-episode evaluations with no success and zero passed gates. Its complete
state remains in results/racing_common_reward/ppo_s0/checkpoints/latest.training.zip.
This is adaptive diagnostic pruning, not a statement that PPO cannot converge.
The shared-reward SHAC seed 0 passed full checks at u600/700/800, then passed
100/100 in the automatic historical benchmark. That cohort was already used
for development. The untouched u800 model separately passed a new 100-episode
benchmark at seed base 3,000,000; trajectories and configuration are under
results/frozen_racing_common_reward/shac_s0. The other two SHAC seeds and the
other algorithms remain unverified for this common recipe.

Frozen deterministic Racing replay from eight identical initial states found
the u2900 PPO policy timed out with zero gates while the u700 SHAC policy
completed all five gates. Under the actual gamma .99, realized returns were
about .51--.64 versus 2.82--2.95. Thus these trajectories do not exhibit the
Navigation reward-order inversion. Raw states/actions/rewards and hashes are
in racing-return-comparison under the verification directory. Timeout sums
omit the critic bootstrap term and are not expected stochastic policy returns.

The CLI now accepts an explicit benchmark.frozen_seed_base without changing
validation seeds. Keep this opt-in: inserting a null default into archived
resolved configs would break exact config identity on continuation. The six
targeted CLI checks passed for the final opt-in implementation (3.99 s).

The original Orbax retention removed entire model directories, including
episode CSVs. Early pilot case data has therefore been lost; aggregate console
counts must not be presented as reconstructed individual trajectories. All 41
remaining small evidence files were snapshotted with hashes before stopping
the four owned pilots, and all four exited cleanly with complete recovery
states. The retention fix removes only training arrays, preserving metadata,
reports and episode evidence. Array-bearing checkpoint eligibility and selected
actor/best-model retention rules remain unchanged. This does not change the
training objective or the comparison settings. All four exact states were resumed after the nine targeted retention checks
passed. Logs confirm advancement beyond u1638/u1614/u180/u105 respectively;
pruned model_000120 still retains both episode CSVs and reports. Resume did not
change training configurations or objectives.


At the first retained full batch64 Depth screen (u240, 491520 interactions),
D01/D02/D03 succeeded 8/7/1 out of eight each; static scenes all failed.
This 16/48 result is developmental and does not establish stability. The
matching 16-world model u960 lost its earlier D03 successes; individual rows
from that old checkpoint were already pruned, so only its logged counts remain.

The next comparison starts Depth APG and SHAC from scratch with exactly the
batch64 task, actor, sensor, control, curriculum, horizon and interaction budget.
Use their existing differentiable implementation, historical actor LR .0003,
and SHAC state-only critic LR .001; PPO-only critic and clipping switches are
disabled. Algorithm-specific critics are an explicit allowed difference.
Full configuration and decision rules are in candidate-configs/differentiable-design.json.


The complete batch64 Depth u300 screen reached 21/48, but retained only 8/16
of u240 successes, with 13 newly successful cases; D02 regressed from 7/8 to
0/8. The raw-case recomputation script is summarize_navigation_pilots.py in
this verification directory. It ignores incomplete reports, verifies CSV
counts/case identities, and records unavailable early history explicitly.
The complete u300 state was copied and byte-verified in
`depth-batch64-retention-pair/`; u240 training arrays had already been pruned,
so its evaluation rows remain usable but a continuation from u240 is unavailable.

Racing next tests the successful SHAC optimizer (h128, target alpha .95,
bootstrap .2) under progress1/additional-failure0, the task reward with prior
PPO success. Historical SHAC trials did not combine these optimizer settings
with this reward. This changes two shared reward coefficients together and
cannot separate their individual effects. Base task penalties and physical
success rules remain unchanged. candidate-configs/racing-p1f0-design.json
records the question and decision before launch.


At batch64 u360 (737280 interactions), Depth retained zero of the 21 u300
successes and had five different successful cases; LiDAR lost all eight u300
successes. This occurs before training updates enter curriculum phase 1. Both
larger-batch candidates continue, but neither has solved retention. The 16-world
controls were adaptively pruned after repeated regressions, exiting cleanly at
u2252 Depth / u2202 LiDAR with complete training states. They did not finish the
planned 2211840 budget; compare only actually shared interaction checkpoints.

A frozen Depth u300 counterfactual evaluates the original observations versus
all-zero depth/depth_mask (the current encoder's unavailable-frame input), on
S01/D01/D03 and the same fixed seeds. First replay must reproduce all original
case events; weights, other observations, physics and recurrent resets stay
fixed. Missing-input degradation alone will not prove geometric competence.
The script saves complete trajectories and case comparisons in the verification
directory. It is development evidence and cannot fill an acceptance cell.

The acceptance collector previously omitted config.observation from the shared
Actor-input comparison. It now checks this existing field; all 25 collector
checks pass, including a deliberately mismatched input contract. Rechecking the
same reuse manifest still yields 3 passed / 14 missing / 1 incomplete. This
change affects evidence validation only, not any live learning computation.


A bounded minimum-clearance objective is implemented behind the explicit existing
reward_options field. It replaces only the old time-average clearance term with
increments in worst episode proximity, using the already stored min_clearance.
Its entire-episode charge remains <=.1, so the .5 total-cost bound and 1.475
success-over-failure return gap remain valid. The approach gradient at .5m is
.2 versus 6.6667e-5 for the time-average term at dt=.1/duration300. These are
local reward derivatives, not observed policy-gradient causality. It is a task
risk-objective change, not policy-invariant shaping. Nine targeted tests pass,
including real PPO/APG/SHAC updates and exact full-state continuation.

Before launching the prepared new scratch recipes, fork two matched PPO
continuations from the preserved Depth u300 state: unchanged control and minimum
clearance treatment. Run180 actual updates, all48 fixed cases every60updates,
using the original curriculum denominator. The existing diagnostic driver now
accepts this single task-objective override and separately records original and
branch configs. No teacher or weights from another algorithm are used.

The Depth sensor counterfactual completed: the exact original case events replayed
at 20/24 successes (S01/D01/D03 = 8/7/5). Zeroing only Depth and its validity mask
left 3/24 successes, retaining 3/20 original successes. Raw trajectories, CSVs and
case identities are retained. This establishes input dependence, with the
missing-input distribution limitation above; it does not establish avoidance
competence.

The standalone frozen CLI now records the actually loaded archive path and payload
SHA-256. The collector accepts explicitly mapped external benchmark runs only
when that identity, selected model, inputs, seed partition and config checksum
match. A real replay of the existing Racing SHAC p2/f40 seed0 model on the already
consumed 3million cohort reproduced all100 case events, gate orders and completion
steps, with the matching payload hash. This is pipeline verification, not a new
independent test. Fixed checkpoint cohorts now use their configured seed semantics
without relaxing consecutive-pass or quality requirements. All51 targeted
collector/frozen-CLI checks pass; the real linked report remains 3passed,
14missing,1incomplete, with SHAC seed0 C5 verified at u600/700/800.

Racing SHAC p1/f0 seed0 completed at u1600 after three consecutive full checks:
u1400=100/100, u1500=97/100, u1600=100/100. Its automatic development benchmark
is100/100. This supports proceeding with the preregistered shared-reward
PPO/APG checks and remaining seeds; it is not a completed three-seed cell.

At Depth u360, the minimum-clearance branch has20/48 successes but retains10/21
source successes; control has8/48 and retains7/21. Both lose all source static
successes. The branches continue to the prespecified u480; the intermediate
aggregate difference does not meet the retention criterion.

The control continuation differs from the original live curve. A first-update
audit confirms exactly matching sampled rollout/environment, RNG, recurrent
memory and critic results, but repeated execution of the same compiled GPU
update produces different actor parameters. Across production and diagnostic
compilations the largest actor-parameter difference was0.00092085 in one audit.
With `XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops=true
--xla_gpu_autotune_level=0'`, repeat, production and diagnostic outputs have the
same full-state SHA256 and every compared leaf difference iszero. Raw metrics
and hashes are saved in restore-update-audit.json and restore-update-repeat-*.json.
This is a verified numerical reproducibility issue and local remedy, not evidence
that GPU nondeterminism uniquely causes policy collapse. OpenXLA documents the
separate compilation/execution mechanisms and deterministic flags at
https://openxla.org/xla/determinism .

Next, from the same Depth u300 state, continue180 updates with unchanged task,
reward, optimizer, RNG, curriculum denominator and fixed48-case checks every60
updates, changing only those process-local GPU flags. Retention below80% at
u480 rejects this numerical setting as a sufficient remedy. Passing that screen
would require scratch replication and the full formal stability protocol; it
would not retroactively invalidate observed outcomes of previous runs. This
diagnostic records current source identity and XLA flags, independently of the
source-checkpoint provenance. The existing two clearance branches finish as
launched, preserving their numerical conditions.

The process-wide deterministic continuation failed its frozen source replay
preflight before any learning: S02 seed1000001/world7 changed from SUCCESS to
COLLISION. Its owned process was terminated with no learning updates; the partial
raw evaluation and explicit failed_preflight record are retained. This numerical
change must not be counted as forgetting during learning.

Scope the deterministic options to the compiled update only, using JAX's public
`jax.jit(..., compiler_options=...)` API, and leave initialization/evaluation on
their original numerical settings. The diagnostic's --deterministic-update
switch implements exactly this scope; no production-wide setting is changed.
The same180-update/48-case/60-update/80%-retention decision remains in force,
with a new source replay required before learning. Reference:
https://github.com/jax-ml/jax/blob/main/docs/201/controlling-xla.md .

Both clearance branches completed the preregistered180 updates (368,640 additional
interactions). Control u300/360/420/480 =21/8/0/0 successes. Minimum-clearance
=21/20/22/4, retaining21/10/8/4 of the original21 successes. Its peak did not
persist and final retention4/21 rejects this objective as a sufficient remedy.
Both final full training states and complete48-case CSVs are retained. The scoped
deterministic-update branch replayed all source case events exactly before
learning; its long-run effect remains unmeasured.

The policy-cloning diagnostic was preregistered in
`candidate-configs/rehearsal-diagnostic-design.json` before collecting any data.
It uses only fresh stochastic train/procedural episodes from the frozen Depth
u300 PPO policy; no evaluation cases, other algorithm weights, historical PPO
advantages, or critic targets are reused.47 of64 episodes succeeded, yielding
188 windows of64 steps with32-step burn-in. Collection cost was20,360 active
interactions,94,400 allocated simulation slots and394.87 wall seconds. Source
policy mean recomputation max error was0.0000614822. Data and source-state hashes
are in `depth-success-rehearsal/corpus.json`; source state remained unchanged.

The diagnostic adds0.1*KL(old policy || current policy) to executed actor updates
before existing gradient clipping/Adam. This adopts the policy-cloning term idea
from CLEAR, not its full algorithm, V-trace or value cloning. Stored teacher GRU
memory plus32-step burn-in approximates changed-policy history. The sampled PPO
actor loss and gradient norm do not include this extra term; the KL probe is
logged separately. Analytic zero-loss and restoring-gradient tests passed (2
tests,4.59s). The zero-weight real GPU audit reproduced the unchanged production,
diagnostic and repeated full-state hash28a8b4379af435fc862d1374ec11417e7e53feffd0bcd6a0cf7b3e480619cd63;
initial KL was1.35e-8, post-update probe0.00417205. Positive-weight correctness
and controlled continuation are still required before claiming benefit.

Racing p1/f0 PPO seed0 completed67,108,864 interactions with status
budget_exhausted and final0/100. SHAC's single-seed success therefore does not
establish a usable three-algorithm recipe. A historical PPOu200 frozen replay
through today's formal loader was rejected because its old resolved learning
metadata lacks a sensor contract. Preserve that failed run; a separate direct
forward diagnostic checks the state-only actor and absent sensor explicitly,
using the original consumed100-case cohort. It does not relax the production
loader or count as a new acceptance result.

The first direct Racing replay omitted the historical evaluation_index=2, so
its100/100 was on seed1500000 rather than1500200. The case-identity assertion
rejected that comparison; the run and explicit invalid_case_replay note remain.
The corrected replay matched all100 source case identities/events and100/100
successes, actor parameter hash0521989a63b109a99cd66b8c223527d2ae3e86667c046e90cd40d83b7535fb56.
This confirms forward execution for that frozen policy, not identical training
numerics. Racing APG also exhausted67,108,864 interactions: u8100=97/100 then
actual finalu8192=0/100, without the required consecutive triple.

The0.1 policy-cloning real update produced identical repeated/production/
diagnostic full states and finite metrics. Its same-sample post-update KL was
0.00346153 versus zero-weight0.00417205. The nonzero intervention changes actor
updates as intended; this single-step check does not establish retention. The
preregistered180-update branch is running as `depth-rehearsal` with the same
scoped deterministic compilation as its control.

While those continuations run, `critic-calibration-design.json` fixes a separate
read-only mechanism check: eight new procedural initial states,16 stochastic
replicas each, for frozen sourceu300 and unchanged-reward finalu480. Both use
the same initial states, curriculum fraction and action random stream. The
criterion compares initial critic predictions to complete undiscounted returns
and reports every state's uncertainty. No evaluator cases, actor fitting or
critic training enter this check. It tests calibration at initial states only;
any observed bias cannot by itself establish a unique cause of collapse.

The paired frozen-policy calibration completed256 full stochastic episodes,
68,585 active interactions and329,600 allocated slots (302.51s before,
151.47s after). The u300/u480 policies succeeded107/128 and98/128 respectively.
Raw rewards/actions/events/positions are in `depth-critic-calibration/{u300,u480}.npz`.
An independent NumPy recomputation matched every full return and raw-file hash;
initial replica equality, finite rewards, terminal semantics and unchanged
learner-state hashes passed the runtime assertions.

All eight states are reported. At state5, success fell13/16 to2/16 and mean
return1.387 to-0.469 (paired delta-1.85549; descriptive bootstrap95% interval
[-2.40547,-1.14384]), while predicted value rose0.863 to1.434. State6 fell
5/16 to0/16; its return-change interval includeszero. Other states mostly
improved, so aggregate training progress cannot establish preservation of old
capabilities. At state3 both policies succeeded16/16, but predicted values
1.159/2.097 remained far below complete returns4.929/4.944.

These are fixed training-state calibration errors, not proof of a unique critic
implementation defect. The privileged critic sees26 geometry rays rather than
the whole world, and this experiment does not separate observation aliasing,
function approximation, policy drift and bootstrap bias. The GAE paper explicitly
describes the value-based variance/bias tradeoff:
https://arxiv.org/abs/1506.02438 . Future return-target interventions must retain
the same source state, reward and interaction budget, and be compared by actual
case retention rather than lower critic training loss.

The scoped deterministic-update control finished all180 updates: u360/420/480
successes2/7/1, source-success retention1/2/0 out of21. Its final1/48 therefore
rejects numerical determinism as a sufficient remedy despite exact source replay
and first-update reproducibility. The complete final training ZIP is retained.

The policy-cloning branch completed its first full post-learning check at u360:
22/48, retaining18/21 original successes and adding4 cases (three source cases
lost). Its matched scoped control at u360 was2/48 and retained1/21. This is a
controlled short-run improvement after122,880 learning interactions, with an
additional20,360 active collection interactions explicitly charged to the
diagnostic corpus. Keep the weight, data and protocol unchanged through u480;
the prespecified final80%-retention gate and later formal95% stability requirement
remain in force. Do not promote this intermediate result to convergence.

The policy-cloning diagnostic finished: u300/360/420/480 successes21/22/9/4,
source retention21/18/8/3. The final3/21 fails its80% gate. Its complete training
ZIP payload hash was verified; preserve the short positive result and subsequent
failure together. This fixed-corpus weight.1 mechanism is not retained in the
next learning candidate. All four batch64 goal-reaching pilots also completed
2,211,840 interactions and their final48-case screens were0/48. Their complete
Orbax training states remain available; failed pilots do not have separate Actor
exports. File hashes and exact terminal status are in `completed-goal-pilots.json`.

`critic_target_audit.py` replays the first512 recorded actions of every calibration
episode under the original full states. Positions and rewards matched exactly
(max error0), all terminal events matched, and independent NumPy GAE computation
agreed with the production function. The shared preparation helper reproduces
the original complete initial-state hash after its extraction from the collector.
No new policy actions, parameter fitting or evaluation cases enter this audit.

Mean absolute per-state target bias (u300/u480): h32/lambda.95=.910/.935;
h32/lambda1=.854/.728; h128/lambda.95=.890/.911; h128/lambda1=.405/.139;
h512/lambda.95=.890/.911; h512/lambda1=.048/.002. These estimates refer to these
eight training states and16 stochastic replicas, not a general guarantee that
longer returns improve optimization. Raw values and all six target arrays per
policy are retained in `depth-critic-targets/`.

The resulting preregistered paired continuation uses h512,64 worlds,64 minibatches,
four epochs and the same sourceu300 complete state. Both arms retain512 frames
per optimizer minibatch, but recurrent sequence composition/horizon differs from
the earlier h32 run. Only lambda differs between the new pair (.95 versus1).
Run12 updates (393,216 learning interactions), full48-case checks every4updates;
actual final update312 must retain>=80% source successes before promotion. Both
arms replayed the original21/48 case events. No rehearsal or calibration examples
enter their training. The long-lambda1 actual-update audit verified finite metrics
and identical repeated/production/diagnostic full-state hash
70c23a4717ceb86598a3e9e57956d9c99b784223828f8476d8b8b795cf413bee. Every subsequent
update's metrics are retained, including the actual KL-based actor skip fraction;
configured optimizer steps alone do not establish equal executed actor updates.

Racing's new APG scratch control changes only actor lr.001 to.0001 (plus output),
keeping the existing p1/f0 task and67,108,864-interaction/three-pass protocol. The
existing higher-rate run's97/100 at8100 then0/100 at8192 motivates testing a
smaller step size. It is one seed's controlled candidate, not a unique causal
explanation or a completed cell; other algorithms' evidence is not transferred.

The existing calibration initial states were reconstructed with exactly the same
full-state hash. Seven were20m starts, one80m; state5's regression is therefore
inside the training reset-distance support. Its initial clearance was0.644m and
its world had54 obstacles,35 moving. The fixed source screening starts span
95.764–96.243m, beyond the20/50/80m reset distances. This is an explicit coverage
gap, but it cannot alone explain the observed within-training-state regression;
reset support also does not describe every state visited later in a trajectory.

`critic_trajectory_audit.py` independently checks the existing raw file hashes,
returns and terminal event arrays, reports all eight states and illustrates state5
in `depth-critic-calibration/state5-regression.png`. Sourceu300 had13successes,
2bounds failures and1collision; finalu480 had2successes and14collisions, with
median collision time2.316s. All16 replicas remained active for the first2s.
Mean x displacement then increased1.041m to2.619m; the former detour disappeared
in most failed trajectories. This shows a concrete loss of avoidance behavior,
not a unique explanation of why the update selected it. All states and both
policies' trajectories remain available; the highlighted case was selected for
the largest already observed paired return decline.

Racing PPO now has one additional current-source scratch candidate. The historical
`scratch_v02/racing_ppo_s0_trial2` raw100-episode CSVs at900/1000/1100 each verify
100successes with the legal gate order. Its optimizer options motivate changing
critic lr.0005->.001, entropy.001->0, KL limit.02->0 and value clip10->0 together,
retaining the current common p1/f0 task, Actor, physics, observations and budget.
This is a bundled recipe candidate, not single-factor causal evidence; no old
weights or moments are used. Seed1's historical interruption is not acceptance.
The full diff, source hashes and decision rule were saved before launch in
`candidate-configs/racing-ppo-reference-options-design.json`.

Both h512 branches completed the first matched check after131,072 interactions:
lambda.95=24/48 with14/21 source successes retained and24collisions;
lambda1=28/48 with18/21 retained,11collisions and9timeouts. Each adds10 new
successes. The first four updates configured1,024 actor minibatches per arm;
actual execution was252 forlambda.95 and520 forlambda1, verified from the
logged skip fractions and the production update loop. The observed first-check
advantage is therefore not explained by lambda1 simply executing fewer actor
minibatches. This remains an interim comparison; the fixed393,216-interaction
final retention gate has not yet been evaluated.

The h512/lambda.95 second complete check atu308 fell to6/48 and5/21 retained
after262,144 additional interactions (all three static scenes0/8). Continue the
preregistered final check; lambda1 has not yet completed the corresponding check.

Current-source Racing PPO reference-options seed0 completed the required triple
at1400/1500/1600, each100/100, and automatic development benchmark100/100.
Raw CSV counts, legal gate orders, finite positive completion times, identical
development case identities and disjoint benchmark identities were checked.
The selected inference archive payload hash matches its metadata. Selection cost
was26,214,400 interactions; recorded training wall time502.059s. Exact records
and hashes are in `racing-ppo-reference-outcomes.json`. Seeds1/2 were launched
with configurations verified to differ only by seed and output; no initial
weights or optimizer states were supplied. Reserved4million final test remains
unused and no additional formal matrix cell is claimed.

Racing PPO seed1 selected500/600/700; seed2 selected200/300/400, all passing
checks100/100. Seed1's earlier200/300 passes were followed by0/100at400 and
did not count as a triple. Every earlier failed check remains retained. All three
fixed recipes passed the collector's individual C1/C3/C5/C6 audit, after which
the selected models were frozen on the preregistered4million cohort. Each final
test was100/100, with matching actual loaded payload hashes and disjoint training/
validation/test case identities. No test failure informed this recipe.

`audit_racing_ppo_frozen.py` reuses the acceptance collector and native RScope
test helper. All six representative replay bundles loaded through the upstream
viewer APIs; recorded times/positions matched the raw NPZ exactly, embedded
geometry/asset hashes matched, and initial/middle/final MuJoCo restorations
matched the recorded drone position. The viewer GUI was not opened. Per-seed
Wilson95% intervals are reported for episode sampling only; three training seeds
do not establish a population-wide training-seed success rate.

The explicit `acceptance-manifest.json` now selects the nine previously verified
Tracking runs and these three frozen Racing PPO runs. `acceptance-current` reports
3passed/1incomplete/14missing: all three Racing PPO seed entries pass, but its
cross-algorithm Actor comparison remains unavailable until the other Racing
algorithms supply complete selected evidence. No collector gate was relaxed.
SHAC p1/f0 seeds1/2 are now running with seed/output-only changes from seed0.

## User clarification: reuse scratch evidence and allow different rewards

The user explicitly resumed the goal and clarified that reproducible scratch convergence
need not be repeated under the current source revision. Algorithms may use different
training rewards provided the task goal remains the same. The current three-full-check
protocol and the PPO Navigation stability window remain in force.

The collector now reports per-algorithm acceptance independently of the cross-algorithm
configuration comparison, and can reuse explicitly selected prior verification reports.
It compares saved configuration, run record and frozen-run selection before reuse; it
does not repeat episode audits, policy reads or hashing. The targeted collector suite
passed38 tests. Initial pytest autoload picked up an unrelated ROS plugin with missing
lark; disabling plugin autoload ran the existing suite without installing dependencies.

The reused matrix passes7/18 cells: Tracking all3, Racing PPO/APG, and Depth APG static
and dynamic. Its18 selected runs reuse previous full verification. Depth APG uses the
historical scratch_depth_apg_s0/1/2 cohort already audited in results/acceptance/scratch.json;
no v1 runtime loader was restored. Historical SHAC/LiDAR successes remain recorded; their
single-full-check or incomplete consecutive-check evidence still needs the precise gap
filled before promotion under this protocol. No new training was launched for matrix reuse.

The redundant Racing APG h128/lr.0001 p1/f0 process received SIGINT and saved its full
recovery checkpoint with status interrupted. Its old three-seed p0/f0 evidence suffices
for Racing BPTT acceptance. Existing SHAC p1/f0 seed1/2 runs remain active because that
cohort still lacks three consecutive full checks. The h512/lambda.95 Depth continuation
finished at4/48; h512/lambda1 had28/48 at308 and was still finishing its bounded window.

## Protocol choice and completed continuation comparisons

User explicitly chose screening + one full validation + independent frozen benchmark.
Three scratch seeds remain mandatory; different algorithms may use different rewards for
identical task objectives. Historical accepted runs keep their actual selection rules.
The collector now reads existing v3 metadata/TensorBoard and one-time legacy frozen audits.
All 30 selected runs are now cached for future collection, with 14/18 accepted cells.
The five new v3 Navigation full reports contained six primary scenes, correctly omitting
extensions; the old collector's demand for S06/D06 at selection was wrong. The recorded
primary CSV evidence was reused to correct selection scope without replay or rehash;
`correct-primary-validation.py` records that one-time correction. Frozen eight-scene
reports still include the extensions. The 18 earlier valid audits were never repeated.
Different early-stopping/serialization/selection rules are disclosed; optimization recipe
comparison still includes reward, environment, sensor, actor and optimizer configuration.

Long-return same-state branches completed393216 new interactions, all from update300:
- h512 lambda.95 minibatches64: final4/48, retains1/21.
- h512 lambda1 minibatches64: final8/48, retains8/21.
- h512 lambda1 minibatches4: final21/48, retains10/21; first two checks26/48.
- Same minibatches4 with stratified reset sampling: checks10/48 (retains7),16/48
  (retains11),3/48 (retains0). Active starts actually reached96.638m, with19–35 of64
  worlds starting beyond90m. The rollout likelihood and recurrent checks both had0 error.
None meets the preregistered17/21 short diagnostic promotion threshold. Better calibrated
long-return targets and broader starting coverage are each insufficient to prevent collapse.
The final stratified model and every failed branch's full training state remain on disk.

The newly finished Racing rewardp1/f0 SHAC cohort: seed0 accepted1600, seed2 accepted1900,
seed1 budget exhausted2048 with final53/100 (u1900=100 thenu2000=0). These exploratory
runs are not substituted into the older, internally consistent accepted p2/f40 cohort.
The redundant Racing BPTT h128 run was stopped after1115 updates, preserving its full state.

Next diagnostic uses the completed h512/lambda1/minibatches4 branch as control and
multiplies only separate critic fitting by8. Reference: OpenAI Spinning Up's PPO
implementation has independent train_pi_iters/train_v_iters; this supports independent
value fitting as a standard optimization choice, not proof of a drone-specific cause.
Existing long-target rollout explained variance0.08–0.26 and the10/21 final retention
motivate the test. Preregistered393216 new interactions and final17/21 promotion gate
are unchanged. Actual Optax critic counts increment128 in the first four updates;
first-update likelihood/recurrence and fixed48 replay checks are reused from the
completed control. The final result is pending.

Reproduction disclosure: the oldest reused scratch_depth_apg_s0/1/2 records contain
source identities and dependencies but no git_commit field. They establish historical
three-seed scratch success, not a newly verified exact historical source checkout.
The newer scratch_v02/depth_apg_s0/1/2 batch also has accepted results and explicit
source commits; it has not replaced the already audited cohort or caused retraining.

The critic8 comparison completed all12 updates and393216 interactions. Checks were
19/48(retained15/21),11/48(retained8/21),14/48(retained7/21). Every actual Optax
critic count advanced128. Last pre-update explained variance0.15766 and returnRMSE1.0984
show no clear value-quality improvement over the control (0.15783 and1.0525).
Extra regression does not meet the17/21 retention gate; no candidate is promoted.
Final complete state and per-case outcomes are preserved. No training remains active.

Outstanding PPO diagnosis: low long-return value accuracy remains observed, but neither
longer lambda1 targets, increased regression, larger minibatches nor broader reset
coverage has prevented regression. A future architectural/value-information intervention
needs evidence of which missing state/history predicts return before starting another
scratch cohort. Actor uses GRU192 while the privileged critic remains a stateless MLP;
this is a potential information mismatch, not a proven cause. Existing MC diagnostics
start actor memory at zero and cannot establish effects of in-flight history aliasing.

## Frozen memory diagnostic and fixed-scene scratch candidate

The preregistered frozen-memory comparison completed without optimization. Eight
in-flight states from the same update300 recovery point were selected by episode age,
with16 paired stochastic continuations per state and common action keys. Actual GRU
memory versus zero memory changed initial raw actions (norm differences0.055–0.214),
but none of the eight full-return comparisons met the preregistered material threshold
(absolute mean difference>0.2 and descriptive paired95% interval excluding0).
Only one state's successes changed,13 versus14/16. This does not support adding a
memory-conditioned critic now; zeroed in-flight memory is an intervention, not an
assertion about all naturally occurring histories. Report and raw arrays remain in
`artifacts/verification/ppo-stability-20261015/depth-critic-memory/`.

Existing fixed-scene PPO results provide a closer task reference than the failed
procedural goal-reaching pilots. The historical `scratch_v02/depth_ppo_s0` full check
at1120 had25,25,0,24,20,23 successes in S01/S02/S03/D01/D02/D03 (117/150).
S03 failed every recorded full check. These are observed candidate results, not
formal acceptance; no old model was replayed or rehashed for this comparison.
The newly inspected S03 episode rows contain25 collisions at each of the seven checks,
with median flight times7.126–46.606s; there are no S03 timeouts in this historical run.
The compact failure summary is `fixed-scenes-historical-failure.json` beside the design.

The new seed0 recipe uses the existing Navigation8 fixed-scene path, random actor
and critic initialization,64 worlds,horizon512,lambda1,4 minibatches and2 updates
per scene. It keeps the finite-horizon goal-reaching reward and richer goal/rotation
observations from the current pilots. The8,388,608-interaction budget yields256
updates, with fixed48-case screening every16 updates. The stop rule allows the first
passing full validation or completion of the budget; ordinary patience cannot stop
the run early. Full25-case validation still requires23 successes per primary scene;
900k additional stability interactions and independent seeds/testing remain pending.

This is a bundled scratch recipe trial, not a controlled proof that training
distribution caused collapse. Fixed scene geometry matches the historical task;
validation/test initialization seeds are separate, but unseen geometry generalization
is not claimed. No critic8 fitting, memory architecture or rehearsal was retained.
Config/design were committed at17ddbd8 before launch; run.json records the same
commit, existing source digest and actual per-scene sampling. The active run is
`results/fixed_scenes_long_return/depth_ppo_s0`. Production code was not changed.

The first complete screen atu16/524288 interactions recorded30/48 successes:
S01=8,S02=8,S03=0,D01=0,D02=8,D03=6. S03 andD01 each had8 timeouts;
D03 had2 collisions. Successful scene median flight times were260–270s, close
to the unchanged300s deadline. This is a different failure pattern from the old
fixed-scene S03 collisions and is not a stability pass. The30 actual successful
case identities are stored as the retention anchor in
`artifacts/verification/ppo-stability-20261015/fixed-scenes-long-return-progress.json`.
Only complete newly written screens are summarized; prior rows and model identities
are reused without rehashing. The preregistered budget/stopping rule remains unchanged.

The second complete screen atu32/1048576 interactions dropped to0/48, retaining0/30
initial successes. Failures were46 timeouts and2 collisions (one each inD02/D03).
Thus the bundled fixed-scene recipe has already shown regression and is not promoted.
This does not isolate training distribution as a cause because reward/observations/
horizon also differ from the historical fixed-scene reference. The run continues to
its registered budget or first full pass; no successful final/stability result exists.

## First full scratch pass and unchanged continuation window

The fixed-scene Depth run subsequently screened4/48 atu48,40 atu64,23 atu80,
12 atu96,37 atu112,32 atu128 and48 atu144. Retention of the first30 successes
was0,29,13,2,27,23 and30 respectively. All completed screen rows are cached in
`fixed-scenes-long-return-progress.json`; no earlier model/CSV hashes were repeated.
Atu144/4718592 interactions the first full validation passed150/150,25/25 in
every primary scene. Its actual full-validation seed base is1500000, distinct from
the1million eight-case screens. The original full3D action interface remains used.

During this run, a concrete sampling support gap was identified: scene switches
reset after102.4s while task duration is300s and the Actor observes remaining time.
The frozen model32 clock clamp left S01/D01 at0/16, all timeouts, so that intervention
was rejected. This does not establish that episode coverage is irrelevant; it fails
to support a direct clock-input repair. Remaining-time observation and partial-episode
bootstrapping are standard distinctions described by
https://proceedings.mlr.press/v80/pardo18a.html; the paper is not a project root-cause proof.

Those recorded trajectories ended nearx98m butz3.9m, missing the goal[98,0,3]
and0.5m sphere: mean final distances0.956m/0.932m, maxima1.209m/1.081m.
A separately preregistered frozen intervention retained learned horizontal actions
and used the existing goal-altitude PD. Scores became8,8,8,7,6,3, or40/48;
all8 failures were collisions. This supports a vertical-control contribution for
this failed model, not universal stability or a unique cause. It intentionally
changes the action interface. The proposed scratch two-action config is saved but
not launched because the original3D run then achieved its first full pass.
Historical horizontal-action Fan-reward trials and velocity/yaw trials also failed;
these were read before choosing the diagnostic, not repeated.

`tools/ppo_stability_window.py` restores the complete selected learner, including
optimizer/RNG/world/recurrent state and current scene index. It uses unchanged
production Trainer.update and the same2updates-per-scene reset/switch schedule.
The registered900000 minimum rounds to28updates/917504 interactions;122880 cadence
rounds to4updates/131072. Checks areu148/152/156/160/164/168/172 on the same150
full-validation cases. Every scene must retain>=23/25 and at least143/150 original
successes must remain. The entire window completes even if an intermediate check
fails; actual final model172 is saved. Original150-case outcomes are read once and
reused; the source is not evaluated again. Four targeted gate/identity tests and Ruff
passed. The direct runtime verifies resolved learning config, geometry, source identity,
and actual interaction/update counters. No production learning implementation changed.

The window runs in `results/fixed_scenes_long_return/depth_ppo_s0_stability`, launched
fromfbae846. An independent LiDAR seed0 trial was also launched from that commit in
the same result group. Its config differs from Depth only in sensor, Actor kind and
output, verified by a structural comparison; every parameter starts randomly.
Three-seed evidence, per-seed stability and independent testing of actual final
models remain outstanding. The automatic original2million benchmark does not replace
that final test. The14/18 reused matrix is unchanged.

### 2026-10-16: first stable-window failure and LiDAR resource fix

The unchanged Depth continuation failed its gate: u148 lost6 D03 cases(19/25),
u152 lost22 S03 cases(3/25), andu156 had S03=0/25,D03=17/25. Cached case identity
and retention are in the window report; no initial checkpoint replay was repeated.
All registered28 updates still run. This is a first full pass, not stable convergence.
The predesigned altitude-PD scratch trial is therefore no longer deferred: its
existing40/48 frozen intervention motivates the one-variable control change, but
cannot supply formal training or dynamic-scene acceptance evidence.

Original LiDAR seed0 failed with XLA RESOURCE_EXHAUSTED requesting56.49GiB before
any completed update. Commit2cde4fc wraps only the LiDAR recurrent PPO scan body
in `jax.checkpoint(..., prevent_cse=False)`, recomputing PointNet activations.
The [JAX API](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html)
documents this reverse-mode memory/computation tradeoff and the scan-specific CSE
setting. No network, action, batch, objective, optimizer or dependency change.
Four CPU tests passed(35.28s), including real PointNet output/gradient equivalence
with masked NaNs/empty clouds, recurrence resets, and actual physical PPO update
plus state restoration. Ruff check/format passed. Original failed artifacts remain.
The fresh `lidar_ppo_s0_remat` run completed full-sized update1:32768 interactions,
38.418s update time, finite metrics,3895MiB device-wide sampled memory including
Depth. This proves the original batch can execute; it is neither a peak-memory
bound nor convergence. The first update skipped15/16 actor steps under its KL
guard; subsequent learning and scene-level quality remain empirical questions.

The full unchanged Depth window completed all28 updates and917504 interactions
with exit0. Scores at148/152/156/160/164/168/172 were144/128/117/148/122/146/134
out of150; only160 and168 met both gates. Actual final172 has25/25 on five scenes
and9/25 onD03, with16 collisions;134/150 original successes remain(89.33%). Training
updates took553.09s; total window wall time2038.96s includes evaluation, setup and
concurrent LiDAR/Depth candidate work. Full report was copied unchanged to
`artifacts/verification/ppo-stability-20261015/depth-fixed-scenes-stability-report.json`.
New episode tables alone were aggregated in `fixed-scenes-stability-failures.json`;
source/cached cases and hashes were reused. No final independent test is warranted
for this failed stability candidate. No matrix regeneration or historical reruns.

The LiDAR rematerialization candidate completed a full16-step actor optimizer
schedule without KL skips or OOM as well as its initial guarded update. Its first
two complete screens were0/48 then12/48 at32(1048576 interactions):S01=8,D02=4,
others0. Those12 actual success identities now form the retention baseline.
The altitude-PD scratch candidate launched from9953183; it is still within its
registered budget and has no complete first-screen result yet. Both candidates
retain their original stopping and validation rules; neither is accepted.

### Matched Actor learning-rate continuation after the first full pass

The new LiDAR screens at48/64/80 were31/0/8 out of48. Of its initial12 successful
cases, retained counts were8/0/0. The altitude-PD scratch first screen was6/48,
with36 timeouts and6 collisions; a later screen is still running.

The existing fixed3D Depth control now supplies a complete28-update comparison
from the same150/150 source. The new registered branch changes only Actor LR
0.0002->0.00002, including its trainable log_std; Critic, entropy coefficient,
scene schedule, action and reward remain unchanged. Both branches restore the
same complete original state. This is distinct from the earlier failed quarter-LR
LiDAR experiment with the old curriculum/reward; its negative outcome was read
before choosing this source-specific experiment.

`tools/ppo_stability_window.py` accepts the explicit LR change and a completed
matched-control report, reusing original case records and source/config identities.
It requires the imported training code to be at the original clean Git commit;
the detached `ppo-window-reference` worktree pinsfbae846. The runner commit ise76f44f.
Changed continuation metadata stores its actual configuration and labels the
retained original identity `source_config_sha256`; it does not assign that identity
to the changed configuration or compute a new digest. Recorded package versions
match the original source. Tests(12 passed in12.48s) cover LR-only mutation,
invalid rates, cached-source mismatch/duplicates, unchanged Adam moments with0.1x
actual optimizer delta, and existing scene/retention gates. Ruff passed.

The actual first rollout's15 reward/advantage/value/event metrics match the saved
control exactly (max difference0). First two actual Actor parameter delta norms
are0.062493 and0.070112, with no KL-skipped steps; mean KL is0.00011698/0.00019449.
This is active optimization, not a frozen policy. The same seven150-case checks
and917504 interactions will finish even after any quality failure. The run is
explicitly diagnostic; adoption would require a reproducible from-scratch schedule,
the remaining independent seeds and independent testing of actual final models.
The [CleanRL PPO source](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
uses learning-rate annealing; this10x post-first-pass reduction is a project
experiment, not that exact linear schedule or a known convergence guarantee.

The altitude-PD scratch u32 screen completed37/48:8,8,0,8,8,5; all six initial
successes remain. S03 has8 timeouts, D03 has3 collisions. This is developer-screening
improvement, not a full25-case validation or stable-window pass.

### Parallel fixed-scene sampling with continuing episodes

Existing sequential configuration changes scene every2x512x0.1=102.4s. The completed
original28-update stability window forced896 unfinished worlds to reset at14 switches,
and sampled no actual300s task timeout. Its source successful full-validation median
flight times were160.652--183.598s across primary scenes. Source minimum body clearances
were0.04163m(D02) and0.08395m(D03), indicating fragile trajectories, not a proven cause.
The matched0.1x Actor-LR continuation produced139/126/62 out of150 at148/152/156;
all failed. In the third check S03/D03 were0/25,D01=7,D02=5. Continue its registered
window; these results do not support a simple excessive-Actor-step explanation.

The new intervention uses64 worlds with8 worlds on each original Navigation8 scene
at every update, instead of2 full-batch updates per scene. Per16updates each scene
still receives65536 interactions. Independent world states continue until actual
episode termination/truncation. Parallel batching jointly changes temporal scene
distribution and removes forced early resets; this experiment cannot isolate their
individual contributions. It changes neither geometry nor physical task deadlines.
The [MuJoCo Playground wrapper](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/wrapper.py)
and [Gymnasium vector environments](https://gymnasium.farama.org/api/vector/sync_vector_env/)
provide the common semantics of independent batched worlds and per-world episode resets.
SceneBatch dispatches to existing Scene clearance/raycast without another physics
backend, training state or wrapper hierarchy. Fixed Navigation geometry only;
unsupported procedural/per-object reward combinations fail at construction.

Paired fresh seed0 config changes only learning.scene_sampling=parallel and output
relative to the original full3D Depth long-return config. It retains256 updates,
8388608 interactions and16-update screening; no teacher, replay training data or weights.
The registered option2 quality, three-seed, retention and independent-final-test
requirements remain unchanged. Training success alone cannot replace those checks.

Implementation spans scene dispatch, environment construction, CLI sampling/resume,
actual per-world episode/replay identities and the stability continuation runner.
26 distinct targeted tests passed across three runs after fixing new test fixtures:
float32 JIT/eager ray tolerance, gamma=1 for goal-reaching, static sampling provenance,
and typed PRNG key comparison. Last run4/4 passed in29.13s. Real physical PPO updates
continue past the old two-update boundary, restore exactly, and reach actual finite
task timeout; masked reset preserves the live world. Existing sequential resume,
screening selection and stability/LR gates pass. Config comparison confirms only
the two registered changes. No BPTT/SHAC rerun, historical payload/hash audit, matrix
regeneration or packaging regression. GPU feasibility and convergence remain unproven.

New completed screens only were ingested: LiDAR96/112/128=0,144=39/48 with all12 initial
successes retained; Depth altitude-PD48=16/48 with0/6 initial successes retained.
These results retain their original run identities and budgets; neither is accepted.

Parallel scratch launched from811eabe(PID2452923), and completed its first full64x512
GPU update:32768 interactions, finite metrics, Actor gradient norm0.76566, KL-skipped
fraction0, update time86.10s including compilation. Device-wide sampled6948MiB includes
the three other live jobs and is not a peak bound. Run metadata records8worlds for
each of8scenes. Evidence is `fixed-scenes-parallel-first-update.json`; copied identities
come from the run recorder, with no extra digest calculation. This establishes execution,
not learning stability; the seed0 run continues under its original registered budget.

Subsequent newly completed screens: LiDAR160=8/48(D01only), retention0/12;
altitude-PD64=24/48(S01/S02/D01), retention6/6. S03 has8collisions, D02/D03 each8timeouts.
No new full validation, independent benchmark or matrix credit from these screens.
