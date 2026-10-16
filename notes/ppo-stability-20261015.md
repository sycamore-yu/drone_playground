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

### Full-episode coverage observed and reporting uncertainty completed

Actual parallel training first reaches the300s finite deadline at update6, with2
timeouts; update7 adds2more. The original sequential continuation's0timeouts and896
forced live-world resets are reused from registered evidence. At229376 interactions,
this proves complete episode coverage is now possible; it does not yet establish
better validation success. New episode-coverage metrics are preserved in
`fixed-scenes-parallel-episode-coverage.json` without restoring models or replaying cases.
LiDAR176 again scored0/48. The LR continuation's fourth check160=136/150 also failed.

The objective also requires uncertainty for all chosen seeds. The existing cached
report had counts but no intervals. The collector now derives per-policy/per-scene
95% Wilson intervals from verified counts, plus equally weighted training-seed mean,
range and sample standard deviation. Episodes from different trained policies are
not pooled. Missing counts remain unavailable; no acceptance threshold changes.
The standard-library formula was checked against the
[SciPy binomial interval API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html),
including0/25,23/25,25/25,100/100,1/1,7/50. Sixteen targeted tests pass in3.25s;
41unrelated collector cases were not rerun. The cached matrix was refreshed with
raw-audit and digest entrypoints replaced by raising assertions. All30 runs reused;
all prior benchmark summaries and14/18 statuses match exactly. Evidence is
`acceptance-uncertainty-check.json`. For scale,25/25 has Wilson interval86.68--100%,
while23/25 gives75.03--97.78%; these are conditional test-sample intervals, not a
claim about arbitrary new training seeds or unseen geometry. Three-seed spread is
descriptive and does not alter the agreed reproducibility protocol.

### Completed LR continuation and first parallel screening

The matched0.1x Actor-LR branch completed all28updates/917504 interactions and exited0.
All seven checks failed:139/126/62/136/144/130/139 out of150. Actual final172 gives25/25
on five scenes and14/25 onD03, retaining139/150(92.67%). The unchanged control passed
2/7 checks but also failed overall; it ended134/150. Mean paired success-count change
is-9 per check, and the reduced-LR minimum62 is worse than control117. Thus the5-case
final improvement does not establish stability or an overall controlled benefit.
Every update has a positive finite Actor parameter delta norm(range0.04675--0.10151).
Raw report copied unchanged to `depth-postpass-lr01-report.json`; matched summary in
`depth-postpass-lr-comparison.json`. Source cases and control data reused, no replay
or additional hash. Update wall1144.99s,total4856.39s include different concurrent GPU
workloads; they are not an isolated LR-cost comparison. No final independent test or
new lower-LR sweep from this negative branch alone.

Parallel fresh Depth first complete screen at16/524288 interactions scored6/48:
S01/S02/D02 all8timeouts,S03/D03 all8collisions,D01=6successes+2timeouts. Original
sequential fresh seed0 at equal interactions had30/48, with no shared successful
cases. Exact development-screening case identities matched using cached records;
new comparison is `fixed-scenes-parallel-comparison.json`. Full-episode coverage is
verified but a learning-quality benefit is not established at this first check.
Continue the original budget rather than turning this early observation into a
claim about eventual convergence or a unique cause.

New altitude-PD screens80/96 are39/40 out of48, retaining all6 initial successes;
S03 remains0/8 (timeouts). LiDAR192/208=0,224=19,240=38; last screen retains10/12,
withS03=0/8 collisions andD02=6/8. No full-pass selection from these candidates yet.

Final acceptance integration remains required: the generic collector currently links
independent benchmark input to the first selected checkpoint, not a later completed
PPO stability window. Before adding any PPO Navigation cell, connect and validate the
original first-pass selection, all registered continuation checks, actual final model
and independent final frozen evaluation as one chain. Existing14 cells are unchanged;
do not substitute the initial automatic benchmark for the actual final model's test.

## Actual final-model acceptance linkage and completed LiDAR budget

The initial benchmark of a selected PPO policy does not establish post-window quality.
The existing collector now joins the already checked initial selection to a complete
unchanged PPO window and a separate frozen test of its actual last model. The active
manifest enables this additional Navigation requirement with an empty mapping until
a candidate exists. No historical BPTT/SHAC run or accepted Racing run was rerun.

The window check uses recorded case identities, source/config identity, 900000 minimum
extra interactions rounded to updates, rounded 122880 check cadence, actual update logs,
all six scene thresholds and 95% original-case retention. It rejects changed recipes,
missing checks, rollback and diagnostic LR continuations as original-recipe acceptance.
Final frozen execution records the actual loaded Orbax metadata and requires an unused
seed base at least 3000000 with matching task/scene/actor/physics inputs. The original
selection benchmark and stage costs remain visible separately. Historical reports may
omit this task-specific manifest section; the active task manifest explicitly includes it.

31 distinct targeted checks passed, including real v2/v3 load recording, negative chain
fixtures and reuse that prohibits repeated raw audits. The 30-run cached matrix remained
14/18 with collect_run/digest replaced by raising assertions. Evidence is
`artifacts/verification/ppo-stability-20261015/final-stability-chain-check.json`.
This is report correctness evidence, not a successful real PPO stability/final-test run.
No parameter-byte hashes or replay were added. A consumed final test must be recorded
explicitly; reporting code cannot infer whether a researcher consulted a result.

LiDAR rematerialized seed0 ended normally at u256 / 8388608 interactions with
`budget_exhausted`. Its u240 38/48 fell to u256 0/48: all 48 cases reached the actual
300-second task timeout, with zero of the initial 12 successes retained. No final
independent test is warranted. Existing prior screening records were reused, and only
the newly completed final table was ingested.

Depth altitude-PD u112 is again 40/48: five scenes8/8, S03 eight timeouts, initial
six successes retained. Parallel full3D Depth u32 is39/48: four scenes8/8, D03 7/8
(one collision), S03 0/8 (eight timeouts); all six initial successes retained. The
matched sequential u32 was0/48. The paired comparison reuses the cached case records;
this local improvement does not establish stable convergence. Both current Depth
candidates continue their predeclared budgets. No new hyperparameter sweep launched.

Parallel u48 completed during reporting:38/48, S03 eight timeouts, D03 two collisions, other four scenes8/8; initial six successes retained. Matched sequential u48 was4/48. Cache/comparison and status updated from the newly completed screening table.

## S03 trajectory diagnosis and nominal-reset comparison

The new u112 altitude-PD screen40/48 and u48 parallel screen38/48 both had eight
S03 timeouts, but existing CSVs contained no terminal positions. Two frozen
diagnostic replays retained the exact eight development cases, task, sensor and
Actor, adding only existing progress/trajectory recording. The first CLI attempts
failed before simulation because unselected models lack separate Actor exports.
The preserved full states were then loaded through load_model_state; shape/dtype,
resolved configuration and update counters passed. No original model was altered
or manually rehashed. Both completed diagnostics reproduce all original case events.

PD u112 ends at approximately(13.51,-0.53,3.00),84.49m from the goal. Its final30s
net displacement is0.002--0.012m and mean speed0.0011--0.0044m/s. The nearest
terminal cylinder clearance is1.756--1.764m. Parallel u48 has two near-stationary
cases around(40.86,-11.40,4.93),58.30m from goal; six cases finish around
(76.95,-9.80,4.25),23.25m from goal, with4.69--5.28m traveled but only1.93--2.18m
net displacement in the last30s. These establish interior stalling/oscillation, not
near-goal settling, but not its unique cause. The goal remains0.5m and deadline300s.
Measured rows, source provenance, raw trajectories and standalone figure are under
`artifacts/verification/ppo-stability-20261015/s03-timeout-diagnostic/`.

Meanwhile PD u128 fell40->0/48, losing all six initial successful cases: S03 eight
collisions, D03 one collision/seven timeouts, other four scenes eight timeouts each.
Parallel u64 is32/48, retaining its initial six successes: S03 five collisions/one
out-of-bounds/two timeouts; D03 only1/8; D02 7/8; S01/S02/D01 each8/8. Thus neither
altitude control nor currently observed parallel mixing establishes stable convergence.

The next registered48-update continuation uses the intact parallel u48 state. Only
future episode resets switch from whole-map random starts to the existing nominal
start jitter. Current physical episodes, RNG and optimizer are retained. This tests
coverage of the actual task's full routes; the mixed failure locations do not establish
an exclusively early-route coverage problem. The live unchanged run supplies matched
u64/u80/u96 checks. A final pilot pass needs S03>=6/8, retention>=37/38 and other
main scenes>=7/8; temporary peaks do not count. No final-test cases or training from
evaluation data are introduced. Details and falsifier are in `nominal-starts/design.json`.
Start-state distribution is a standard curriculum variable; reference: Florensa et al.,
https://proceedings.mlr.press/v78/florensa17a.html . This comparison is not their
adaptive reverse-curriculum algorithm. A beneficial continuation would still need
a from-scratch recipe and the unchanged formal stability/independent-test protocol.

Nominal-reset pilot launched from commit17b84c6 (PID2530705), preserving all source/control outputs. Source/control case comparison now shows u64 retains32/38 and u80 retains30/38, with no new successful cases. Later ordinary screens: PD u144 remains0/48(all timeouts); parallel u80=30/48(S01/S02/S03/D01/D02/D03=8/2/0/8/8/4). No matrix promotion.

The nominal-reset branch completed its first real update u49 (32768 interactions): finite metrics, Actor gradient norm0.52205765, KL skip0, update time73.977s including compilation. Complete-state/configuration/geometry assertions passed. This validates execution only; first screening at u64 remains pending. Diagnostic wall time begins after restoration and GPU costs are concurrent, so setup/isolated throughput are not implied. First-update evidence records approximate process start time separately.

### First paired reset result and offline geometric falsification

The nominal-reset branch u64 completed524288 additional interactions and scored40/48:
S01/S02/D01/D02/D03 each8/8, S03 eight timeouts. It retains all38 source-u48 successes
and adds the two previously failed D03 cases. The matched unchanged control scores32/48,
retaining32/38; the treatment has eight exclusive successes and loses no control-success
case. This is an intermediate retention benefit, not S03 convergence or a final pilot
pass. The registered u80/u96 checks remain required; the actual u96 gate is unchanged.
`nominal-starts/comparison.json` compares already verified case-event records, with
assertions for exact case identities, matched interactions and paired count arithmetic.
It does not reread old CSVs, replay policies or recompute hashes.

The parallel scratch control u96 is31/48:8/8/0/8/7/0, retaining31/38 source-u48
successes. S03 has seven timeouts/one collision, D02 one collision and D03 eight
collisions. The altitude-PD scratch u160 remains0/48, all timeouts. Both continue their
registered budget; neither establishes stable convergence. Cached comparison with the
original sequential u96 is31 versus12 successes, which does not imply eventual
stability for the parallel candidate.

An offline route probe asks whether the observed S03 stalls require an initial increase
in straight-line goal distance to escape. Using the scene XML and the16 recorded
terminal positions, plus the nominal start, it constructs a0.1m eight-neighbour planar
grid with body radius0.07m and0.15m extra obstacle margin. Diagonal corner cutting is
excluded. SciPy sparse Dijkstra supplies routes; assertions verify predecessor cost
descent, summed path length and exact segment-to-cylinder body clearance. All17 routes
have zero negative direct-progress edges across the entire route, with minimum segment
body clearance0.16983m. Nominal route length is99.8936m versus96m direct distance.
This does not support necessary distance-increasing detours as the specific explanation,
so no geodesic-reward candidate is added. It does not establish dynamically feasible
flight, Actor discoverability, or zero benefit from a different shaping signal. Routes
hold observed altitude and still require vertical settling at the goal.

Evidence and reproducible offline entrypoint are `s03-timeout-diagnostic/geodesic.json`
and `geodesic.py`; no policy training or simulation was performed for this check.
The reference mechanism is distance-progress reward, used with a geodesic distance in
Habitat navigation:
https://aihabitat.org/docs/habitat-lab/habitat.tasks.nav.nav.DistanceToGoalReward.html
and https://aihabitat.org/docs/habitat-lab/habitat.config.default_structured_configs.html .
Our grid is an approximate geometric diagnostic, not Habitat's navigation implementation.

### Reset benefit did not persist; per-scene training evidence

Nominal resets u80 scored26/48 (8/2/0/8/8/0), retaining26/38 source successes versus
the unchanged control30/48 and30/38. Thus the first-check improvement did not persist.
The final u96 remains required. The original parallel scratch u112 fell to8/48, only
D03 successful, retaining6/38 source-u48 cases; the other five scenes all timed out.
Its actual training u97--112 recorded565 successes among810 completed episodes,
69.8%, so that aggregate cannot stand in for full-route evaluation. Altitude-PD u176
scored18/48 (0/8/0/8/0/2), with0/6 first-screen successes retained.

`nominal-starts/rollout_by_scene.py` (8dcdf4e) restored both intact u64 states and
collected one native512x64 frozen stochastic rollout each, with no optimizer update.
It adds missing per-scene observability for the concrete S03 failures; it does not
repeat an acceptance evaluation or model-hash audit. Both probes completed, with
finite arrays and exact agreement between done masks and the sum of event masks.
Aggregate event counts match the original u65 scalar records: random resets20success,
20collision,1out-of-bounds; nominal resets13success,5collision,1out-of-bounds. This
agreement is not a claim of exact trajectory equality under separate compilation.

S03 contributed six collisions and no successes with random resets; nominal resets
contributed one collision and no successes. Other scenes did complete successfully.
These small frozen samples expose a training gap in this segment, not a long-run
S03 success estimate. Source S03 x positions span28.39--97.13m versus5.96--88.85m;
the different in-flight states are a consequence of prior reset-distribution changes.

With gamma=lambda=1, every sample preceding the last actual termination in its world
uses an observed terminal return; the remaining tail uses the last Critic estimate.
The S03 bootstrap fractions are76.78% (random) and94.29% (nominal). Target RMSE is
0.8631/0.4384, but these are errors against bootstrapped targets, not full Monte Carlo
returns. Across all scenes230/41 of32768 predictions exceed the loose success-return
upper bound1+remaining_distance/20; this establishes some impossible value estimates,
not a unique explanation of the regressions. Raw arrays, world positions/clocks,
per-scene counts and target coverage are in `nominal-starts/training-rollout/`.

The next controlled change uses the same original parallel u48 full state and changes
only horizon512->2048, retaining random resets.204.8s covers typical160--184s successful
routes better than51.2s. Previous procedural h512 and critic-fit8 negative results are
explicitly retained; no old experiment is rerun. The1,572,864-interaction diagnostic
contains12updates and checks at u52/u56/u60, matched by interactions to the cached
controlu64/u80/u96. Final gate remains S03>=6/8, retention>=37/38, other primary scenes
>=7/8. All three checks must finish. A missing final gate rejects this change as a
sufficient repair at this budget, without an automatic horizon sweep.

Larger rollout also means four times the batch and one quarter the update count;
improvement would not uniquely isolate bootstrap error. Existing runner defaults retain
the nominal-reset behavior; argument/budget setup checks confirm identical three
interaction milestones and reject unchanged-control or combined-intervention requests.
No native learner/network code changed. Registration: `long-rollout/design.json`.
Reference: OpenAI Spinning Up's PPOBuffer distinguishes true terminal returns from
value bootstrap at a sampling cutoff:
https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html .
It supplies estimator semantics, not evidence that2048 will solve this task.

The nominal-reset branch then completed all48updates and1,572,864 additional
interactions, exiting0. Final u96=38/48 (8/8/0/8/8/6), retains37/38 source successes,
loses one old case and gains one new case. Control u96=31/48, retains31/38. The
three-check comparison is40/26/38 versus32/30/31, but S03 never succeeds and final
D03 remains6/8, so the registered final pilot gate fails. Complete state is preserved
at `depth_ppo_s0_nominal_starts/model_000096`. Wall time after restoration2626.206s;
shared-GPU setup/throughput is not an isolated cost comparison. No final frozen test
or additional scratch seeds are launched for this failed intervention.

After that process completed, the registered horizon2048 continuation launched from
dbc62df (PID2580615), using the same u48 full state and original random resets.
The default runner behavior was preserved while budgeting checks by actual interactions.
Its new12-update run is `results/fixed_scenes_parallel/depth_ppo_s0_h2048`; no first-update
or convergence result is claimed at launch. The two existing scratch candidates continue:
PD u192=25/48 (8/8/0/8/1/0), initial6/6 successes regained; parallel u128=40/48,
all primary scenes8/8 except S03 eight timeouts. Their earlier collapses remain recorded.

Horizon2048 first actual update u49 completed131072 interactions, Actor gradient norm
0.37937066 and no KL skip.104 training episodes ended,70 successfully; this aggregate
is execution evidence only. The165.587s update includes compilation. A device-wide
sample afterward was6890MiB including the other two scratch jobs, not a peak measure.
Complete-state restoration and resolved configuration checks passed: only the two
saved horizon fields and output path differ from the original source. No rematerialization
or other native code change was needed. Evidence: `long-rollout/first-update.json`.

Parallel scratch u144 subsequently completed7/48: S03 six successes/two collisions,
D03 one success/seven timeouts, and S01/S02/D01/D02 all timeouts. All six initial
successes were lost while six previously unsuccessful S03 cases became successful.
This adds evidence of cross-scene regression; it does not establish gradient conflict
as the unique mechanism. The matched original sequential u144 was48/48. No new
matrix cell is promoted and the registered scratch budget continues.

### First long-rollout screening and independent LiDAR parallel run

The horizon2048 branch's first registered check completed at modelu52, after524288
additional interactions. It scores40/48: five primary scenes8/8, S03 eight timeouts.
All38 source-u48 successes remain, with the two previously failed D03 cases newly
successful. Matched original controlu64 is32/48 and retains32/38. These are compared
by interactions, not by update count, in `long-rollout/comparison.json`. This reproduces
the nominal-reset branch's first-check success set but does not imply identical policies
or sustained benefit; the later two checks and actual-final pilot gate remain required.

LiDAR's completed sequential rematerialized run had a concrete final0/48 failure,
and no LiDAR training was active. The same parallel-sampling change is now independently
tested for LiDAR rather than inferring its outcome from Depth. The new candidate source
YAML exactly matched the completed control's resolved config before changing only
scene_sampling and output. Original reward,64x512 batch,1024points, GRU192, optimization,
random-start distribution and random seed0 are preserved. No checkpoint/teacher is used.
Existing sensor/SceneBatch and rematerialization checks are reused, with no native-code
change or repeated acceptance/model audit. Registration and config committedfc55cc8;
PID2596335 writes `results/fixed_scenes_parallel/lidar_ppo_s0`.

The first actual LiDAR update completed32768 interactions in57.978s including compilation.
Actor gradient norm22.7521 triggered the KL guard after one of16 actor minibatches;
the following15 updates(u2--16) executed all16 actor minibatches each. Thus execution
was verified without claiming the first optimizer schedule had no skipped steps.
Device-wide memory was9190MiB including three other training runs, not a peak or
isolated cost comparison. Actual config differs only in the two registered fields,
and existing per-scene geometry identifiers match the completed control. Evidence:
`lidar-fixed-scenes-parallel-first-update.json`.

Its first screening u16 is0/48, all collisions; median flight times for the six primary
scenes are167.772/173.09/84.260/52.381/118.509/19.182s. The completed sequential
control also had0/48 at these524288 interactions. This is early development evidence,
not proof of improvement or convergence. Each new case table is summarized once and
comparison reuses the control cache. The full8388608-interaction budget, option2 selection,
post-pass stability, three scratch seeds and independent actual-final test still apply.

Other newly completed screens: altitude-PD u208=24/48 andu224=33/48, retaining6/6 first
successes; parallel Depth u160=8/48, exclusively S03 success, while the other five
primary scenes all time out and its first six successes are lost. The continued task
is joint success and retention, not combining these different checkpoints' best scenes.

### Subsequent LiDAR screens and matched long-rollout training outcomes

LiDAR parallel u32/u48/u64 completed0/10/26 successes out of48, versus the cached
sequential control12/31/0 at the same interactions. Its first nonempty successful
set is the ten cases at u48 (D02 eight and D03 two). At u64, only five of those ten
remain successful, while21 previously unsuccessful cases improve. The scene counts
are8/8/0/0/3/7 in S01/S02/S03/D01/D02/D03 order; all failures are collisions.
At u80 it falls to14/48 (0/6/0/8/0/0), with all ten initial successes now lost;
S03 has eight collisions and the other26 failures are timeouts. The corresponding
cached sequential control was8/48. Thus a higher total than control is not evidence
of preserving learned behavior. These are new scheduled screens, not repeated
acceptance. The registered scratch budget continues.

The first two complete horizon2048 training blocks are now available from the original
scalar metrics, aligned to524288 interactions each. At treatmentu49--52 andu53--56,
actual completed episodes are445 and467, with311 and356 successes (69.9% and76.2%).
Cached controlu49--64 andu65--80 are529 and661, with357 and394 successes (67.5% and59.6%).
Treatment collisions110/98 and timeouts11/6 compare with control146/199 and2/5.
The block-mean Critic RMSE against bootstrapped targets is0.9575/0.8730 versus
0.7289/0.8277; these different evolving policies and targets do not measure true
return calibration. Counts span all eight scenes and include live episodes carried
from the fork, so neither per-scene competence nor terminal-target coverage follows
from these aggregates. The second development screen is still running at this update.

`long-rollout/training-summary.json` preserves the new scalar records and reuses the
completed control summary. Assertions check update/interactions alignment, outcome
counts summing to completed episodes, and agreement with logged successes. No model
loading, replay, acceptance rerun, fresh hash audit or native-code change was needed.

### Parallel Depth reaches complete screening; LiDAR still exchanges scene skills

Depth parallel u176 completed48/48 after its u160 result of8/48. All six initial
successful cases return. Median successful flight times are98.989/102.117/116.973/
93.217/100.930/95.987seconds in S01/S02/S03/D01/D02/D03 order. The native option2
selector is now executing the first25-case-per-scene full validation; screening alone
does not meet selection or stability requirements. No extra acceptance was launched.

A detached `ppo-parallel-window-reference` checkout at the original811eabe runtime
has been prepared for the already registered unchanged continuation. The existing
`tools/ppo_stability_window.py` supports parallel sampling and full-state restoration,
so no native code or new runner is required. Launch only after a complete passing
full-validation report and the native first-pass selection record exist. The window
remains28updates,917504interactions, seven checks, all six scenes>=23/25 and>=95%
case retention at each check. The runtime pin avoids the later CLI provenance-only
change being mistaken for the original training source; no old model replay/hash
audit was added. The window has not yet started.

LiDAR parallel u96 reached42/48 (8/8/2/8/8/8), recovering all ten initial cases.
At u112 it fell to15/48 (0/8/7/0/0/0), losing all ten again:16collisions and17timeouts.
S03 improves while other scene performance regresses, as previously observed for
Depth. This does not isolate gradient interference or another unique cause. Both
matched sequential controls were0/48, but that relative improvement does not satisfy
the absolute joint-success and retention gates. Its original scratch budget continues.

### Parallel Depth first full pass and unchanged continuation

The native option2 full validation at parallel Depth u176 completed150/150, all six
primary scenes25/25, after5767168 scratch interactions. Its first-pass selection
record names `checkpoints/model_000176`, evaluation index11 and seedbase1500000.
The existing continuation runner has now launched (PID2646288, session62169) from
that complete state with unchanged learning settings. It imports the pinned811eabe
runtime from `ppo-parallel-window-reference`; the runner source is2147949.
The normal runtime-source and resolved-config checks passed after full-state restore.

The already registered stability window is28updates/917504 interactions, checks at
u180/184/188/192/196/200/204, >=23/25 in every primary scene and>=143/150 original
successful cases at every check. No early stop, rollback or peak-model substitution.
Output is `results/fixed_scenes_parallel/depth_ppo_s0_stability`; command and launch
details are added to `candidate-configs/fixed-scenes-parallel-design.json`. The source
CLI's automatic seedbase2000000 benchmark is development information; independent
post-window final testing remains reserved at3000000 and must use the actual last
model. The matrix is still14/18 and no additional scratch seed has been launched.

The checkpoint writer saves training arrays once per update and atomically replaces
metadata. After the completed full evaluation and selected checkpoint were recorded,
the existing immutable state could be restored while the source CLI continues its
automatic initial benchmark. No source replay or extra manual hash audit was needed.
Before launch, device-wide GPU use was9414/24564MiB across the four existing jobs;
this is a headroom sample, not a peak or isolated cost measurement. The first actual
continuation update u177 then completed32768 added interactions (total5799936),
Actor gradient norm0.811892 and no KL skip. Update time145.405s includes initial
compilation;46of50 completed training episodes succeeded. These are execution
metrics, not formal retention evidence; the first registered full check is u180.

The long-rollout branch's second matched screen u56, after1048576 added interactions,
again scores40/48 and retains38/38 original cases. Cached controlu80 scores30/48 and
retains30/38. Treatment S03 still has eight timeouts while the other five scenes are
8/8. Thus two checks support a retention benefit for the learned source cases, but
the final third check and S03 acquisition gate remain unmet.

LiDAR parallel u128/u144/u160 scores44/24/28 out of48, retaining7/0/4 of its ten
initial successes. Sequential controls at these same interactions are0/39/8.
At u160 the scene counts are8/8/0/8/4/0, with20collision failures. At u176 it recovers
to40/48 (8/8/0/8/8/8), all ten initial cases retained, but S03 still has eight collisions;
the sequential controlu176 was0/48. Results still fluctuate and do not establish
joint-task convergence or stability.

### First parallel stability check passes; remaining window continues

The unchanged parallel Depth continuation completed its first registered check at
u180, after131072 actual additional interactions. All six primary scenes are25/25,
and all150 first-full-validation successes are retained. This passes the first check,
not the seven-check window. Native evaluation already checked case identities and
outcomes; `depth-fixed-scenes-parallel-stability-report.json` copies that validated
runtime record without reading the CSV again or auditing model hashes. Future copies
are triggered by newly completed checks or terminal window status.

The original source training process exited after its automatic initial benchmark:
the six primary scenes total150/150, S06=24/25 andD06=25/25, seedbase2000000.
This remains development evaluation of selectedu176, not final testing of the eventual
post-window model. Source run wall time13345.644s includes that evaluation and overlaps
the beginning of the continuation process; stage durations must not be summed as
elapsed calendar time. The actual final benchmark at a reserved independent cohort
still depends on passing the complete window. No extra seeds or new recipes were started.

The horizon2048 diagnostic completed all12 training updates throughu60, totaling
1572864 added interactions; the actual-final screening is still running. Its last
524288-interaction training block has373/499 successful completed episodes versus
cached control444/719. The higher success fraction does not imply more successful
completions per interaction or known terminal-target coverage. Mean bootstrapped-target
RMSE is0.9271 versus0.8387, still not a Monte Carlo return-calibration comparison.
The updated `long-rollout/training-summary.json` preserves all12 scalar records and
uses the existing validated control summary; no new frozen probe was run.

New scratch screens remain mixed: altitude-PD u240=31/48 (8/8/0/7/8/0), keeping its
initial6/6 successes but timing out on all S03/D03 cases. LiDAR parallelu192/u208/u224/
u240 scores36/40/25/38 out of48 and retains7/10/8/8 of its initial ten cases. Its
u240 scene counts are8/8/5/8/8/1 with ten collision failures. The sequential control
at u240 was also38/48; neither a peak nor this equal total establishes convergence.
Both original scratch budgets continue to their prescribed endpoint.

### Completed parallel stability, remaining scratch candidates and long rollout

All four jobs reached their registered endpoints; no early termination or rollback.
The parallel Depth u176 control finished28 updates/917504 added interactions at u204.
Seven full scores are150,124,150,150,149,149,142 out of150. At u184 S01=11/25 and
D03=13/25; final u204 has D02=24, D03=18 and the other four25. Final retention142/150
fails95%, and D03 also fails23/25. All28 measured updates are finite, Actor gradients
are positive, with no KL skips. Training-update time1026.380s; whole continuation
2583.994s including setup, evaluations and saves. The source's initial development
benchmark is a separate overlapping stage; these wall times are not summed as
calendar elapsed. The complete native report was copied once; no old episode tables,
models or hashes were re-audited. No independent final test follows this failed window.

LiDAR parallel seed0 exhausted8388608 interactions at u256=38/48, scene counts
8/8/0/6/8/8, preserving10/10 of its first nonempty screen. S03 has eight300s timeouts,
D01 two collisions. The original sequential control ends0/48. Better final performance
is insufficient for convergence, and the u128 peak44/48 was not selected. Recorded
training wall time5133.241s includes the native training/evaluation run on a shared GPU.
Altitude-PD seed0 also exhausted8388608 at u256=32/48:8/8/0/8/8/0. Its S03 now has
eight collisions, D03 eight300s timeouts; six initial successes remain. Recorded wall
time18146.718s. These candidates did not reach full validation and receive no final test.

The h2048 continuation completed1572864 added interactions. Its three matched scores
are40,40,32/48 and source-case retention38,38,32/38. Final scenes8/8/0/7/8/1 and events
32SUCCESS,1COLLISION,15TIMEOUT fail the preregistered sufficient-repair gate. The earlier
retention benefit does not persist. Actual final model is model_000060; measured
post-restoration wall time6132.805s includes all checks. No further horizon extension.

### Registered entropy-bonus removal from the same parallel first-pass state

This is a single-setting matched diagnosis, not a new training framework. Native
PPO uses a reparameterized tanh-transformed action entropy, so its bonus can affect
both policy means and log_std. The control's exploration_std rises0.095068 at u177
to0.097858 at u204 while recurrent route performance fluctuates. This correlation
and the scalar loss values do not show gradient dominance or causality. Earlier
Racing entropy0 was bundled with other settings, so it cannot answer this question.

[SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) defaults
to ent_coef0. [Andrychowicz et al. section3.8](https://arxiv.org/pdf/2006.05990) reports
limited regularizer benefits in its five continuous-control tasks except HalfCheetah.
Neither source establishes a drone remedy; both support treating the bonus as an
optional standard regularizer and testing its removal without changing the algorithm.

From the exact parallel u176 full state, set entropy_weight0.005->0 only in both saved
config representations. Preserve all weights, Adam moments, RNG, live worlds/clocks,
GRU state, trainable variance and stochastic action sampling. Use pinned native811eabe
and reuse the completed original control and validated baseline cases. The existing
window tool gains one boolean option, with no native-learning code change. It rejects
combining the change with Actor LR reduction, no-op/inconsistent source coefficients,
and using a diagnostic as the unchanged reference. Its diagnostic flag also prevents
formal acceptance even with unchanged LR. The applicable29 tests pass; no unrelated
packaging, source hashes or historical training are rerun.

Prelaunch design is `candidate-configs/depth-parallel-postpass-entropy0-design.json`.
The complete matched budget remains28 updates/917504 interactions and seven full
checks. Every check must satisfy23/25 in each scene and143/150 retention; any failure
rejects removal alone at this state/budget. Preserve the actual last model and report
all checks, with no coefficient sweep or later-window substitution. A positive result
would still need a declared scratch recipe, three seeds and final independent tests.

Entropy-removal continuation launched from runner cd5a0b8 with pinned native811eabe,
PID2693576. Full restoration/setup115.830s; first actual update177 took59.243s including
compilation, added32768 interactions. All15 registered pre-optimization rollout/target
statistics exactly match the original control; Actor parameter delta norm0.493405,
gradient norm0.810575, no KL skip. This confirms matched execution, not stable quality.
Evidence: `depth-parallel-postpass-entropy0-first-update.json`.


### Entropy removal completes and fails; frozen component diagnosis registered

The entropy0 branch completed all28 updates/917504 added interactions. Checks at
180/184/188/192/196/200/204 are134,137,150,145,150,150,150. At u180 D03=9/25;
u184 S03=19 and D03=18; u192 D03=20. The actual final model recovers all150,
but the registered all-check gate fails. No later subwindow substitution, entropy
sweep, independent final test or scratch-seed promotion. Completed paired scores,
retention and actual parameter-update range are in
`depth-parallel-postpass-entropy0-comparison.json`. Every update changes Actor
parameters with no KL skip. Previous full-sized velocity/yaw and horizontal-action
trials already failed; their existing records were read, not rerun.

The next diagnosis changes frozen parameter blocks instead of another optimizer
setting. Existing native Actor modules provide a natural boundary: CNN plus state/
perception projections and GRU produce features; a bias-free action_head maps them
to three accelerations. The auxiliary velocity head does not control frozen actions.
Reuse the original parallel full-pass u176 and its first failed control checku184,
including their validated150/150 and124/150 case records. Evaluate exactly two new
hybrids: u176 representation with u184 action_head, and the reverse. No original
model replay, weight hash, training, scene change or new final-test cases.

The300 new development episodes use the exact original150 identities per hybrid.
For both, report all scenes, recovered original losses and newly lost previously
retained cases. A sufficient frozen block-only repair needs every scene23/25 and
source-case retention143/150; failure rejects that block restoration as sufficient.
Cross-model feature/head co-adaptation may matter, so even a positive result does
not establish a unique cause or prove that freezing a block will train stably.
Only then consider a targeted learning intervention, without automatic architecture
changes. Prelaunch design and reproduction command are in
`candidate-configs/depth-parallel-components-design.json`; the one-off script reuses
native load_policy/evaluate and existing new-table scoring. Three identity-based
assembly self-checks prove source dictionaries and all other parameter references
remain unchanged; Ruff passes. No production code or dependency change.


The frozen component experiment launched from4cd42f3, native811eabe, PID2713095.
Both150-episode evaluations completed and the final comparison was saved. The u176
representation/u184 action_head hybrid scores128:25,25,22,25,6,25. The reverse
scores129:25,25,8,25,21,25. Both recover all26 original u184 losses, but introduce
22/21 failures on previously retained cases. First hybrid new failures are22
collisions; second has10 collisions and11 timeouts. Neither meets the registered
all-scene gate. No single-block restoration is promoted as a sufficient frozen
repair, and no module-freezing training follows automatically. These results do
not identify a unique mechanism behind the interacting parameter changes.

The process exited1 only after writing both full evaluations and status=completed:
cleanup iterated the evaluator cache's PolicyMethod as though it were an environment.
The cleanup now skips entries without sim, matching existing window behavior. A
small self-check executes the actual finally block with two environment stubs and
one method entry, confirming both closes and no method close. No evaluation was
rerun; original output/log remains. No native learning, dependency or task change.
### Complete sampled-return pilot registered after component counterfactuals

The next intervention removes the remaining rollout-cutoff value bootstrap while
retaining the existing512x64 PPO prefix, full recurrent optimizer passes, entropy0.005,
reward, actions, scene assignment and complete u176 learner state. Each unfinished
copied world continues under its frozen stochastic behavior policy to first true
termination. With gamma=lambda=1, its detached realized tail gives complete sampled
prefix returns. This is a standard return estimator with a project-specific simulation
implementation, not a new algorithm framework or a claim of uniquely identified cause.
Old procedural calibration already found small512-step target bias without solving
regression; high bootstrap fraction alone does not prove bias in the current state.

Pre-registered budget:8updates,262144prefix transitions, full150-case development checks
after4/8updates, each six scenes>=23/25 and retention>=143/150. No early stop or rollback.
All active tail samples count toward learner interactions; allocated padded simulator
slots and wall time are separate. Matched optimizer updates/prefix samples do not imply
equal total sample cost. This is not the full28update stability window, scratch evidence
or final benchmark. The completed original control is reused without model rehash/replay.

Implementation reaches native PPO, Trainer validation/accounting, existing training
documentation, targeted tests and a bounded evidence runner. Seven new checks pass,
including real finite-horizon update, exact checkpoint continuation and unchanged live
stream/RNG. Default-path targeted regression:24passed/1failed/34deselected in217.00s.
The failing `test_perception_real_update_and_complete_sensor_checkpoint[lidar-ppo]`
asserts a change in the first point encoder; it reproduces on unchanged811eabe in26.24s
at the same assertion. This is an existing small-sensor integration gap, not a green
full suite. A separate fixture investigation should inspect valid acquired points
before changing that assertion; this Depth pilot does not alter sensor acquisition.
Runner uses existing checkpoint/evaluation/case-quality helpers and inline assertions
for exact first-rollout comparison, finite real updates, tail completion and accounting.
The design and command live in candidate-configs/depth-parallel-mc-tail-design.json;
no current candidate GPU execution has yet occurred at registration.

The pilot launched from8675f29, PID2734377. Full-state setup114.672s; actual first
update97.949s, Actor delta0.50891 and all10 raw rollout statistics exactly match the
control. Extra tails:24234active transitions/82496allocated slots, all terminate.
No outcome claim before the full checks. Source provenance comes from native records.

The inherited LiDAR test failure is now isolated: all six Actor observations had zero
valid points when the test shrank each MID360 scan frame to32rays. Keeping its actual
20000-ray frame progression and selecting32 policy rays yields21valid points at step6.
Only the test fixture changes. Existing PPO/APG/SHAC tests all pass in62.25s, including
encoder parameter changes and exact checkpoint/inference recovery. The original24
passing regression cases and7 new estimator checks are not rerun after this fixture-only
change. Evidence is lidar-fixture-diagnostic.json; the production sensor is untouched.

### Complete-return pilot finished: late improvement, registered gate failed

All8updates completed, process exit0. u180=145/150 with S03=21,D03=24, four
timeouts and one collision; u184=148/150 with S03=D02=24, two collisions. Retention
is145/150 and148/150. The completed control is150/150 then124/150. At u184 the
candidate recovers all26 control losses while creating2 new losses; at u180 it
creates5 losses. Therefore the registered both-check gate fails, despite the late
paired improvement. Do not extend to28updates, promote a recipe, or run final tests
on the strength of final recovery. The optional mechanism remains default false as
a reproducible experiment with a measured late benefit, not a stable accepted recipe.

Actual costs:262144prefix +200301active tails =462445interactions;668672allocated
tail slots. Eight nonzero Actor deltas0.50891-0.56725; no KL skips or incomplete tails.
Update wall364.594s; complete execution excluding setup498.817s; setup114.672s.
Actual final learner/Actor is results/fixed_scenes_parallel/depth_ppo_s0_mc_tail_pilot/
model_000184. Native report,8 metrics rows and two new episode tables were copied once
to artifacts/verification/ppo-stability-20261015/mc-tail with comparison.json.
Single sampled tail/value RMSE contains return variance and is not a calibrated
conditional-value bias. No new unique root cause is established. The matrix remains
14/18; all four PPO Navigation cells still need three-seed stable scratch evidence.
No candidate process is active; a next intervention has not yet been registered.

### Scene-gradient conflict versus sampling variability probe registered

After the complete-return pilot, measure the original last-goodu180 state (150/150),
not the changed-recipe branch. Two copies use unchanged/folded810002 source RNG with
identical live physical/GRU/objective state; total65536training interactions. Native
811eabe is pinned, no manual source/model hashing or evaluation replay. Existing
eight training scene groups each split into two4world halves. Per-group PPO losses
retain global advantage normalization and original entropy; their average gradient
must reconstruct an independent full-batch derivative. Report within-scene, between-
scene and cross-draw cosine/norms, plus one local full-batch Adam-step direction and
surrogate change. This step is a counterfactual, not the native16-minibatch update.

The preregistered gate requires repeated substantial inter-scene conflict involving
S01/D03, positive within-scene half similarity, >0.5cross-draw coherence, and local
harm under both negative aggregate gradient and inherited Adam. It prevents selecting
PCGrad from a negative cosine alone or Adam momentum alone. Two draws cannot establish
population gradients, a unique cause, curvature conditions or episode-level stability.
PCGrad paper section2.2 explicitly says conflict alone need not be harmful. No native
learner change or PCGrad integration is made for this probe. Three analytic cosine
self-checks and Ruff pass; full gradient identity will be checked on the actual data.
The exact design and command are candidate-configs/depth-scene-gradient-probe-design.json.

The first probe attempt collected32768transitions then failed gradient reconstruction:
relative error0.00033434236 exceeded5e-5. No conflict conclusion was accepted. Original
report/log are retained as attempt0-report.json/attempt0.log. Group forward shape4
versus full64 is a suspected numerical source, not yet proven. The corrected probe
keeps the complete64-world forward and masks group losses; tolerance stays5e-5.
Failed-attempt data were not saved before its assert, so the repeated32768sampling
interactions are explicitly charged (98304total including two final draws). New raw
rollout/gradient arrays will be saved before assertions to allow diagnosis without
more sampling. This is an execution correction, not an outcome-dependent gate change.

The fixed64-world masking attempt reduced reconstruction error to8.360139e-5, still
above5e-5. Full gradient norm0.441662, absolute discrepancy3.69218e-5; float64 host
reduction does not remove it. The saved attempt1-gradient-identity.npz and rollout
allow a no-resampling numerical recheck. Reuse the already supported scoped compiler
options (exclude_nondeterministic_ops=true,autotune_level=0), with highest matmul
precision only inside the differentiated loss. Collecting physics remains original.
Require exact repeated full gradients and retain5e-5 decomposition tolerance.
Replicate0 is reused; only replicate1 needs new32768samples, so total physical work
including the lost attempt0 remains98304. Old attempts did not record full elapsed
wall time; cumulative wall cost is unavailable. This measures gradients with explicit
numerical settings, not a claim of bit-identical native minibatch optimization.

### Gradient diagnostic complete; full optimizer batch pilot registered

The scoped deterministic/highest-precision loss passes: both repeated full gradients
are exactly equal, and half-group reconstruction relative errors4.6543e-7/5.5661e-7.
The eight cross-draw scene cosines are[.33255,-.01022,.48907,.21340,-.08341,.03494,
-.11443,-.14369]; none meets0.5. Within-scene half cosines are negative in5/8 and
4/8scenes. Every scene has a descending aggregate-gradient direction and a negative
actual local full-batch Adam surrogate change in both draws. No PCGrad selection
pair qualifies. This does not prove population gradient variance or a unique cause;
it rejects automatic scene-gradient projection under the preregistered rule.
Sampling cost98304including the lost first attempt,65536samples in the two measured draws; last
execution172.826s including113.893s setup. Earlier complete wall cost was not recorded.
Raw rollouts/gradients and failed attempts remain local; report.json is the new record.

Next bounded intervention changes only native minibatches4->1 from originalu176,
not the MC-tail branch. Same64x512rollout/four epochs and original learning rates;
Actor and Critic Adam steps both16->4. Source state, scenes and cases stay matched
to the completed control. Eight updates262144interactions with full150-case checks
at180/184; both require all six>=23/25 and retention>=143/150. No early stop, rollback
or automatic final-test use. Full-batch gradient observations motivate the candidate,
but fewer optimizer steps and changed Critic fitting prevent unique variance attribution.
The existing continuation tool gains a full-batch flag and diagnostic-only short budget;
no native learner or new optimizer abstraction is needed. Existing collector rejection
now has an explicit full-batch diagnostic case.33targeted tests pass25.70s; the expanded
Actor+Critic counter assertion also passes separately in19.20s before launch.

Before pilot outcomes: native PPO shares one learner RNG between collector, epoch
permutations and minibatch entropy draws. Changing4->1minibatches consumes fewer
optimizer RNG splits, so only the initial RNG/source state and first rollout are
pathwise matched; later action draws differ. This is an explicit further limit on
unique causal attribution. No custom RNG-burning mechanism is added to the native
full-batch configuration. The three-seed requirement remains necessary.

Full-batch pilot is live, PID2779340, runnerf2b6a7a and native811eabe. Source setup
114.611s; firstactualupdate177 took57.840s including compilation, Actor delta0.137621
and no KL skip. All15 saved raw-rollout/target statistics exactly match the control.
This verifies initial matching and real updates, not retained capability. Native
report correctly marksdiagnostic=true and4optimizer steps per update.

### Full-batch pilot complete: both checkpoints fail

All8updates completed and process exited0. u180=141/150, S03=16/25 with7collisions
and2out-of-bounds; u184=135/150, D03=10/25 with15collisions. Other five scenes pass
25/25 at each check. Retention141/150 and135/150. Original control150/150 then124/150
is reused; exact recovered/new/common loss counts are in full-batch/comparison.json.
No gate passes, so do not extend the budget, promote a recipe, or use final tests.
Local full-batch surrogate improvements did not predict episode-level preservation.

Cost262144interactions; Actor deltas0.13564-0.15093, no KL skips. Training134.899s,
including evaluations368.563s, setup114.611s. Actual final state remains at
results/fixed_scenes_parallel/depth_ppo_s0_full_batch_pilot/model_000184. Cached new
reports,metrics and both150case CSVs are under full-batch/. No native code changed.
First failed S03 collision is world4 at15.892s; all nine S03 failures occur14.708-
15.998s. D03 failures occur19.776-20.310s. These early clustered failures justify
recording a paired critical-obstacle trajectory rather than inferring behavior from
aggregate loss. No such new trace evaluation has yet been registered or executed.

### Paired S03 trajectory measurement complete; parallel LR-only window registered

Fifty recorded frozen episodes reproduce every original case identity/event: controlu180
25/25 and full-batchu18016/25. All seven collisions hit obstacle_154; two failures cross
the6mupperboundary. At x~15.5m, focusedworld4 control has vy=-0.3696m/s while candidate
has+0.1704m/s. The candidate collides15.892s at[15.912,.094,5.734], still82.134mfromgoal;
control completes109.38s at0.499m. Minimum control clearance0.212m. Native recorded
commands are roll/pitch/yaw/thrust, not normalized Actor actions. Measurement54.423s,
no new training, no other scenes or final cases. Raw traces and all50case summaries
remain s03-full-batch-trajectories/; the matched picture localizes behavior but does
not identify a unique optimization cause. First-terminal truncation and finite-cylinder
distance analytic self-checks pass; all recorded events/times/initial positions match.

The minimal untested setting for the current parallel source is the existing0.1xActorLR
continuation. Prior sequentialu144 0.1xLR failed every check and is explicitly retained;
this measures interaction with eight-scene coverage every update, not a repeat from
that sequential source. Keep original4x4updates, all other settings, and full28update
917504window with seven150-casechecks. Every check must pass allsix>=23/25 and retain
>=143/150. No earlystop/rollback or finaltest. Existing LR-only correctness tests are
reused; no new optimizer/regularizer/code needed. Exact command and rejection rule are
candidate-configs/depth-parallel-postpass-lr01-design.json.

### Parallel Actor LR-only window completed and failed

All28updates917504interactions completed; exit0. Seven scores139/128/150/148/147/125/139.
Onlyu188/u192pass; finalS03=14/25, otherfive25/25, retained139/150. u200S02=0/25.
EveryactualActorupdatepositive, noKLskip. First15rawrollout/targetmetrics exactlymatch
control. Cachednewreport/metrics/sevenCSVsandreportsareparallel-postpass-lr01/; no
source model rehash or replay. Actualendstateu204 retained; noextension/promotion/finaltest.
The next larger-physical-batch experiment is registered separately:256worlds,2updates,
262144interactions comparedto8full-batch64worldupdates. Bothhave32Adamstepsperlearner
and64worldoptimizerbatches. Morephysicaltrajectories and slowerpolicyrefresh differ;
not a unique variance attribution. Sourceworlds cloned contiguously with native
Crazyflowworld_mask for physical leaves and staticcore.n_worlds corrected. A realCPU
Depthupdate caught the missingstaticfieldbeforelaunch. Nativealgorithmunchanged.

World-expansion verification is now complete:35other scopedchecks passed; expanded
realDepthtest passed38.69s afterusing Crazyflowworld_mask and correctingstaticn_worlds.
It checks4->8worlds/twoscenes,64x48sensor, preservedglobalphysics arrays/parameters/Adam/RNG,
expandedphysical/sensor/GRU/objectivehistory, nonidentical stochasticclonepositions,
actual24cumulativeinteractions andtwoActor/CriticAdamcounts. Earlier testfixtures
usedunsupportedDepthsizes and a NumPy typed-key conversion; these were corrected
before the meaningful physical expansion test. No GPUtraininglaunchedbeforepassing.
LRwindowcost410.982s updates,966.367s inclchecks/saves excluding113.572ssetup.
Actor parameterdelta.058846-.098535, noKLskip.

Timing correction after inspecting the actual timer and reconciling its parts:
ppo_stability_window starts its clock BEFORErestoration. Thusfull-batch368.563s
andparallelLR966.367s INCLUDE114.611s/113.572ssetup respectively; earlier wording
excluding/plussetup waswrong. Correctedderivedcomparisonfields/designresults/status;
rawrecordsunchanged. Fullbatch134.899train+117.159eval+114.611setupplusIOmatches368.563.
LR410.982train+436.604eval+113.572setupplusIOmatches966.367. MC-tail isdifferent:
its498.817timer startsaftersetup and is364.594train+132.840evalplusIO; its114.672setup
isindeedadditional. Do notaddsetupagainfornativewindowresults/newworlds256pilot.

### Largerphysicalbatch pilot passes bothchecks; fullwindow registered

256worldpilot completedexit0: u177148/150[D02/D03=24], u178149/150[D02=24],
other scenes25. Bothgatespass; two/onecollision. Relative tofullbatch atmatched
131072/262144interactions, recovers9/15losses andadds2/1newlosses. Secondcheckrecovers
all26originalcontrol lossesbutaddsoneD02loss. No uniquevariance attribution.
Actor deltas.378837/.364521, noKLskip.32Actorand32CriticAdamsteps,262144interactions,
120.242supdate,total370.985sincluding118.520setup. Savedactualu178andnewcached
reports/CSVs/metrics/comparison underworlds256-pilot/.

A freshsame-source full7update917504window ispreregistered. This intentionally repeats
the firsttwo checkpoints to test reproducibility with knownnativeGPUnumerics; it is
not a newindependenttrainingseed. No resume abstraction/code needed. Every one of
seven checks must passallscene/retention gates; complete throughactualu183. Shortpilot
cost isadditional, notpartof the917504budget. Stilldevelopment,notformalacceptance.

Nextcollection-only preregistration is depth-parallel-success-episodes-design.json.
Oldrehearsalnegative explicitlyreread:proceduralu30021/48,47/64stochsuccesses,188sparse
64-stepwindowswith32burnin,0.1KL,final4/48ret3/21. Newproposedcorpus isoriginalparallel
u176self-policy,64newtraininginitialstates,deterministicdemonstrationactions,complete
successfulobservationhistories fromzeromemory; noevalstate. Require>=4successesineachsix
primaryscene andteachermeanrecompute<1e-4beforeanylearningcandidate. SourceGaussian
log_stdcomesfromfullstate. No BCweightselectedorcodeimplementedyet. CLEARsection3
primarypaperread(policyKLterm,notfullV-trace/valueclonealgorithm). Thisis a prerequisite
forapotentialPPO+self-policy-cloning diagnostic,notprooforformalrecipe.

### Full256worldwindow terminal; nextbounded from-scratch check

Completedexit0,917504interactions,7updates:149/125/144/150/150/150/150.
D02=0then19causestwofailures. Final150andlastfour150do noterasefailures.
Native source/caseidentitychecks remain valid; all7Actorupdatespositive.352301-.387329,
noKLskip.112Adamstepsperlearner,246.986supdate,total720.247sincluding118.671setup.
Initial15rawmetrics exactlymatchpilot; firstActorparamdelta.3788757vspilot.3788374.
Firsttwochecks differfrompilotby3and24successoutcomes. Cachednewrecordsareworlds256-window/.
Actualfinal183kept. Noextension/reanchoring/finaltest ofthisfailedpostpasscandidate.

Finalfourpassingchecks motivate the smaller nextimplementation choice:existing256world
configfromscratchseed0,notnewpolicy-cloningcode. The collection-only fallback isnow
explicitlydeferred/notimplemented. Presetonlychanges64->256worlds,evaluation/checkpoint
16->4(to retain524288interactioncadence) andoutput. Same8.388608m totalbudget64updates,
noinitialcheckpoint/resume,otherfields identicalbyassertion. NativeHydracomposition
exactlymatchesafterexcludingtwooptionalWarpimportnoticesfromstdout. Sourcepinned811.
Newdesign/configdepth-fixed-scenes-worlds256-scratch-design.json and
depth_fixed_scenes_parallel_worlds256_s0.yaml; nothree-seedclaim. Firstfullpass mustbe
followedbyunchanged256worldfullstability; auto2mbenchmarkremainsdevelopment.

Scratch256Depthrun islive PID2855827/execsession50540, launchedfrom8db5dfb; native
run.jsonconfirms811eabe/d3f2443source. Firstactualupdate1has131072interactions,finite
metrics,Actorgrad.578099,depthvalidfraction.385905,noKLskip. Completedtrainingepisodes
0/221success atinitialrandompolicy, notscreening orconvergence. Firstupdate74.101s,
native sessionwall237.467sincludinginitialization. Firstscreenpendingatupdate4/524288.
Nootherownedtrainingjobslive. Do notrelauch/resetthisprocess. Exactcommand/config/design
aredepth-fixed-scenes-worlds256-scratch-design.json; outputresults/fixed_scenes_parallel_worlds256/depth_ppo_s0.

### Constant256 scratch candidate: first two completed screens

The random-initialized native811 Depth run completed u4/u8 screening at524288/1048576
interactions with0/23 successes outof48; u8 scene counts8/8/0/7/0/0. The original64world
control at identical interactions scored6/39. All23 current successes belong to the
control success set; no additional control cases succeed. Native CSVs do not contain
true goal-progress fields, which remain explicitly unavailable. The new artifact script
`summarize_worlds256_scratch.py` checks complete48case identity/scene/event consistency,
reads each new screen once, and reuses both earlier current and original control caches.
The registered8,388,608 budget continues unchanged. No convergence claim or new loss.

### Constant256 scratch first full pass and unchanged stability registration

New complete screens u12/u16/u20/u24 are40/16/39/48. The40->16 loss is retained as
acquisition-stage regression. Actual first full validation u24 at3145728 interactions
is149/150, six scenes25/25/25/25/25/24. This is earlier than original64world seed0
first full pass at5767168, but remains single-seed evidence. Native automatic2m
benchmark has identical primary counts; auxiliaryS06=0/25,D06=25/25. It is development
only and does not consume the3m final cohort. Native process exited0; total wall
1685.970s, native session wall1684.968s, sum24 updates804.822s. These wall measures
already include setup/evaluation; do not add them again. Full reports/CSVs, run record
and24 metrics cached once underworlds256-scratch. Complete model stays native.

Registered unchanged sourceu24 continuation7 updates917504interactions, everyupdate
150case validation at the original1.5m seedbase. Source149 successes require142
retained, plus every scene>=23/25. Keepfullsource physics/history/RNG/Adam,256worlds
and originalPPO settings. No diagnostic flags/world expansion. Execute entirewindow
evenifanycheckfails. This is the stability prerequisite, not final3seed acceptance.
