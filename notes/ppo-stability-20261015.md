# PPO stability experiment design

Objective: the user attachment `6bfd2216-f69e-4121-97b8-8f1181e64d2a/pasted-text-1.txt`,
including stable scratch Depth/LiDAR PPO and the original 18-cell BPTT/SHAC/PPO matrix.
This is a working experiment design, not evidence of convergence or a new architecture.

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
