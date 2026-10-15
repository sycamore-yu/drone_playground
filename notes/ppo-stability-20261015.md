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
training objective or the comparison settings. Resume the exact saved states
after the targeted retention checks pass.
