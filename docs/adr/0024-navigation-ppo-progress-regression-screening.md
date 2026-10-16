# ADR-0024: Navigation progress and policy-regression diagnostics

**Status:** Accepted; training selection superseded by [ADR-0026](0026-training-benchmark-release-separation.md).
**Approval first recorded:** 2026-10-14T10:48:42Z, the timestamp of commit `e261127`. The original message time was not retained. The user reaffirmed the two principles at 2026-10-14T12:20:24Z.
**Implementation:** The former progress screening and retention logic was integrated into `main`. ADR-0026 removes it from current training; Task distance controls, observations, and historical experimental evidence remain.

**Last reviewed:** 2026-10-16T05:37:19Z against `main@7d59162` and the working tree.

## Motivation and historical decision

An early Fan-inspired experiment reached D02 8/8 after a long zero-success
period. Zero success alone therefore did not establish that a long-range policy
had stopped learning. A depth mixed-distance run later fell from 26/48 at update
2880 to 0/48 at update 4320. Paired rollouts showed that all 26 lost successes
ended at the lower height boundary. Reward, value fit, and a newer checkpoint
could not establish retention of previously successful behavior.

The historical screening workflow recorded initial, final, and minimum goal
distance; actual progress; movement before failure; path length; terminal time
and event; height range and boundary margin; goal-height error; and value-target
RMSE/explained variance. Improvements and risk changes required calibrated,
matched thresholds. Continued safe progress could justify using the remaining
declared budget; reward alone could not reset patience. Two-scene probes used
2 × 8 episodes, and complete screens used 6 × 8 episodes.

The workflow retained best and latest checkpoints and compared repeated full
screens. Persistent loss of success or height control triggered causal diagnosis
rather than an automatic budget increase. Resuming a retained checkpoint created
a new experiment identity. Mixing action components from different checkpoints
was a counterfactual diagnostic, not an accepted policy.

These diagnostics remain useful measurements. They no longer control checkpoint
selection, early stopping, or retention in the current training loop.

## Speed and start-distance experiments

The user requested fast, safe completion without specifying a 3 m/s target or
hard speed limit. Existing target-speed features, perception costs, and Fan's
`max_speed: 3.0` were recipe assumptions. They were not user requirements or an
explanation for hovering 2.94 m from the goal.

The first execution record for the approved removal of a speed target and a
20 m start-distance comparison is 2026-10-14T12:38:41Z. Commit `f9eb5c0` introduced
`navigation_distance_v01`: separate depth and LiDAR runs, starting with 20 m and
then 80, 10, 30, and 50 m. Within this batch, only the true three-dimensional
`start_goal_distance_range_m` changed. Setting the earlier
`min_start_goal_distance_m=20` had still sampled 80 m starts and was not the
requested comparison.

The new baseline replaced target-speed features with relative goal position and
used time, progress, clearance, and safety costs. It removed target-velocity
tracking loss while retaining world-frame acceleration, orientation input, and a
privileged Critic. Comparisons with older recipes changed several factors;
only within-batch comparisons isolated start distance.

The first budget was 2,211,840 interactions, or 4,320 updates. Success-rate
patience could not fire before that budget, and external zero-success pruning
was disabled. Eight-episode training-distribution probes and 48-episode fixed
screens were recorded separately. This was a bounded experiment, not calibrated
progress-based stopping. Earlier Fan and VisFly runs stopped after saving full
recovery states at updates 786 and 268. Their original data remain unchanged.

## Stability evidence reviewed on 2026-10-15

With mixed starts and static-obstacle count scaled to 0.7, the seed-zero depth
run peaked at 21/48 and ended at 2/48; LiDAR peaked at 11/48 and ended at 0/48.
For identical S01 initial states, depth changed from 8/8 at update 2880 to 0/8
at update 4320. Measured discounted-return calibration error changed from
12.44 to 55.13. These observations do not identify a unique optimizer cause.

The local `navigation_stability_fixes_v01` batch tracked episode identities:
new successes could not compensate for loss of reference successes. A 6/8 to
6/8 example retained only five of six prior successes and failed the retention
check. Full checkpoints were copied independently of rolling cleanup. The depth
update-240 checkpoint deleted before archival was enabled was recorded as
unrecoverable. Training parameters, optimizer, and RNG were not rolled back.
The experiment-specific script and original records remain local evidence.

Actor learning rate, rollout length, GAE lambda, and value clipping had separate
controls. The 128-step arm compared lambda against its matching 128-step control.
The `fan2025_arrival` Task reward replaced continuing hover/survival rewards
with a time penalty and a one-time arrival reward. Both Fan controls disabled
speed penalties. Physical comparisons at 0.51 m and about 0.97 m confirmed that
arrival returned more reward than continued hovering; that result did not show
that PPO had learned to arrive.

The former 23/25 release threshold and training-time screening protocol are
historical. Current experiments report actual completion, independent benchmark
outcomes, and retention of previously successful episodes under their declared
protocol, without restoring the removed selection state machine.
