# ADR-0020: Diagnose perceptual PPO before replacing the learner

**Status:** Accepted for an isolated experiment, 2026-10-10

## Findings

The existing [ADR-0014](0014-rl-environment-and-method-interface.md) already
establishes a standard Brax \`Env.reset/step/State\` adapter. It explicitly keeps
native Brax PPO/APG as **optional baseline algorithms**, not replacements for
the in-house PPO/APG/SHAC. \`learning/brax_adapter.py\` and
\`tools/smoke_brax.py\` exist. The adapter currently rejects sensor observations
and native Brax is not installed in the primary Pixi environment. For
Depth/CNN/GRU and LiDAR/PointNet/GRU, swapping the PPO library is therefore
**not drop-in**: it would also change the policy network, recurrent rollout
contract, normalization, action distribution and checkpoints.

The old repository's Navigation PPO employed
\`brax.training.agents.ppo.train\`, a **4D velocity/yaw policy**, its own
velocity-control inner loop, \`MotionNavigationReward\`, 128 environments,
8 minibatches, 4 updates per batch, \`gamma=.995\`, \`clip=.1\` and nonzero
entropy. This is not the same causal experiment as the new **3D acceleration**
policy with dense Zhang/Liu costs, 16 differently randomized environments,
a CNN/PointNet+GRU Actor and independent privileged Critic. Old archived
\`results/runs/navigation/ppo/final-acceptance-heldout-*\` records show
**0/6 arrivals for both static and dynamic trials**. The older
three-seed Depth-static success reported in \`docs/status.md\` belongs to
other perception-learning evaluations. It cannot establish that old Brax PPO
solved the identical Navigation PPO benchmark.

Old-source references:
\`drone_playground/src/drone_playground/configs/experiment/navigation/ppo.yaml\`,
\`.../control/controllers/velocity.py\`,
\`.../environments/tasks/rewards.py\`,
\`.../learning/train.py\`.

DiffAero (\`cfg/algo/ppo.yaml\`) uses normalized advantages, 4 epochs,
8 minibatches, value clipping and nonzero entropy. VisFly's PPO wraps
Stable-Baselines3 and logs KL, clip fraction, policy/value losses, explained
variance and action standard deviation. These are *choices for controlled
comparisons*, not evidence that a particular hyperparameter alone fixes the
current navigation failure.

## Decision

1. Keep standard Brax Environment compatibility, but **do not replace the
   current recurrent PPO blindly**. Default native Brax PPO and the current
   Brax adapter do not preserve the perception Actor/GRU contracts as-is.
   Separately smoke-test Brax on supported state-only tasks; do not mix its
   policy checkpoints with native JAX Actor weights.
2. Preserve existing acceleration/Zhang/Liu experiment histories and fixed
   benchmark geometry. Add opt-in \`Action(level=velocity_yaw)\` executing
   velocity error through an explicit PID-like world-acceleration and the same
   Crazyflow \`attitude_thrust\` plant. Preserve old-like limits/gains and
   record them in the action contract. Do not grant the policy extra geometry.
3. Add \`learning.options.reward_profile=legacy_motion_navigation_v2\` as
   an **opt-in training-only** reward: velocity tracking, clearance, progress,
   arrival bonus, failure penalty and 0.01 scaling. This is a causal reference
   inspired by the old source, not a claim of identical training or network.
   Require \`loss=null\`, Navigation PPO and \`velocity_yaw\`.
4. Introduce an opt-in per-world \`SceneDistribution.start_sampling=
   collision_free\`. Training may begin at different distances along the
   course, including near the terminal goal, with collision-clearance
   rejection. Frozen S01-D06 geometry, reset seeds and success criteria are
   unchanged. Clearance alone is **not a proof of path existence from every
   randomized start**, so later procedural maps still need a graph/reachability
   validation.
5. Expose training-only diagnostics: actual forward progress in metres,
   collision and out-of-bounds fractions, remaining distance, Critic
   explained variance/RMSE, advantage positive fraction, raw advantage/
   progress correlation, sensor valid fraction, KL and clipping statistics.
   These logs help distinguish Critic overfitting from poor reward and
   action learning; do not tune on one-time frozen benchmark results.
6. Give the optional old-inspired **composite reference recipe** its own named YAML
   \`navigation_procedural_{depth,lidar}_ppo_velocity\`. Admission at
   procedural trial 4 only if the preceding candidates have failed.
   All seeds begin with random weights; the existing Seed 0 trial 3 remains
   immutable. Compare wall time, interactions/sec, progress, first collision,
   checkpoint-validation and independent benchmark using the SAME suite.
   This candidate changes several ingredients together (action, reward,
   reset distribution and PPO sampling). A success can justify a subsequent
   one-factor-at-a-time study but **cannot identify which ingredient caused it**.
7. The automatic loop is not proof of convergence. Only the existing
   three-independent-seed frozen protocol may increase a formal matrix cell.

**Reward ownership clarification:** [ADR-0021](0021-task-owned-reward-contract.md)
supersedes the temporary Learning-side reward implementation. Existing
in-flight Trial-4 settings retain their original configuration for exact
checkpoint recovery. New experiments select reward under Task.

## Acceptance

- Both velocity policies pass CPU JAX and GPU short-update tests; action
  output dimension is 4, applied setpoint is true attitude/thrust, and
  reward sign and terminal scale have direct unit tests.
- Different seeded procedural environments remain distinct; start samples
  are collision-free at reset. Historical artifacts and existing fixed
  geometry hashes do not change.
- New runs record PPO advantage-progress and Critic diagnostics with finite
  values and keep complete resumable optimizer/RNG/recurrent state.
- Old recorded policy successes are not silently credited to Brax or to the
  new procedural runs.
- No GPU comparisons under mismatched environment count or learning budget
  are called an algorithm speedup.

## References

- Brax: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground/blob/main/learning/train_jax_ppo.py
- DiffAero: https://github.com/flyingbitac/diffaero
- VisFly: https://github.com/SJTU-ViSYS-team/VisFly
- Stable-Baselines3 recurrent PPO: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html

## 2026-10-11: Native Brax navigation replication (measured)

Re-ran the unmodified old repository's `experiment=navigation/ppo` with
`env=navigation/static`, source revision `da343fd`, Brax 0.14.2, JAX 0.9.2,
Crazyflow 0.3.2 and a real RTX 4090. The old environment executes a
50 Hz velocity/yaw policy over Crazyflow at 500 Hz, with a sensor-fusion
64×64 feedforward network and the `motion-aware-v2` task reward. Training
uses the **fixed S01/S02/S03/S06 geometry**, unlike the new procedural
Navigation training distribution; checkpoint evaluation uses separate reset
seeds but not unseen geometry. The original repository is left untouched.

| Independent old run | Worlds | Real steps | Static checkpoint evaluations | Best arrivals |
| --- | ---: | ---: | ---: | ---: |
| `legacy-brax-nav-static-ppo-s0-20261011` | 16 | 131,072 | 5 × 8 episodes | 0/8 |
| `legacy-brax-nav-static-full-s0-20261011` | 128 | 2,097,152 | 5 × 8 episodes | 0/8 |
| `legacy-brax-nav-static-resetmix-s0-20261011` | 128 | 2,097,152 | 5 × 8 episodes | 0/8 |

The full run completed normally, with a nonzero Actor parameter L2 change
(3.462) and checkpoint updates at 0, 524,288, 1,048,576, 1,572,864 and
2,097,152 interactions. At the final checkpoint, 6 of 8 episodes collided,
2 left the boundary and none reached the goal. The corresponding mean
evaluation returns were −10.266, −10.084, −10.204, −10.170 and −10.561.
There is **no observed old native PPO navigation convergence**, and no
justification for replacing the new recurrent PPO with Brax on this evidence.

The old reward scaling is relevant for future controlled diagnosis. For a
96 m nominal start-to-goal displacement, its progress contribution is only
`0.01 * 0.2 * 96 = 0.192`, versus an arrival bonus of `+1` and a terminal
failure cost of `-10`, before continuous motion/time penalties. This can
incentivize risk avoidance without solving the task; it is a hypothesis, not
a proved root cause. At 50 Hz, the old `discounting=0.995` discounts a bonus
48 seconds away to approximately `0.995 ** (48 * 50) = 0.0000059` of its value;
this long-horizon signal issue is also a hypothesis, not proven causation.

The third run changed only `training.reset_randomization.position` to a
0.25/0.25/0.50 mixture of departure, near-goal and free-course starts,
retaining the old Brax learner, network, controller, reward and identical
nominal evaluation cohorts. It completed at 2,097,152 interactions with
a nonzero Actor parameter L2 change (3.444). All five checkpoint evaluations
again recorded zero arrivals out of eight. All eight final episodes collided;
final mean evaluation return was −10.141. Mixed starts alone **did not
establish convergence**.

The original three independent runs and their configurations are preserved
under `drone_playground/results/runs/navigation/ppo/<run_id>/`. Because neither
the original configuration nor this one-factor variant converged, no Brax
PPO replacement or old-policy checkpoint transfer is justified. Further work
should independently test the reward/time-horizon contract rather than
mistaking Critic loss reduction for navigation success.

The new repository's existing Brax compatibility smoke was also executed
via the isolated old Python environment:
`tools/smoke_brax.py --algorithm ppo --num-envs 4 --timesteps 64` passed
with a nonzero native policy update.
This checks only the **state-observation adapter**, not Depth/CNN/GRU or
LiDAR/PointNet/GRU convergence. Do not merge their checkpoints or equate
these different observation/action/training contracts.
