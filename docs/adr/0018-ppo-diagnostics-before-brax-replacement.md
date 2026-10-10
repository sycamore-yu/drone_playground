# ADR-0018: Diagnose perceptual PPO before replacing the learner

**Status:** Accepted for an isolated experiment, 2026-10-10

## Findings

The existing [ADR-0012](0012-rl-environment-and-method-interface.md) already
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
