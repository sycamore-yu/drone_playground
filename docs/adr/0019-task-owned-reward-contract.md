# ADR-0019 — Task-owned rewards, Learning-owned auxiliary losses

**Status:** Accepted · 2026-10-10

## Decision

**The Task/Environment domain owns all physical task reward semantics and their
configuration.** No PPO, APG, SHAC, Actor, Critic or neural-network concept may
appear in a Task reward implementation.

- `simulation/tasks.py` owns the `Task.reward_profile` and optional
  `Task.reward_shaping` contract.
- `simulation/task_rewards.py` implements the task cost terms: progress,
  arrival, failure, clearance and height penalties, plus the optional
  older `legacy_motion_navigation_v2` formulation.
- `NavigationEnv.cost` exposes the same configured task reward to
  independent evaluation and the optional Brax Environment. The shared
  `BaseEnv.task_reward_components` interface gives Learning the task cost
  and named task metrics without inspecting `Task.name`.
- `learning/objective.py` combines those task terms with explicitly
  selected Zhang/Liu perceptual prediction, acceleration and jerk losses.
  Learning algorithms update parameters and estimate advantages; they do
  **not** select or implement goal/obstacle/terminal reward equations.

The corresponding configuration for **new** experiments is:

```yaml
task:
  options:
    reward_profile: legacy_motion_navigation_v2
    reward_shaping:
      task_weight: 1.0
      progress_reward_scale: 0.0
      failure_cost: 0.0
      altitude_weight: 0.0
      height_boundary_weight: 0.0
learning:
  options:
    reward_profile: default
```

`learning.options.reward_profile` and the original task-reward shaping fields
were stored in older full TrainingState configuration identities. Existing
running Trial-4 velocity PPO recipes are **byte-for-byte preserved** for
checkpoint restoration; `create_environment` translates their historical
non-default Learning selector to `Task.reward_profile` before constructing
`Task`. It never changes the saved run or frozen benchmark contract.
The two new `*_ppo_velocity_task.yaml` presets are opt-in, separate
experiments; they are **not** automatically substituted for in-flight jobs.

For old runs without explicit Task shaping, `EpisodeObjective` passes the
historical coefficients to the task-domain evaluator. If Task explicitly
supplies shaping, those parameters take precedence. This is a transitional
configuration migration, not a duplicate implementation of reward math.
Eliminate the historical Learning fields only in a subsequent versioned
breaking migration, after their outstanding checkpoints are retired.

## Invariants and acceptance

1. Default Task reward arithmetic and all existing frozen test geometry,
   termination priorities, numeric coefficients and acceptance thresholds
   remain unchanged.
2. A new reward profile must be chosen and recorded as part of the Task
   contract, not inferred from a particular PPO implementation.
3. Independent `env.cost`, `env.task_reward_components` and
   `EpisodeObjective.cost` return the same task cost when no auxiliary
   Learning loss is enabled.
4. Validation and training may use the same Task reward with different
   observation/Method/Actor choices; no Task imports anything under
   `learning/`.
5. Existing on-disk velocity PPO run configs and their optimizer, JAX RNG and
   recurrent-state checkpoints remain restorable, with no silent replay of
   new reward logic.

Supersedes the temporary reward file placement documented in
[ADR-0018](0018-ppo-diagnostics-before-brax-replacement.md), and implements
the existing ownership decision in
[ADR-0012](0012-rl-environment-and-method-interface.md).
