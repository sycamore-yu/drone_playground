# ADR-0021: Task-owned rewards and Learning-owned auxiliary losses

**Status:** Accepted.
**Approval:** Originally recorded as 2026-10-10; the exact approval time was not retained.
**Implementation:** Task owns physical reward equations and explicit reward profiles. Learning composes task costs with auxiliary losses. The former configuration translation is removed under ADR-0026.
**Last reviewed:** 2026-10-16T06:24:13Z against `main@7d59162` and the working tree.

## Decision

Task owns physical task reward semantics. PPO, BPTT, SHAC, Actor, Critic, and
network implementation details do not belong in Task reward equations.

- `simulation/tasks.py` owns `Task.reward_profile` and `Task.reward_shaping`.
- `simulation/task_rewards.py` implements progress, arrival, failure, clearance,
  height, and the optional `legacy_motion_navigation_v2` formulation.
- `BaseEnv.cost` and `BaseEnv.task_reward_components` expose task costs to
  evaluation, Learning, and the optional Brax adapter.
- `learning/objective.py` combines task costs with explicitly selected
  perception, acceleration, jerk, and velocity-prediction losses.

Current recipes select the physical reward explicitly:

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
```

`EpisodeObjective` still accepts the existing objective coefficients from
Learning; explicit Task shaping takes precedence. This coefficient composition
uses the same task-domain cost implementation. The remaining Learning-side
`reward_profile` field checks agreement with the already constructed Task; it
does not select or reconstruct the Task profile.

## Historical transition

The initial implementation translated a non-default
`learning.options.reward_profile` into `Task.reward_profile`. That preserved the
then-running velocity-policy configurations while introducing separate
`*_ppo_velocity_task.yaml` recipes. The original on-disk configurations and
checkpoints remain historical evidence.

[ADR-0026](0026-training-benchmark-release-separation.md) removes that translation
from the current factory. Both current velocity-recipe variants now explicitly
set the Task profile. Old configurations are executed with their original
source revision; the current loader does not fill missing configuration values
or restore the old ZIP format.

## Checks

Task and Learning objective tests compare their physical cost when Task owns
the shaping coefficients and auxiliary losses are disabled. Real short CPU
updates pass for the depth and LiDAR velocity recipes. The task name, physical
geometry, terminal-event ordering, and reward equations remain unchanged by
this ownership refactor. Benchmark and training use the same Task implementation.

ADR-0026 separately removes the historical aggregate acceptance thresholds and
training-time checkpoint validation. Those changes do not alter per-episode
success conditions or retrospectively change older experiment results.

This decision supersedes the temporary reward-file placement in
[ADR-0020](0020-ppo-diagnostics-before-brax-replacement.md) and follows the
ownership boundary in [ADR-0014](0014-rl-environment-and-method-interface.md).
