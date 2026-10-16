# ADR-0010: Training-only privileged PPO critic

**Status:** Accepted.
**Approval:** Originally recorded as 2026-10-09; exact time was not retained.
**Implementation:** Present in PPO and the environment's privileged observation function. Convergence remains a separate experiment.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Decision

Use `learning.options.critic_uses_privileged=true` for an independent value network
in depth/LiDAR OA PPO. Reuse the existing PPO updates, recurrent actor, GAE, and
checkpoint state. The default is false; it is mutually exclusive with a sensor
critic. Continuation requires an identical saved learning contract.

The training critic receives body features, true global position, goal displacement,
quaternion, linear/angular velocity, previous action, normalized time, signed
clearance, and 26 ideal world-frame rays. Directions are the nonzero vectors in
`{-1,0,1}^3`, capped at 40 m. Features have fixed dimensions across scenes and are
not restricted by sensor noise, latency, or field of view.

The actor keeps its existing body and device observations. Frozen inference loads
only actor parameters and never calls the privileged observation function. Label
this as asymmetric PPO, with training information permissions recorded explicitly.

## Evidence and limits

The original implementation record reports five targeted checks and 239 CPU tests;
it did not establish privileged PPO convergence. The design follows the input
separation in pinned [DiffAero PPO](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/algo/PPO.py)
and [OA state](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/env/obstacle_avoidance.py).
Physics, networks, and task rewards remain project-specific.
