# ADR-0036: Consistent egocentric Navigation acceleration commands

**Status:** Implemented as an opt-in, release-pending trainable policy contract.
**Scope:** Navigation policies using three-axis `acceleration`; tracking/racing,
`velocity_yaw`, `body_rate`, planners and previously accepted checkpoints keep
exactly their previous action contracts.

## Evidence and problem

`NavigationEnv._observe_state` encodes velocity, target direction and gravity-up
in the aircraft body frame, with sensors also mounted and encoded in the body
frame. The original three-axis `Action(level=acceleration)` assumed the Actor
outputs world-frame net acceleration. Two states with the same egocentric
observation but different world headings can require different policy outputs.
Adding nine body-to-world rotation matrix values to the Actor input makes the
world-frame output inferable, but introduces global-heading dependence and
increases network state size. On the newer matched procedural training suite,
the original body-observation/world-action SHAC baseline achieved 0/100, while
the rotation-augmented SHAC baseline achieved **50/100** (S01/S02 each 25/25;
S03/S06 each 0/25), with worst-scene success rate 0. The latter is a
demonstrated partial improvement, not evidence that either representation
fully solves the task. The body-action change must be measured separately.

## Contract

For Navigation, the preferred experimental three-axis command is the **net
kinematic acceleration in body coordinates**, not accelerometer specific force:

`a_world = R_world_from_body(q_issue) @ a_body`

The physical controller already expects world net acceleration, adds gravity
compensation and outputs the existing attitude/thrust setpoint. Rotate the
command once at the policy decision/queue issue time; preserve the resulting
world-frame command throughout physical control delay. Do not rotate it again
at a later physics tick. The same transformed world acceleration must feed the
PointMassLag surrogate. Named physical acceleration/jerk losses use the world
command; action regularization can still use the declared normalized policy
command frame. The physical observation `up_body` remains available for tilt,
and the Actor need not receive a body-to-world rotation matrix.

The `Action` contract declares `frame` and persists it with the checkpoint.
`frame=world` remains the default to preserve the current 14 accepted
Tracking/Racing recipes and old checkpoints. Only `level=acceleration` opts into
`frame=body`. Other frame/level combinations continue to validate or reject as
before. LOTF already uses `body_rate` in body coordinates, not acceleration.

This is **not** a blanket claim that all UAV policies should output full-body
acceleration. A yaw-stabilized heading frame (horizontal axes rotated by yaw,
vertical aligned to gravity) is another conventional choice; adopting it would
also require transforming the observation consistently and declaring a distinct
action contract. We choose full-body here because that is what the current
velocity/goal and mounted sensors already express.

## Primary-source comparisons

- [DiffAero](https://github.com/flyingbitac/diffaero) exposes multiple dynamical
  models (full quadrotor and point masses), sensor inputs and BPTT/SHAC/PPO. It
  supports the general principle of separating policy representation from
  dynamics; it does **not** prove a particular local acceleration convention.
- [P2M](https://github.com/arclab-hku/P2M) combines LiDAR-based perception with
  end-to-end navigation in clutter; its value here is task/sensor evaluation,
  not a mandate to copy an unverified action coordinate convention.
- [NavRL's released ROS runner](https://github.com/Zhefan-Xu/NavRL/blob/main/ros2/navigation_runner/scripts/navigation.py)
  expresses intermediate policy velocity in a local target-aligned coordinate
  system and explicitly converts it to the world command for PX4/ROS. It uses
  velocity commands, **not** the same acceleration controller as this project.

## Identified adjacent issues (with evidence status)

1. **Verified, fixed for candidate:** `Action.setpoint` expected world
   acceleration while Navigation Actor saw body velocity/goal. The new opt-in
   conversion uses `Rotation.from_quat` and a reference-controller equivalence
   check under nonzero yaw, pitch and roll. Neither policy frame nor physics
   sign is left implicit.
2. **Verified, fixed for candidate:** `PointMassLag` custom JVP uses a **world**
   `model_input`, delivered through `CommandState.inputs`. Simply rotating the
   Mellinger setpoint without rotating that delayed input would produce a
   forward-identical but derivative-inconsistent surrogate. The issued physical
   queue now stores the world acceleration in both native and differentiable
   paths, and the model's Jacobian is checked.
3. **Verified, fixed for candidate:** `EpisodeObjective.cost` (Zhang/Liu)
   previously called `decode(action)` assuming physical **world** acceleration,
   including its acceleration/jerk history. The converted world acceleration is
   now supplied explicitly. The paper Fan acceleration path has an analogous
   conversion; previous-action smoothing remains a declared command-local
   difference (not a claim of exact physical jerk for changing attitudes).
4. **Verified, not part of the 42 supported OA acceleration baseline:** The
   `velocity_yaw` Navigation policy takes the same body-relative state but its
   `Action.setpoint` compares the output velocity directly to world velocity.
   This is another potentially under-observed world-frame policy. It requires
   its **own** observation/action redesign and tests, rather than silently
   changing the existing historic velocity policy and its evidence.
5. **Verified train-vs-test distribution mismatch:** Matched procedural scenes
   do not cover the horizontal beam requiring a vertical passage in fixed S06;
   original SHAC S06 collided there. The coordinate correction cannot teach
   flight through unobserved geometry by itself. A new distribution/source
   experiment is required if the updated body-action baseline still fails.
6. **Conditional timing mismatch:** With nonzero sensor latency, the delivered
   `points_body` are in the **acquisition-time** body frame, while current
   kinematic state features use the present body frame. The default studied
   latency is zero, so this cannot explain the existing failure; a future
   latency study needs motion compensation or an explicit relative pose/time.
7. **No-op setting, not a fix:** `initial_log_std` controls PPO Gaussian action
   sampling, not the deterministic BPTT/SHAC path. Procedural initial positions
   come from the scene generator, not the old random-navigation-start flag.

## Independent checks and promotion rule

- Unit: distinguish frames and retain 3-D normalized bounds; reject unsupported
  body horizontal/velocity action contracts; arbitrary-rotation setpoint equality
  against the old world policy; analytic JAX acceleration Jacobian.
- Dynamics: forward-equivalent PointMassLag and native steps with world-frame
  delayed input; finite nonzero model gradients; named physical-cost consistency.
- Regression: old default world contracts, fixed-world rotation diagnostic,
  original control models, historical OA recipe resolution and sensor paths.
- Real training: a small, from-scratch CPU BPTT/SHAC rollout smoke validates
  finite updates without claiming convergence.
- Results: create a distinct source-pinned Autoresearch session with new
  `+method.action.frame=body` and `++observation.navigation_world_rotation=false`.
  Preserve the historical loss/reward/optimizer, training distribution, seeds,
  N/horizon/iterations and fixed 24/96 model selection. Compare the four fixed
  scenes by `eval_worst_scene_success_rate` and publish only after a separate
  untouched 100-case independent confirmation.

Do not change the public OA release recipe in place while it has no accepted
body-action checkpoint. Continue iterative diagnosis with the already authorized
single GPU queue; once proved, promote the successful recipe only in its cell.
