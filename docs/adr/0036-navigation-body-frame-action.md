# ADR-0036: Consistent egocentric Navigation acceleration commands

**Status:** The body-frame Action conversion is implemented. All new OA Train
defaults select it, while their independent release acceptance is pending.
**Scope:** Navigation policies using three-axis `acceleration`; Tracking/Racing,
planners and previously accepted checkpoints retain their own documented frames.
Navigation `velocity_yaw` uses body-frame velocity in new recipes (see ADR-0037).

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
The legacy Action class fallback `frame=world` is retained for existing
Tracking/Racing recipes and old checkpoints; all published OA Train YAMLs now
explicitly select `frame=body`. The `velocity_yaw` interface also permits
body-frame *velocity* in new Navigation recipes while retaining its explicit
world-yaw command. LOTF already uses `body_rate` in body coordinates.

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

1. **Verified, fixed for OA defaults:** `Action.setpoint` expected world
   acceleration while Navigation Actor saw body velocity/goal. The new declared
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
4. **Verified and fixed for new Navigation velocity recipes:** The previous
   `velocity_yaw` Navigation policy used body-relative observations yet sent
   a world-relative velocity request. It now explicitly sets `frame=body` and
   rotates the requested velocity at issuance into the world PID. Legacy world
   checkpoints remain distinguishable by their action contract.
5. **Verified train-vs-test distribution mismatch, addressed by a new training
   generator:** Matched procedural scenes originally omitted S06's horizontal
   beam/vertical passage. New versioned OA defaults sample randomized beams,
   including stationary beams among dynamic obstacles. This does not modify
   fixed benchmark geometry or imply complete independent validation.
6. **Conditional timing mismatch, ego-motion now compensated:** Delayed LiDAR
   `points_body` refer to the acquisition body frame, whereas policy kinematic
   inputs use the current body frame. New OA defaults transform acquired points
   into current body coordinates and reproject Depth snapshots, retaining
   timestamps and masks. This does not deskew a rolling LiDAR or predict
   obstacles that have moved since capture. See ADR-0037.
7. **No-op setting, not a fix:** `initial_log_std` controls PPO Gaussian action
   sampling, not the deterministic BPTT/SHAC path. Procedural initial positions
   come from the scene generator, not the old random-navigation-start flag.

## Independent checks and promotion rule

- Unit: distinguish frames and retain 3-D normalized bounds; reject unsupported
  body horizontal actions; arbitrary-rotation setpoint equality
  against the old world policy; analytic JAX acceleration Jacobian.
- Dynamics: forward-equivalent PointMassLag and native steps with world-frame
  delayed input; finite nonzero model gradients; named physical-cost consistency.
- Regression: old default world contracts, fixed-world rotation diagnostic,
  original control models, historical OA recipe resolution and sensor paths.
- Real training: a small, from-scratch CPU BPTT/SHAC rollout smoke validates
  finite updates without claiming convergence.
- Results: create a distinct source-pinned Autoresearch session using the new
  body-action, variable-beam and time-aligned observation defaults.
  Preserve the historical loss/reward/optimizer, declared training distribution, seeds,
  N/horizon/iterations and fixed 24/96 model selection. Compare the four fixed
  scenes by `eval_worst_scene_success_rate` and publish only after a separate
  untouched 100-case independent confirmation.

Public OA Train recipes now request body actions and the new training/observation
semantics, but **checkpoint-level convergence acceptance remains per cell**.
Continue through the authorized single GPU queue; freeze and promote a trained
model only after separate held-out release confirmation. Historical world-frame
records remain reproducible with explicitly pinned configurations.
