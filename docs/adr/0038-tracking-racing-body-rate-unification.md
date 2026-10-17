# ADR-0038: One Tracking/Racing policy action and retire unused velocity experiments

Approved: 2026-10-17. Status: implemented; new from-scratch convergence acceptance
is pending per combination. Supersedes the Tracking/Racing action-choice and
Navigation velocity-experiment portions of ADR-0020, ADR-0036 and ADR-0037.

## Physical meaning and literature

The learned action for every Tracking/Racing Train method is now
`[T, omega_x_B, omega_y_B, omega_z_B]`: collective thrust in newtons and body
angular velocity in rad/s, in the installed Crazyflow body's FLU coordinates.
The scalar thrust is along the aircraft thrust axis; it is not a world-vector
component. The policy order is explicitly rearranged to Crazyflow's native
`[omega_x_B, omega_y_B, omega_z_B, T]` exactly once in Action.setpoint. The state
angular velocity is already body-frame in Crazyflow; it must not be rotated
again. Position/trajectory reference vectors remain world-frame and orientation
is available in the observation. Body-rate commands are not Euler-angle rates.

This follows the action class used by **Champion-level drone racing using deep
reinforcement learning** (Kaufmann et al., Nature 2023), which commands collective
thrust and body rates through the low-level Betaflight controller. Its thrust
is mass-normalized, while this repo stores physical newtons; that normalization
must not be copied as the same numeric control quantity.
Source: https://www.nature.com/articles/s41586-023-06419-4 (Methods and Fig. 2).

For trajectory tracking, **DATT: Deep Adaptive Trajectory Tracking for Quadrotor
Control** (CoRL 2023), Section 3.1, also outputs desired total thrust and body
rates: https://arxiv.org/html/2310.09053 . It uses a separate first-order
closed-loop response model; the current migration does not claim to reproduce
that full controller or copy its model gains.

The previous attitude/thrust action is not inherently invalid. For example,
**Vision-Based Agile Landing on Turbulent Waters** (Angelis et al., 2026),
Eq. (1), predicts `[roll, pitch, yaw, collective thrust]` tracked by an attitude
controller: https://arxiv.org/html/2605.23717v1 . The two interfaces differ by a
control layer: holding a roll-angle target is not holding roll angular velocity.
The PX4 control diagrams explicitly separate attitude and rate controllers:
https://docs.px4.io/main/en/flight_stack/controller_diagrams .

We unify the *learned Tracking/Racing* interface to avoid changing action
semantics when selecting a backward model. We retain physical attitude/thrust
setpoints for planners, MPC and the existing OA acceleration controller. A
higher-level Navigation policy that emits velocity plus yaw is a different
control-layer experiment, not a third mandatory RL interface.

## Implementation

All fourteen public Tracking/Racing Train recipes, the direct environment
defaults, and Command Builder emit `body_rate` with `frame=body`. Native and
LOTF share their original body-rate execution. Physical simulation remains
Crazyflow first-principles; historical objectives, optimizer settings, horizon
and iteration cap are retained. A change in action is not permission to alter
the task, evaluation thresholds or rewards.

PointMassLag previously accepted world net acceleration only. For CTBR it now
uses the existing exact body-rate quaternion integration and maps collective
thrust through that attitude to world net acceleration, then applies the same
first-order acceleration lag. This is explicitly a **reduced closed-loop
backward model** with kinematic attitude, not a new rigid-body forward model.
It substitutes quaternion/position/velocity/acceleration derivatives while the
actual forward trajectory and other derivatives stay native. Queued delayed
commands are used at their actual application time. OA PointMassLag's original
acceleration-input variant remains unchanged in the same implementation.

The representation change invalidates the old native and acceleration-based
Tracking/Racing models as acceptance evidence for the new interface. Even the
already-body-rate LOTF combinations receive the requested new training and
evaluation in this migration. Keep original records as Imported Prior rather
than deleting or rewriting them.

## Cleanup

Remove the unused `velocity_yaw` learned action, its extra velocity PID wrapper,
its four public experiment YAMLs, and its specific `legacy_motion_navigation_v2`
reward. Remove the duplicate learner-owned reward-profile selector; active
reward profiles remain owned by Task. Retire tests solely supporting those
removed experiments and replace them with a test rejecting the removed API.
Historical results and pinned source worktrees remain intact; their exact old
semantics remain reproducible there rather than through a growing compatibility
layer in current release code. Other experiment families with active, distinct
published methods are retained; this cleanup is not permission to delete data.

## Verification and experiment protocol

A deterministic test resolves the actual Command Builder's fourteen commands
and requires the same CTBR action for every algorithm/backward-model choice.
Tests also assert thrust/rate ordering, positive pitch response, zero-rate hover,
finite four-channel derivatives, unchanged forward transitions and no derivative
before an action's declared delay. Unsupported retired actions fail explicitly.
Relevant sensor, OA, controller, learning and checkpoint tests still apply.

Prepare a fresh Autoresearch `recipe={}` baseline for each of the fourteen
required combinations using `prepare_training()`. This is required coverage,
not a parameter grid. Each session pins source, action, task, seed, environments,
horizon, budget and predeclared Best. New reference-task runs use the same
24-episode quick check and 96-episode confirmation as the release protocol;
old 8/100-check histories remain Imported Prior. Public execution defaults to
GPU without changing the optimizer or rollout batch. Use the existing single GPU queue; retain
its active OA training without restarting. Reviewed development winners are
frozen before a separate 100-episode evaluation on unused release seeds. Only
that result accepts the new recipe; historical acceptance is a separate column.
Additional independent training seeds follow first acceptance.
