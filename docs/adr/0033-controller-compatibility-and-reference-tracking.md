# ADR-0033: Controller applicability follows the reference contract

**Status:** Proposed for release; implemented and tested as a local `main`
candidate. The user's approval covers RPS98 Full-State NMPC integration;
the exact final release catalog remains reviewable before publication.
**Scope:** Public Train/Play/Benchmark combinations and general tracking NMPC.

## Context

An MPC algorithm is not automatically compatible with every flight task.
Racing contouring optimizes progress along ordered gates; navigation in
unknown obstacles requires a planner producing a collision-free trajectory.
Ordinary trajectory tracking optimizes fidelity to a supplied time reference.
Mixing these produces misleading method comparisons and unsupported commands.

## Decision

Use three independent classifications: **task**, **method role**, and
**reference contract**. A controller's physical output (BodyRate vs
AttitudeThrust) is separately enforced by Simulation. No task-specific
solver should be offered simply because it can output the same motor commands.

| Controller | Reference contract | Tracking | Racing | Navigation |
|---|---|---|---|---|
| Sampling MPC | Time-valid position/velocity trajectory | Yes | Yes | With an EGO/SUPER planner |
| LSY Attitude MPC | Time-valid position/velocity trajectory | Yes | Yes | With an EGO/SUPER planner |
| Full-State NMPC | Time-valid position/velocity/acceleration trajectory | Yes | Yes | With an EGO/SUPER planner |
| Racing MPCC | Ordered gate course and arc progress | No | Yes | No |
| Perception-Aware Sampling MPC | Time-valid trajectory and known Racing gate landmarks | No | Experimental | No |

Controller source code uses a single `PlannerController` host execution
contract; no additional task-specific dispatch paths are necessary. For
Navigation, only `Planner + Controller` is presented publicly. General NMPC
does not become an obstacle-avoiding planner simply by receiving goal states.
EGO-Planner and SUPER are independent ROS workers; their generated trajectory
must remain valid over the sampling horizon. Planning success and tracking
quality are measured independently in Benchmark.

**RPS98/mpc** (<https://github.com/RPS98/mpc>) provides the full-state
quaternion/position/velocity and collective-thrust/body-rate control
formulation. The local Full-State NMPC is independently implemented in
acados using nominal Crazyflow mass/thrust parameters. It derives an attitude
reference from planned acceleration and yaw, then outputs a physical BodyRate
Setpoint to the unchanged Crazyflow engine. It is a **named adaptation**, not
an exact copy of the upstream ROS/C++ integration or an obstacle-aware MPC.

## Release command matrix

Keep `train`, `play`, and `benchmark` as three public entrypoints. The Command
Builder's `controllerCapabilities` is a declarative source for supported
task/reference matches. The catalog does not perform automatic combinations
of every controller, task, sensor and solver.

- Train: 42 configurations (unchanged).
- Play: 380 configurations: policies, generic trackers for Tracking/Racing,
  task-specific Racing controllers, and the supported Planner + Controller
  combinations for Static/Dynamic OA.
- Benchmark: 128 configurations, following the same method applicability rules
  with fixed suites and explicit performance metrics.
- Total: **550 unique commands**. Every command must compose through Hydra;
  independent end-to-end execution is an additional release acceptance gate.

The older 230-command smoke matrix and 486/490 configuration revisions are
historical verification artifacts. They are not substitutes for the current
550-command execution acceptance. Existing results and checkpoints remain
unchanged by this decision.

The existing Mellinger and SO3 tracking controllers are available through
their Hydra controller configurations. They are not automatically multiplied
across the current release Builder's method matrix, because that matrix is a
deliberately selected set of MPC comparisons. Their omission from the Builder
does not mean the controllers are unsupported by Simulation.

## RPG quadrotor control

[UZH RPG Quadrotor Control](https://github.com/uzh-rpg/rpg_quadrotor_control)
can provide a further high-speed tracking method through a thin local adapter.
Its `PositionController::run` consumes a state estimate and `Trajectory`, and
can output BodyRate or attitude commands; its rotor-drag-aware feedforward
also consumes reference jerk/snap, heading and heading derivatives. The current
Drone Playground `Trajectory` exposes position, velocity and acceleration,
so a faithful drag-aware implementation first requires an explicit extension
of those reference derivatives and nominal drag coefficients. Generating
arbitrary zero jerk/snap would be a separate simplified variant, not the paper
implementation. Keep it as a proposed, non-MPC baseline until a concrete
port and controlled flight check are approved.

## Consequences

Do not call the public class `Full-State NMPC` Racing MPCC or imply it plans
through obstacles. Do not advertise the experimental Perception-Aware Sampling
MPC as sensor-fed or real-time: it uses known gate landmarks and has not met
the 50 Hz CPU budget. Solver feasibility does not establish flight safety;
native replay and fixed-cases Benchmark remain the reporting standard.
