# ADR-0030: Racing control and browser replay

**Status:** Accepted 2026-10-16; integration scope amended 2026-10-16.
**Implementation:** Selected methods merged into local `main` with independent
Play/Benchmark and MuJoCo browser-replay checks.

## Control-method boundary

Implement acados Racing MPCC with arc-length contouring, lag and progress
objectives on the existing Crazyflow predictive model. Use the published
[LSY acados MPCC](https://github.com/lihengrongfei/lsy_drone_racing_ws26)
architecture as a technical reference. The new solver uses local stage
linearization and is **not** an exact port of the LSY project or Romero's paper.

Also include the native JAX/Crazyflow known-landmark Perception-Aware Sampling
MPC as an explicitly experimental controller. Its image-plane/field-of-view
objectives were inspired by Falanga et al.'s PAMPC and the sampling pattern of
DeepMind MJPC. This controller does **not** use the original RPG ACADO solver,
ROS1, or images measured from a physical camera. It has not passed a 50 Hz
real-time or full Racing benchmark and must be labeled accordingly.

The user subsequently excluded two methods from the `main` integration:

1. The original [RPG Time-Optimal](https://github.com/uzh-rpg/rpg_time_optimal)
   optimizer + Attitude MPC trajectory adapter. Its original numerical solver
   can converge without satisfying the project's collision model; closed-loop
   Crazyflow tracking is not yet accepted.
2. The original [RPG PAMPC](https://github.com/uzh-rpg/rpg_mpc) ROS1/ACADO
   controller. It has not been built and validated as a native host method.

These remain source-backed research in `feature/racing-mpc-implementation`.
Neither original implementation, CSV adapter, pinning utility, nor its
associated tests are distributed from `main`. The implementation does not
vendor external GPL software.

## Execution boundary

Both integrated controllers use the existing `PlannerController` protocol
for one-world host execution and the same Crazyflow plant, gate task, case
initialization and metrics as learned policies and native controllers. Keep
Play for inspecting a flight and Benchmark for paired suites and performance
reports. A full fixed-Level0 MPCC test completed 8/8 cases with five ordered
gate crossings and mean time 14.682 s. The JAX perception-aware controller
is executable but its measured CPU decision time exceeds 20 ms.

## Visualization boundary

Use Autoresearch Dashboard for training metrics and short sampled frames;
MuJoCo WASM for portable browser 3D of the exact recorded flight; RScope for
sensor and planning overlays; and TensorBoard for optional optimizer analysis.
Do not create a second simulator, replay physics loop, or parallel recording
protocol. Export browser JSON and native `.mj_unroll` from the same recorded
timestamps, poses and scene geometry. See [visualization](../visualization.md)
for the one-port SSH workflow.
