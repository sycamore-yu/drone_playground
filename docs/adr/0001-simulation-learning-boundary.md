# ADR-0001: Simulation and Learning responsibilities

**Status:** Accepted. The evaluation ownership is superseded by [ADR-0026](0026-training-benchmark-release-separation.md).
**Approval:** Originally recorded as 2026-10-08; the exact timestamp was not retained.
**Implementation:** Method construction and both execution paths are present. Benchmark extraction is implemented and runtime-tested under ADR-0026.
**Last reviewed:** 2026-10-16T06:24:13Z, working tree based on `7d59162`.

## Context

Optimal control, native planners, and frozen policies need shared tasks, physics,
and evaluation semantics. Learning also needs batched, differentiable JAX execution.

## Decision

Use one installable Python package. Simulation owns tasks, scenes, sensors,
physical execution, runtime methods, and replay. Learning owns PPO, BPTT, SHAC,
objectives, optimizers, and explicit training state. Learning depends on Simulation,
which depends on official Crazyflow. ADR-0026 moves evaluation protocols and report
aggregation into Benchmark.

Construct methods with Hydra before execution. `PlannerController(planner,
controller)` coordinates the actual planner/tracker pair: planning intervals,
trajectory validity, and controller memory. It does not own search, maps, networks,
or an arbitrary sequence of processing stages. Direct policies execute through
their action contract. Task names and sensor names do not determine action units.

JAX methods receive explicit parameters, state, and random keys. Native C++/ROS
methods and host solvers run through host execution. Both use the same environment,
episode rules, and record format. Ordinary RPC calls do not enter pathwise autodiff.
The concrete entry points are `simulation.runner.rollout` and `rollout_host`.

The environment owns the physical clock, command application, sensor acquisition,
and episode events. Trackers are injected into the method. Policy acceleration
commands use the existing Mellinger conversion; attitude and body-rate commands
use Crazyflow's corresponding low-level control. The user confirmed this division
at **2026-10-10T10:46:22+08:00**, reviewed against `v0.2@6ec50c7`. That approval did
not require an arbitrary policy-side controller option.

Dynamics equations and controllers remain independently callable. Algorithms
share the transition interface instead of duplicating rollout logic for every
model. Approved controllers include Mellinger, SO3, attitude MPC, sampling MPC,
and explicit ideal tracking. Trajectories retain a host NumPy representation;
SE3 expansion remains deferred.

## Consequences and evidence

Ideal tracking selects a declared state-response model; a controller cannot
silently overwrite physical state. The SO3 implementation follows the pinned
[EGO SO3Control source](https://github.com/ZJU-FAST-Lab/ego-planner/blob/bfda51284c8c1b476043255a8145ef925a3778a5/src/uav_simulator/so3_control/src/SO3Control.cpp).
Input conversion, nominal parameters, and remaining low-level control are
specified in [control models](../control-models.md). The implementation lives in
`simulation/methods.py`, `controllers.py`, `mpc.py`, and `runner.py`.
