# ADR-0004: Native C++ and ROS integration

**Status:** Accepted.
**Approval:** Originally recorded as 2026-10-08; exact time was not retained.
**Implementation:** Present in the Python client and independent ROS1 worker.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Context

EGO-Planner and SUPER use C++/ROS; physical flight uses Crazyflow/JAX. Their
boundary needs explicit time, measurement, frame, and trajectory contracts.

## Decision

Use Protobuf/gRPC for initialization, episode reset, decisions, and close. The
ROS1 Noetic worker converts messages and invokes the actual upstream solvers.
Exchange physical state, depth or point-cloud measurements, goals, and trajectories
with units, frames, timestamps, and validity intervals.

Simulation consumes native decisions, passes references to the selected controller,
and advances Crazyflow. Methods can be deployed independently while sharing task,
Benchmark, and replay semantics. Adapt ROS2 methods to their real dependencies
when separately requested.

## Implementation layout

The 2026-10-09 naming update placed the handwritten `RosPlanner` client and generated
wire code in `src/drone_playground/simulation/ros_planner/`. The independent C++
build is `src/ros_planner_worker/`. Generated files remain separate from handwritten
code. The RPC name is `drone_playground.ros_planner.v1.RosPlanner`; older worker
images require rebuilding. See [native planner operations](../ros_planner.md).
