# External algorithm integration

This package owns external runtime adapters and their transport. Task success,
control and simulation physics stay with their existing modules.

| Module | Responsibility |
|---|---|
| `service.NativeServicePlanner` | Start a program or connect to an address implementing the shared gRPC contract |
| `ros1.RosPlanner` | Run the pinned EGO/SUPER ROS1 implementations |
| `ros1/worker.py` | Translate simulated clock, state, sensor data and goals to native ROS inputs |
| `ros1/trajectory.py` | Convert real B-spline/polynomial messages to the common timed Trajectory |
| `ros1/visualization.py` | Convert actual planner visualization to independently timed SFC data |
| `rpc/` | Protocol messages, client/server, serialization and deadlines |

Methods return `runtime.Decision`. Its executable value is a Reference, Setpoint
or Actuation. A sampled tracking reference and planning inspection data are
separate fields. The controller performs the actual trajectory sampling once.

## Add a C++ algorithm

Generate protobuf messages and the gRPC service from `rpc/proto/algorithm.proto`.
Implement the generated `drone::native::v2::Algorithm::Service`; there is no second
project-specific Algorithm base class. Implement Initialize, Reset, Step and Close.
Declare the inputs and outputs actually supported, convert the request to the
solver's inputs, and return the real result with correct units, frame and timing.

A constructor example for an already implemented external service is:

```yaml
_target_: drone_playground.integrations.service.NativeServicePlanner
settings:
  algorithm: my_planner
  output: trajectory
  deployment:
    command: [/absolute/path/to/planner_service, "{address}"]
  parameters: {}
```

This is not a completed SANDO/MIGHTY recipe. Their algorithm-specific conversions
still need implementation. Docker is optional for a native executable; an already
running service can use `deployment.address`.

A trajectory-only service uses a controller configured with
`reference_source: trajectory`. The controller samples the trajectory; the adapter
does not synthesize a second reference. `reference_source: execution_sample` selects an
actual sampled control reference returned by a planner such as the ROS1 backends.

## Add a ROS algorithm

Check its ROS distribution, solver dependencies, input/output message definitions
and simulation-clock behavior first. A ROS adapter publishes the common state,
sensors and goals using those messages, then converts the native trajectory back
to the common interface. It does not replace the algorithm or act as the controller.

The current ROS1 adapter implements EGO and SUPER only. The official SANDO and
MIGHTY repositories use ROS2. Their deployment and ROS2 topic/trajectory adapters
are not implemented here yet. They can reuse the same protocol and downstream
Controller/Dynamics after that adapter is implemented. SANDO additionally requires
its solver installation and license configuration.

First verify reset, time, units, frames and stale output rejection. Then verify a
real solver output reaches the controller and physical step. The echo service in
`tests/native` tests transport only; it cannot demonstrate planner success.
