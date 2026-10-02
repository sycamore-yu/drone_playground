# C++ algorithm SDK

`ros_integrations/sdk/` contains the ROS-independent C++ SDK, not the EGO-Planner or SUPER algorithms.
It exposes lifecycle operations and physical outputs over gRPC.

- `include/drone_native/algorithm.h`: C++ algorithm interface.
- `src/service.cc`: gRPC service implementation.
- `tests/interop_server.cc`: an interoperability fixture, not a planning baseline.
- `CMakeLists.txt`: builds the SDK and optionally the fixture.
- Shared wire schema: `src/drone_playground/rpc/proto/algorithm.proto`.
- Python client/server: `src/drone_playground/rpc/`.
- Physical values and units: `src/drone_playground/actions/commands.py`.

`Trajectory`, `Waypoint`, and typed motion commands have explicit frames, units, timestamps, and validity intervals. The service interface does not require ROS or differentiability. The installed SDK environment remains at `ros_integrations/sdk/.pixi/`.

ROS1 planner deployment is documented in `../ros1/README.md`.
Existing container binaries and frozen experiment bundles are not rebuilt by the naming migration.
