# Native runtimes

This directory contains non-Python build and deployment projects. It is not part of
the installed `drone_playground` Python package.

- `sdk/`: ROS-independent C++ gRPC SDK for external algorithms.
- `ros1/`: pinned source identities, Docker build files and deployment-specific
  patches for the external EGO-Planner and SUPER ROS1 processes.

The transport protocol lives under `src/drone_playground/rpc/`; installed Python
deployment adapters live under `src/drone_playground/integrations/`. Physical
interfaces and application live under `src/drone_playground/control/`; simulation
dynamics live under `src/drone_playground/dynamics/`.

The C++ SDK is not a planner implementation. The ROS1 directory does not duplicate EGO-Planner or SUPER: it fetches pinned upstream sources into its ignored `sources/` cache and builds them inside the configured ROS1 container.
