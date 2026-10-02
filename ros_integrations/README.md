# Native integrations

This directory contains non-Python native integration projects.

- `sdk/`: ROS-independent C++ gRPC SDK for external algorithms. It is an independent CMake/Pixi project and therefore lives outside the Python package under `src/`.
- `ros1/`: deployment, pinned sources, patches and bridge code for the external EGO-Planner and SUPER ROS1 processes.

The Python side of the same interface lives under `src/drone_playground/rpc/`. Physical commands and action application live under `src/drone_playground/actions/`; simulation dynamics live under `src/drone_playground/dynamics/`.

The C++ SDK is not a planner implementation. The ROS1 directory does not duplicate EGO-Planner or SUPER: it fetches pinned upstream sources into its ignored `sources/` cache and builds them inside the configured ROS1 container.
