# ROS1 deployment

This directory owns the operating environment for the pinned EGO and SUPER
implementations. It contains no Python adapter or replacement planning algorithm.

| File | Purpose |
|---|---|
| `Dockerfile` | ROS Noetic and C++ build/runtime dependencies |
| `versions.env` | Upstream revisions, image and container names |
| `setup.sh` | Fetch, patch and build the actual planners |
| `patches/` | Declared deployment-specific upstream changes |

`drone_playground.integrations.ros1.planner.RosPlanner` copies a run-owned worker/protocol
bundle to the existing container and starts it. The worker translates data; the
actual executables remain under `/opt/drone_playground/planners/`.

This is a ROS1 environment, not a generic ROS2 runtime for SANDO/MIGHTY.
`setup.sh` replaces the configured container. Only run it for an explicitly
requested rebuild. Ordinary evaluation does not rebuild or restart the container.
