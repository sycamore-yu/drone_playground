# Native planners

This directory owns the ROS1 runtime used by the native navigation planners in
`drone_playground`. It is independent of FlightBench and builds from
`ros:noetic-ros-base-focal`.

`versions.env` pins the upstream planner repositories and commits. `setup.sh`
checks out those revisions under the ignored `sources/` directory, builds the
project-owned `drone-playground-ros1:noetic` image, starts the
`drone-playground-ros1` container, and builds EGO-Planner and SUPER in isolated
catkin workspaces under `/opt/drone_playground/planners`.

Run:

```bash
bash native_planners/setup.sh
```

The simulator stays outside ROS. Each evaluation episode starts its own ROS
master inside this container and communicates with the bridge using JSON lines
over `docker exec` standard input/output. The planner source checkouts and build
context are intentionally excluded from Git; upstream license files remain in
their respective source trees.
