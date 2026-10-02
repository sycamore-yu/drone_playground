# ROS1 planner deployment

This directory packages the pinned EGO-Planner and SUPER integration. It contains deployment and integration code, not a second implementation of either algorithm.

| File | Responsibility |
|---|---|
| `versions.env` | Pinned repositories, commits, image and container names |
| `setup.sh` | Fetch source and explicitly rebuild the ROS1 container |
| `docker/Dockerfile.ros1` | ROS Noetic build/runtime environment |
| `patches/` | Separate documented upstream corrections |
| `bridge/worker.py` | gRPC service that drives ROS planners and returns complete physical trajectories |

The current container is `drone-playground-ros1`. Its source and executables are:

```
/opt/drone_playground/planners/ego/src/ego-planner/
/opt/drone_playground/planners/ego/devel/lib/ego_planner/ego_planner_node
/opt/drone_playground/planners/super/src/SUPER/
/opt/drone_playground/planners/super/devel/lib/super_planner/fsm_node
```

`planners/super` is a catkin workspace; `src/SUPER` is the upstream repository inside it. They are not two algorithms. The optional host source cache is `ros_integrations/ros1/sources/`; the present deployment is already built inside the container.

The host integration is `src/drone_playground/integrations/ros1.py`. Each new run snapshots the worker, command types and RPC implementation into its own container bundle, then connects via the shared Python gRPC client. Observations flow to the planner; the returned trajectory flows to the configured controller and the same simulation.

Experiment presets live only in `configs/experiment/ego_planner.yaml` and `super.yaml`.
The naming migration does not invoke `setup.sh`, restart Docker, alter planner source, or rebuild binaries. `setup.sh` replaces the configured container and is reserved for an explicitly requested rebuild.
