# Installation

The supported development environment is Linux x86-64 with Python 3.12. Pixi
locks Crazyflow, JAX/CUDA 12, MuJoCo/MJX, Orbax, and RScope.

```bash
pixi install --locked
pixi run autoresearch setup
```

Autoresearch setup requires `git` and `uv` on PATH. It creates the pinned sibling
checkout `../autoresearch-gym/` and its separate Python environment. Native
training uses this repository's Pixi environment. GPU training requires a
compatible NVIDIA driver.

For a CPU controller check, select both the JAX platform and Simulation device:

```bash
JAX_PLATFORMS=cpu pixi run play simulation.device=cpu \
  task.reference=hover task.duration=0.2 output=results/controller-check
```

Inspect a resolved configuration before launching an experiment:

```bash
pixi run drone-playground experiment=oa_static_depth learning=ppo --cfg job --resolve
```

## Attitude MPC

The optional native controller uses acados v0.5.1 and requires a C/C++ build toolchain:

```bash
pixi run setup-acados
export ACADOS_SOURCE_DIR="$PWD/.cache/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
pixi run play controller=attitude_mpc task.reference=hover task.duration=0.2
```

`pixi run setup-acados --source /path/to/acados` reuses a matching native checkout.
Each run generates its solver in its own output directory.

## Native planners

Build and start the independent ROS1 worker as described in
[ROS planners](ros_planner.md). EGO defaults to localhost port 50051 and SUPER to
50052. A worker permits one active client; concurrent experiments need distinct
worker instances. Planner workers do not require host ROS or GPU access.

## Installed package

`pixi run build` creates a wheel with Hydra configurations, scene assets,
Crazyflie display resources, and licenses. `drone-playground` provides Play,
Benchmark, replay, and the native training subprocess. Public training
orchestration uses the source checkout's `pixi run train` task and sibling
Autoresearch installation.
