# Drone Playground

Drone Playground is a JAX quadrotor research platform built on Crazyflow. It
supports Tracking, Racing, Static Obstacle Avoidance, and Dynamic Obstacle
Avoidance with BPTT, SHAC, and PPO. Perception policies use D435i depth or MID360
LiDAR; native EGO-Planner and SUPER run through independent ROS workers.

Use the [Command Builder](docs/command-builder.html) to generate Train, Play,
and Benchmark commands. [Release evidence](docs/validation.md) separates actual
execution checks from policy convergence. This development version has not
completed release acceptance.

## Start

Install [Pixi](https://pixi.sh), then run from this checkout:

```bash
pixi install --locked
pixi run autoresearch setup
pixi run train experiment=tracking learning=bptt seed=0
pixi run play experiment=tracking learning=bptt checkpoint=latest
pixi run benchmark experiment=tracking learning=bptt checkpoint=latest
```

Training runs through Autoresearch Gym and writes its run directory to the final
summary. Navigation training samples independent obstacle layouts; fixed S/D
maps are used only for evaluation.

```bash
pixi run train experiment=oa_static_depth learning=ppo seed=0
pixi run train experiment=oa_dynamic_lidar learning=shac seed=0
pixi run play experiment=oa_static_depth learning=ppo checkpoint=best \
  scene=S01 'benchmark.scenes=[S01]' simulation.num_envs=1
```

`latest` selects the latest published update in the newest compatible run.
`best` selects its recorded highest mean Episode Reward when optional selection
is enabled; otherwise it falls back to latest and records that fallback. An
explicit Orbax model directory selects a particular policy. Only Orbax
checkpoints can be loaded or resumed by this version.

See [installation](docs/installation.md) for GPU, CPU, acados, and ROS setup;
[training](docs/training.md) for budgets and resumption; and
[Autoresearch](docs/autoresearch.md) for controlled candidate search.

## Architecture

```mermaid
flowchart LR
  A[Autoresearch] --> L[Learning]
  A --> B[Benchmark]
  L --> S[Simulation]
  B --> S
  S --> C[Crazyflow]
  S --> N[Native planners and controllers]
```

Learning owns updates, optimizer state, and checkpoints. Simulation owns tasks,
geometry, sensors, methods, physical execution, and replay. Benchmark owns fixed
flight cases, measurements, and reports. Training saves numbered checkpoints;
optional fixed-reward evaluation for Best runs inside Learning, not Benchmark.

`play` runs a selected flight for inspection and optional replay. `benchmark`
runs a declared suite of paired flight cases for every method and produces
complete comparison reports. Both use the same Simulation execution path.

Benchmark reports success rate over every episode and flight-quality metrics
over successful episodes, with contributing counts. It follows FlightBench's
speed, acceleration, jerk, and curvature definitions and adds AvoidBench-inspired
static path excess and Traversability. See the [measurement protocol](docs/benchmark.md).

## Supported combinations

| Task | Policy observation | Algorithms | Native methods |
|---|---|---|---|
| Tracking | State | BPTT, SHAC, PPO | Sampling MPC, Attitude MPC |
| Racing | State | BPTT, SHAC, PPO | — |
| Static OA | Depth or LiDAR | BPTT, SHAC, PPO | EGO + MPC, SUPER + MPC |
| Dynamic OA | Depth or LiDAR | BPTT, SHAC, PPO | EGO + MPC, SUPER + MPC |

BPTT and SHAC can use native Crazyflow derivatives, LOTF, or point-mass-lag
backward dynamics. The forward plant remains Crazyflow. The builder selects
compatible action and loss settings. New physics backends, learning libraries,
and scene families are outside this release.

## Development

```bash
pixi run lint
JAX_PLATFORMS=cpu pixi run test
pixi run build
```

The [documentation index](docs/README.md), [domain glossary](GLOSSARY.md), and
[architecture decisions](docs/adr/0026-training-benchmark-release-separation.md)
describe the current interfaces. Runtime data stay under ignored `results/`;
historical weights and reports retain their original source and protocol.
Upstream licenses and asset attribution are distributed with the package.
