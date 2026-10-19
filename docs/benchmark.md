# Benchmark measurement

Benchmark evaluates a frozen policy, controller, or native planner on a declared
suite. `play` selects an individual scenario for inspection and optional replay.
Both use the same execution and reporting machinery, but only Benchmark is
intended for method-to-method quantitative comparison. Play omits the separate
latency and environmental-complexity sweeps to run an individual flight quickly.

```bash
pixi run benchmark experiment=oa_static_depth learning=ppo checkpoint=latest
pixi run benchmark experiment=oa_dynamic_lidar learning=shac checkpoint=best
pixi run play experiment=oa_static_depth learning=ppo checkpoint=latest \
  scene=S01 'benchmark.scenes=[S01]' simulation.num_envs=1
```

Static OA contains S01, S02, S03, and S06. Dynamic OA contains D01, D02, D03, and
D06. S06/D06 remain separately identifiable extension cases. Geometry and motion
laws are unchanged. Canonical scene offsets keep seed identities stable when a
suite is evaluated in subsets. For each scene, the initial-state batch is sampled
once. Vectorized JAX policies execute these worlds in parallel; native planners
use the identical initial states serially. Each episode records a `case_id`.
`benchmark.seed_base` defaults to 2,000,000;
Autoresearch search evaluation defaults to a distinct 1,600,000 cohort.

### Additional native-MJCF task families

The optional `forest_*`, `maze_*`, `dense_*`, and `dynamic_dense_*` scenes each
provide Easy/Medium/Hard difficulty. They use the same task bounds, physical
collision conditions, Depth/LiDAR geometry, and pairwise benchmark case IDs
as the core maps. Select them via `scene=` for Play or `benchmark.scenes=` for
quantitative comparison:

```bash
pixi run play task=navigation sensor=state method=controller \
  scene=forest_easy 'benchmark.scenes=[forest_easy]' simulation.num_envs=1
pixi run benchmark experiment=oa_static_depth learning=ppo checkpoint=best \
  'benchmark.scenes=[forest_easy,forest_medium,forest_hard]'
```

These assets reproduce task *types*, not the original FlightBench maps or their
reported results. The default S/D suite remains unchanged.

## Metrics

Success rate includes all episodes. The other flight metrics average successful
episodes and record a contributing count for each field. Missing or undefined
measurements are null, including zero-length paths and insufficient derivative
samples. The framework does not impose aggregate quality thresholds.

| Metric | Definition | Unit |
|---|---|---|
| Success rate | Successful / requested episodes | fraction |
| Average speed | Path length / flight time | m/s |
| Average acceleration | Integral of squared acceleration / path length | m/s³ |
| Average jerk | Integral of squared jerk / path length | m/s⁵ |
| Average curvature | Distance-weighted curvature / path length | 1/m |
| Static path excess | 100 × (flown length / reference length − 1) | % |

Motion metrics follow [FlightBench v3](https://arxiv.org/html/2406.05687v3#S3.SS3).
Positions and velocities are recorded in the world frame at the method clock;
central differences of velocity give acceleration and jerk. Only samples up to
the recorded terminal event contribute.

Static path excess follows the collision-free-reference principle in
[AvoidBench](https://arxiv.org/pdf/2301.07430). The implementation uses a
conservative 26-connected A* grid at 0.5 m resolution, with the Task's sphere
radius and the actual initial/terminal positions. It checks continuous segment
clearance. Unreachable grid endpoints produce an unavailable reference. The grid
path is approximate, so a shorter valid flown path can produce negative excess.

Traversability uses a fixed 20 × 8 × 3 interior position grid and sixteen
horizontal directions, with rays capped at 40 m and bounded by the task region.
It divides mean free range by vehicle diameter. Static scenes use time zero;
dynamic scenes use 0, 5, 10, and 15 seconds. Reports retain sampling and valid-ray
counts. This is a declared environmental measurement, not a task success score.

## Computation time

A separate one-world probe measures decision calls after warm-up with device
synchronization. Computation includes observation encoding or native mapping,
policy/planner execution, and setpoint control. Physics, sensor synthesis, and
inner flight-controller ticks are reported in the Simulation measurement.
Native transport/RPC overhead is reported separately from worker computation.
Each field includes mean, 95th percentile, and sample count. Compare latency
only under matching hardware and execution conditions; parallel smoke runs are
not performance benchmarks.

## Reports and replay

`report.json` records per-scene and suite metrics, provenance, protocol, and
completion. `episodes.csv` includes every outcome. `failures.csv` and
`trajectories/failed/` preserve failed episodes separately. Set
`benchmark.record_trajectories=true` to retain all trajectories or
`benchmark.record_replay=true` to export RScope playback. An interrupted report
has `complete: false` and cannot stand in for a complete cohort.

`latest` selects the newest compatible run and its highest committed Actor
update. Compatibility includes task, sensor, algorithm, and static/dynamic
training distribution. `best` selects the highest mean episode return recorded
within that run when enabled, and otherwise falls back to latest. Explicit paths
identify a particular checkpoint, including retained
Actors whose full training states were pruned. Playback restores the saved Actor,
action, sensor, and physical contracts.
