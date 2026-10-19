# ROS-free native C++ planners

Four compiled C++ algorithms integrate into the existing `PlannerController`
and time-valid `Trajectory` interfaces. The physics engine, `train`, `play`,
`benchmark`, and controller selection are unchanged.
Native solvers execute on the CPU host; differentiable JAX training does not
backpropagate through them.

## Build

Build from this checkout in the Python 3.12 environment that runs Drone Playground:

```bash
pixi install --locked
pixi run bash -c 'PYTHON="$(command -v python)" bash tools/setup_native_planners.sh'
```

The script builds with CMake and pybind11 3.1.0 against the *active Python
interpreter* and places imports under `.cache/native_planners/python/`.
`native_planners.import_native()` locates these automatically in a source
checkout. No ROS master, ROS daemon, running worker or gRPC connection is needed.
The pinned third-party sources are fetched into `.cache/native_planners/src/`;
changes are applied from tracked patches, not to the original research checkouts.
A local build can be limited to one algorithm, e.g. `bash tools/setup_native_planners.sh gcopter`.

| Python method | Native algorithm / implementation | Core sources pinned | Requirements |
|---|---|---|---|
| `gcopter` | GCOPTER MINCO + L-BFGS via `gcopterpy` | `aerostack2/gcopter_trajectory_generator_lib@4a1b4c0`, GCOPTER submodule | Ordered world-frame path or known Racing course |
| `ego_native` | A* + rebound B-spline C++ | `chan-yuu/ego_planner_standalone@5e7aa7c`, with boundary patch | Measured Depth or LiDAR hits |
| `super_native` | SUPER exploration + backup trajectory, ROG-Map | `zxh0916/super_planner@62ebba5`, with sensing and boundary patches | Measured Depth or LiDAR hits |
| `mighty_native` | MIT ACL HGP + DecompUtil convex corridors + quintic Hermite/L-BFGS | `mit-acl/mighty@1639402`, `kotakondo/DecompROS2@a2bc42e` | Measured world-frame hit cloud; no predicted moving-agent trajectories |

The original `ego` and `super` methods remain available as the ROS/gRPC
reference implementations. The native EGO implementation uses a third-party
ROS-free fork with some different planner details; evaluate the output and
closed-loop task performance separately before claiming numerical equivalence.

## Run

```bash
# Native C++ optimizer on the publicly known Racing course
pixi run play task=racing scene=racing sensor=state method=gcopter controller=so3 \
  simulation.num_envs=1

# Native map and planning cores with measured onboard sensors
pixi run play task=navigation scene=S01 sensor=depth method=ego_native controller=so3 \
  simulation.num_envs=1
pixi run play task=navigation scene=S01 sensor=lidar method=super_native controller=so3 \
  simulation.num_envs=1
pixi run play task=navigation scene=S01 sensor=lidar method=mighty_native controller=so3 \
  simulation.num_envs=1
```

Use `mode=benchmark` with the same options and fixed scene cases for paired
measurements. `controller=full_state_nmpc` is another compatible trajectory
tracker if acados is installed. EGO/SUPER and MIGHTY use only measured
observations; they cannot read the simulator's privileged scene mesh.
GCOPTER is a trajectory optimizer: it needs externally supplied collision-free
waypoints and corridor bounds in obstacle scenes, so its default task preset
is Racing rather than unknown-map obstacle avoidance.

## Interface and algorithm provenance

Each planner receives measured state `(p,v,a)`, goal, current simulation time,
and, where required, a timestamped measured sensor frame. Point cloud hits are
masked in the body frame and transformed using the acquisition pose before
entering C++. The trajectory returns absolute times and analytic position,
velocity and acceleration. A planner failure terminates that method execution;
it never returns a fabricated successful trajectory.

SUPER retains its real ROG map and backup trajectory decisions. MIGHTY uses
original HGP voxel search, DecompUtil spatial corridors, and Hermite/L-BFGS.
The current MIGHTY adapter uses 3D static observations; prediction and shared
trajectories for other moving agents are separate capabilities of upstream
MIGHTY, not yet part of the single-drone simulator interface. The original
unmodified MIGHTY code also provides richer exploration behavior than this
planning adapter.

An EGO replan recenters occupancy and reuses the A* / optimizer allocations
within an episode. Reset reconstructs its native map and optimizer. The EGO
standalone fork's trajectory boundary tolerance and obstacle-specific optimizer
failures are measured in native tests; successful import or solver return is
not sufficient to establish navigation success.

## MAP and safe corridor visualization

With `benchmark.record_replay=true`, EGO and SUPER record their native
inflated occupancy maps; MIGHTY records occupied voxels from its HGP map.
SUPER exports the exploration safe flight corridor (SFC), and MIGHTY exports
the DecompUtil convex corridor. The viewer derives the visible wireframe
from the original halfspace planes. Sensor hits remain a separate layer.

Both RScope and MuJoCo WASM replay these layers at their recorded times.
WASM has separate **Sensor hits**, **MAP**, and **SFC** switches. The displayed
map is sampled to at most 128 occupied voxels, and SFC to at most 256 edges.
Capture is enabled only when the replay option is set, so benchmark runs
without recording retain their normal planning cost. The overlays use the
planner's measured map, never the simulator's privileged world geometry.

Example fixed-scene replay:

```bash
pixi run benchmark task=navigation scene=S01 sensor=lidar \
  method=mighty_native controller=so3 simulation.num_envs=1 \
  'benchmark.scenes=[S01]' benchmark.episodes=1 \
  task.duration=90.0 benchmark.record_replay=true benchmark.replay_episodes=1 \
  benchmark.replay_max_frames=900 \
  output=results/native-mighty-s01
cd docs/viewer && npm ci && npm run build && cd ../..
python3 -m http.server 4177 --bind 127.0.0.1
```

Forward port 4177 through SSH. Open
`http://127.0.0.1:4177/docs/viewer/dist/index.html` and load
`/results/native-mighty-s01/replays/S01-0000/browser_replay.json`.
Existing replays without SFC/MAP remain compatible.

In paired 5-second S01 runs using the same seed (2000000), SO3 controller,
and ROS worker ports 50151/50152, the recorded per-planning-call mean/P95
was 10.21/12.52 ms for original EGO versus 2.39/2.99 ms for native EGO;
33.06/41.40 ms for original SUPER versus 22.46/25.19 ms for native SUPER;
and 28.03/31.45 ms for MIGHTY. These original 5-second integration runs all
ended at TIMEOUT; their artifacts remain under `tmp/native-overlay-benchmark-*/`.
The comparison gallery now uses the full-duration records listed below.

Full-duration S01 recordings use the same initialization seed (2000000) and
SO3 controller, with a 90-second task limit and 900 evenly sampled WASM frames.
The frame cap preserves the complete time span and terminal event:

The complete Benchmark artifacts on the development machine are stored under
`results/native-planner-full-episodes-20261018/{ego,ego_native,super,super_native,mighty,mighty_fixed}/`.
Each method directory contains `episodes.csv`, `report.json`, planning
decisions, the RScope `.mj_unroll`, and the browser replay with scene assets.
These results are local, ignored artifacts and are not included in Git.
Regenerate the comparison page with `python3 tools/build_native_gallery.py`;
it writes `tmp/native-benchmark-gallery/index.html` from these retained runs.

| Method | Outcome | Flight time | Path length | Minimum clearance | MAP / SFC slots |
|---|---|---:|---:|---:|---:|
| EGO · ROS | SUCCESS | 36.168 s | 96.712 m | 0.396 m | 0 / 0 |
| EGO · pybind11 | SUCCESS | 69.966 s | 101.269 m | 0.418 m | 128 / 0 |
| SUPER · ROS | SUCCESS | 33.584 s | 96.958 m | 0.522 m | 0 / 0 |
| SUPER · pybind11 | SUCCESS | 36.998 s | 97.753 m | 1.170 m | 128 / 57 |
| MIGHTY · before fix | METHOD_FAILURE | 25.000 s | 51.742 m | 0.840 m | 128 / 72 |
| MIGHTY · pybind11 (fixed) | SUCCESS | 44.572 s | 97.088 m | 0.678 m | 128 / 87 |

The old MIGHTY adapter projected a 7 m local goal directly into an occupied
voxel at 25.0 s. The ROS-free map path had omitted the upstream map-initialized
flag, preventing its nearest-non-occupied-voxel recovery. Its 0.15 m planning
radius also produced zero inflation cells at 0.4 m map resolution. The fixed
adapter initializes the map, checks both occupied voxels and original sensor
hit clearance, relocates blocked local goals, and retries shorter or lateral
goals when the continuous Hermite trajectory fails its collision check. Its
default planning radius is 0.4 m, including discretization and tracking
margin. The new MIGHTY completed the original S01 seed in 44.572 s with
0.678 m minimum clearance, while the old 25.0 s failure recording remains
available for comparison. These are single-seed results.
Replay-enabled runs also collect MAP/SFC geometry, so their latency figures
include visualization-data extraction and should not be used for a definitive
speed ranking. Use matched seeds across multiple scenes for success-rate and
performance claims.

## Test and acceptance

```bash
pixi run python -m pytest -q tests/test_native_cpp_planners.py \
    tests/test_native_planner_horizon.py
```

The integration tests execute actual compiled solvers. Static geometry tests
exercise HGP + corridor planning with measured obstacle points. Validation on
2026-10-18 with the existing Crazyflow forward plant:

- All four native pybind11 extensions rebuilt from their pinned sources with
  Python 3.12, NumPy 2.5 and pybind11 3.1.
- `tests/test_native_cpp_planners.py` passed all 14 tests against the freshly
  built extensions, including real C++ optimizers and two real replans.
- Earlier short smoke runs covered a 3-second GCOPTER Racing episode and
  five-second EGO/SUPER/MIGHTY S01 episodes. They ended at TIMEOUT and are
  integration checks, not task-arrival evidence.
- Focused regression run: 75 passed, 1 skipped, and 1 deselected. It includes
  native planner, replay, timing, controller, ROS-planner and evaluation tests.

Task-level acceptance currently covers one full-duration S01 seed: original and
native EGO/SUPER and the fixed MIGHTY adapter reached the goal. GCOPTER completed
its integration run without a planner exception, but that run ended at the
three-second task timeout and does not establish successful Racing arrival.
Multi-scene, multi-seed acceptance remains open. The Command Builder catalog
stays unchanged until broader closed-loop acceptance is complete.

## Source licenses

- GCOPTER original source: MIT; `gcopter_trajectory_generator_lib`: BSD-3-Clause.
- EGO-Planner original code and ROS-free derivative: GPLv3 obligations apply.
- SUPER: LGPL-3.0-or-later notices in its source.
- MIT ACL MIGHTY and pinned DecompUtil: BSD-3-Clause.

The build caches third-party sources outside Git and does not add their
source or binary extensions to this repository. Before distributing a wheel or
binary with EGO, preserve its GPL source and corresponding redistribution
requirements.
