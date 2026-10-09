# ROS Planner worker: EGO-Planner and SUPER

The worker is a ROS1 Noetic C++ gRPC server. It compiles and calls EGO's
`EGOPlannerManager::reboundReplan` (A*, rebound LBFGS B-spline optimization and
feasibility refinement) and SUPER's `SuperPlanner::PlanFromRest` / `ReplanOnce`
(ROGMap, corridor generation, exploration trajectory and backup trajectory
optimization). The Python module imports no simulation or learning implementation.
There is no waypoint fallback.

## Build and run

From the repository root:

```bash
docker build --build-arg BUILD_JOBS=2 -t drone-playground-ros-planner-worker:noetic \
  src/ros_planner_worker
docker run --rm -d --name dp-ros-planner -p 127.0.0.1:50051:50051 \
  drone-playground-ros-planner-worker:noetic
pixi run python src/ros_planner_worker/integration.py --address 127.0.0.1:50051 --planner both
pixi run python -m pytest -q tests/test_ros_planner.py
```

A worker owns its ROS master, clock and one active planner session. Start a second
container on another localhost port for another concurrent client. `Close` releases
the session; the worker remains available for another `Initialize`. Containers
need neither a host ROS master nor host networking, GPU, privileged flags, or map
mounts. Stop only the container you own: `docker stop dp-ros-planner`.

Python client and generated wire modules live together in
`src/drone_playground/simulation/ros_planner/`. The independent C++/ROS build lives
beside the Python package in `src/ros_planner_worker/`. Generated `planner_pb2*.py`
files remain separate so regeneration does not overwrite the handwritten client.
The renamed `RosPlanner` RPC requires rebuilding the worker; earlier image tags
and flight hashes below describe the recorded runs before this naming change.

C++ adapters use `src/ros_planner_worker/.clang-format` with `clang-format-10`.
The Dockerfile pins the ROS base image digest and these upstream commits:

| Upstream | Commit |
|---|---|
| [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner) | `bfda51284c8c1b476043255a8145ef925a3778a5` |
| [SUPER](https://github.com/hku-mars/SUPER) | `2ad3419c127a617c6d7df6925e81a14175a9c096` |

It fetches clean upstream source, generates Protobuf C++ and EGO's ROS message
headers, then compiles with CMake. Ubuntu/ROS binary dependencies come from the
Focal/Noetic apt repositories; their package versions are not snapshot-pinned.
The supplied local dependency image can be used to avoid downloading the full ROS/PCL
dependency stack: add `--build-arg BASE_IMAGE=drone-playground-ros1:noetic`.
Both builds fetch and compile the same pinned upstream sources and new adapters.
Adapters are authored in `src/ros_planner_worker/src`; the base image supplies dependency libraries.
The upstream license files remain in `/opt/upstream` in the image.

Regenerate the checked-in Python wire modules after changing the proto:

```bash
pixi run bash src/ros_planner_worker/scripts/generate-python.sh
```

The current generated modules use grpcio-tools 1.84.0 / Protobuf compiler 7.35.1,
matching the project Pixi dependencies. C++ uses Ubuntu's Protobuf 3.6.1 / gRPC
1.16.1; the proto3 wire contract is compatible across these runtimes.

## Python API

```python
from drone_playground.simulation.ros_planner import RosPlanner, cloud_measurement

with RosPlanner("ego", "127.0.0.1:50051", timeout=5.0,
                   replan_interval=0.1) as planner:
    measurement = cloud_measurement(
        points_sensor, acquisition_time,
        available_time=available_time,
        position=sensor_position_world,
        quaternion=sensor_orientation_wxyz,
    )
    trajectory = planner({
        "position": position_world,
        "velocity": velocity_world,
        "acceleration": acceleration_world,
        "quaternion": orientation_wxyz,
        "goal": goal_world,
        "measurement": measurement,
    }, simulation_time)
    position, velocity, acceleration = trajectory.sample(simulation_time)
    planner.reset()  # Clears map and trajectory, permits time to restart at zero.
```

`RosPlanner(planner, address, *, timeout=5.0, replan_interval=0.1,
max_velocity=3.0, max_acceleration=4.0, sample_dt=0.05,
max_sensor_age=0.25, robot_radius=0.15)` uses seconds, metres, m/s and m/s².
`planner.plan(observation, time, force_replan=False)` and `planner(...)` are
identical. All vectors are unbatched, shape `(3,)`; quaternions are `(w,x,y,z)`.
Crazyflow callers must convert its quaternion ordering explicitly.
`goal_yaw` is optional and in radians. `reset(seed=0)` accepts the common planner
signature; the seed does not reseed EGO's upstream randomized recovery initialization.

Only the listed observation fields are serialized. `cloud_measurement` accepts
finite hit endpoints in the sensor frame. For world endpoints, subtract the sensor
position and use its world translation with identity rotation. ROGMap needs the
real ray origin. Deskew scanning LiDAR returns to that one sensor pose before
calling. Filter invalid/no-return cloud points using the measurement mask.

`depth_measurement(depth, time, *, fx, fy, cx, cy, position, quaternion,
available_time=None)` accepts an `H×W` image of optical **+z depth** in metres,
with zero for no return. Intrinsics use pixel coordinates `(u,v)`, right and down;
the pose transforms this optical frame to world. The C++ service projects valid
depth pixels and passes the resulting measured points through the same upstream
mapping paths. A valid frame with no hits is accepted as an empty measurement;
missing measurements, malformed depth and nonfinite cloud points still fail validation.

`RosTrajectory` exposes `.times`, `.positions`, `.velocities`,
`.accelerations`, `.valid_from`, `.valid_until`, `.status`, `.upstream_status`,
`.sequence`, `.timings` and `.sample(time)`. Arrays contain upstream trajectory
samples and actual analytic derivatives. The host's sample method interpolates
these samples; it raises outside their validity interval. Timing fields are
`mapping_ms`, `solve_ms` and `total_ms`, measured by C++ steady clock. Total
includes worker queue time, not network round-trip latency.

`RosPlannerError.status` identifies `NO_SOLUTION`, `INVALID_INPUT`,
`STALE_INPUT`, `STALE_OUTPUT`, `INVALID_OUTPUT`, transport status codes, or
`RESET_REQUIRED`. A failed decision invalidates cached output. A timeout requires
`reset()` before another solve; upstream C++ solvers are not preemptible, so reset
can wait for the timed-out solve to finish. There is no retry with old output.
Replan scheduling reuses a trajectory only within the configured interval and its
validity. A changed goal or `force_replan=True` requests a new native decision.

For lower-level callers, `decide(state, goal, measurement)` accepts the generated
`planner_pb2.State`, `.Goal`, `.Measurement` and always performs an RPC.
The generated stub also exposes all four RPCs directly.

## Wire contract and execution

The source of truth is
[`planner.proto`](../src/ros_planner_worker/proto/drone_playground/simulation/ros_planner/planner.proto).
The fully qualified service is `drone_playground.ros_planner.v1.RosPlanner`:

| RPC | Request → response | Effect |
|---|---|---|
| `Initialize` | `InitializeRequest → Session` | Select planner and motion/sampling limits; build its map and optimizer |
| `Reset` | `Session → Session` | Rebuild both, increment episode, clear sequence and clock |
| `Decide` | `DecideRequest → Decision` | Validate state/goal/sensor timestamps, ingest measurements, solve, sample native PVA |
| `Close` | `Session → Closed` | Release native objects and permit another session |

World is right-handed ENU, body FLU, all values SI. State, goal and measurement
carry simulation timestamps as signed nanoseconds. A measurement separately
carries acquisition and availability time. The service requires
`acquisition ≤ availability ≤ state time` and enforces `max_sensor_age`.
Sequences increase within an episode and state time never goes backward.
The response echoes session/episode, sequence and input time, provides status,
raw upstream result code, validity and absolute timestamped PVA samples.
EGO raw codes are 1 (success), 0 (failure); SUPER uses 2 (success), 1 (no new
trajectory needed; committed trajectory still valid), 0 (failure).
`SOLVED` is the only status carrying a trajectory. Invalid input and solver
failure never masquerade as a successful stationary command.

The worker fixes ROS time to request simulation time plus one second, to avoid
ROS's zero-time sentinel; this internal offset is removed from output.
Requests are serialized because upstream clocks and some EGO state are global.
No asynchronous ROS callbacks run: mapping completes before optimization starts.

## Port changes and scope

Six small, checked-in upstream patches are applied only to build copies:

- EGO exposes synchronous ingestion using its existing odometry/cloud callbacks.
- ROGMap initialization and first-frame state are per instance, enabling clean episode resets.
- EGO and SUPER release their A* allocations on destruction; repeated resets otherwise
  leak grid pointer arrays or nodes.
- EGO drops an unused gradient-descent header with invalid integer narrowing on GCC 9.
- SUPER initializes its first trajectory boundary with measured position, velocity
  and acceleration instead of snapping position and assuming rest. Its trajectory
  optimizers and subsequent committed-trajectory replanning remain upstream code.

EGO uses its upstream cloud occupancy/inflation mode, clearing and recentering a
30×30×12 m grid each decision and selecting a local goal at most 7 m away. If
that endpoint is occupied in the measured inflated grid, the adapter shortens it
along the goal direction in grid-resolution steps before calling `reboundReplan`.
This only selects an endpoint; the upstream optimizer still generates the path. Its
failed polynomial initialization gets one native randomized-polynomial retry,
matching the recovery step in EGO's `planFromCurrentTraj` state machine. Failure
of both native attempts remains `NO_SOLUTION`. The
upstream cloud path has no temporal fusion or unknown-space guarantee. SUPER uses
ROGMap raycasting with a sliding 30×30×12 m grid and a 14 m ray range; backup
trajectory generation remains enabled. SUPER uses a 1.5 m corridor bound to
avoid face-only corridor intersections with this voxel size. Sensor extrinsics are used for ray origins,
while vehicle state provides the planning boundary. Neither adapter loads a PCD,
scene mesh, obstacle list, nor a privileged map. Native configuration is in
`src/ros_planner_worker/src/ego.cpp` and `src/ros_planner_worker/config/super.yaml`.

The executable integration test verifies real solves, obstacle response, native
PVA consistency (including a moving initial state), repeated replanning, reset,
cloud/depth ingress, valid empty EGO clouds, occupied local EGO endpoints,
EGO's native recovery from a failed moving-state initialization,
malformed/stale-input rejection, and timeout/reset recovery.
Python unit tests use a labelled wire fixture only for transport and failure behavior.
They are not native-planner evidence. S6 flight acceptance requires separate static
and dynamic Crazyflow episodes with control, collision and task outcomes.

## Recorded verification

The 2026-10-09 renamed worker passed the real EGO/SUPER integration checks with
`drone_playground.ros_planner.v1.RosPlanner`. Its image identity, source hashes,
native samples and measured timings are saved in
[`ros-planner-integration-results.json`](../src/ros_planner_worker/ros-planner-integration-results.json).
The Python client, evaluation and recording subset also passed 45 tests under the
expanded Ruff rules. These new checks do not replace the earlier S6 flight records.

Both the pinned clean ROS-base build and the supplied dependency-base build passed
`src/ros_planner_worker/integration.py` for EGO and SUPER. The recorded fixture returned 47 samples
across 2.298 s for EGO and 59 samples across 2.900 s for SUPER, with sensed-box
avoidance and analytic derivative checks passing. Full image IDs and timings are
in [`integration-results.json`](../src/ros_planner_worker/integration-results.json). Six-reset RSS
readings are in [`memory-results.json`](../src/ros_planner_worker/memory-results.json). The initial
eight Python protocol/client tests passed. These results cover service integration only.

## EGO S6 flight acceptance

The 2026-10-08 full flight run passed S6 with **12 static and 12 dynamic episodes**,
four per required scene. Each static scene had safe arrivals. Dynamic outcomes
have no success-rate threshold. All failures remain in the denominator:

| Scene | Safe arrivals | Out of bounds | Native failure | Episodes |
|---|---:|---:|---:|---:|
| S01 | 3 | 1 | 0 | 4 |
| S02 | 4 | 0 | 0 | 4 |
| S03 | 3 | 1 | 0 | 4 |
| D01 | 4 | 0 | 0 | 4 |
| D02 | 4 | 0 | 0 | 4 |
| D03 | 1 | 2 | 1 | 4 |

The accepted recipe uses `navigation_depth`, one Crazyflow world, 500 Hz physics,
50 Hz physical control and **2.5 Hz native replanning**. Native speed/acceleration
remain 3 m/s and 4 m/s²; task collision radius remains 0.07 m. Sensor calibration,
assets, task rules and controller were not changed by this native work.
The run used frozen project Pixi dependencies (Python 3.12.15, JAX 0.11.2 CUDA).

Both image tags `drone-playground-native:clean-noetic` and `:noetic` were rebuilt
from the pinned clean ROS base and identify
`sha256:77376660dd85c6cf667d600fd07433f8d069161cb26966c971519cdf11d008d4`.
The flight worker was a fresh container with no workspace mounts. Start a fresh
worker for a repeatable full run: EGO's randomized recovery uses process-global
random state, which episode reset does not reseed.

```bash
docker run -d --name dp-ego-s6 -p 127.0.0.1:50066:50051 \
  drone-playground-ros-planner-worker:noetic
# Wait for "Native planner listening" in docker logs dp-ego-s6.
pixi run --frozen python -m drone_playground.cli \
  mode=benchmark experiment=navigation_depth method=ego \
  simulation.num_envs=1 simulation.method_hz=50 \
  method.address=127.0.0.1:50066 method.timeout=30 method.replan_hz=2.5 \
  'benchmark.scenes=[S01,S02,S03,D01,D02,D03]' benchmark.episodes=4 \
  output=results/ego_S6_reproduce_s0 benchmark.record_replay=true
```

The [S6 summary](../results/ego_S6_acceptance_s0/report.json) and
[evidence audit](../results/ego_S6_acceptance_s0/evidence-audit.json) record the
results. All 24 final replays match actual recorded poses and times; the final run
contains 403,705 physical steps and 2,047 native solve attempts. The audit also
checks all 36 earlier trial episodes, which remain in their original directories
and are indexed in [prior-runs.json](../results/ego_S6_acceptance_s0/prior-runs.json).
Across all 60 attempts there were 28 arrivals, 25 native failures and 7 bounds exits.

[Final image integration evidence](../src/ros_planner_worker/ego-flight-recovery-integration-results.json)
covers both native planners, including empty cloud/depth frames, occupied EGO
endpoints and the moving-state recovery regression. Nine Python client tests and
Ruff passed in the pinned project environment. The earlier image's
`src/ros_planner_worker/integration-results.json` was preserved.

The flight's exact image, configuration and launch-time source hashes are in
[native-build.json](../results/ego_S6_acceptance_s0/native-build.json) and `run.json`.
Changes to `runner.py` and `cli.py` after launch were not loaded by this run;
the recorded source identity describes the implementation that executed the flight.
EGO's latest-cloud map still has no temporal fusion or unknown-space guarantee.
The four bounds exits and one exhausted native solve are retained limitations.

## SUPER S6 flight acceptance

The 2026-10-08 full run passed S6 with 12 static and 12 dynamic episodes:

| Scene | Safe arrivals | Out of bounds | Native failure | Episodes |
|---|---:|---:|---:|---:|
| S01 | 3 | 0 | 1 | 4 |
| S02 | 4 | 0 | 0 | 4 |
| S03 | 2 | 0 | 2 | 4 |
| D01 | 4 | 0 | 0 | 4 |
| D02 | 2 | 2 | 0 | 4 |
| D03 | 2 | 0 | 2 | 4 |

The run used the full Mid-360 profile (20,000 rays/frame), one Crazyflow world,
500 Hz physics, 50 Hz control and 5 Hz native replanning. Its 24 episodes took
2,002.26 s wall time under concurrent GPU training. The
[S6 report](../results/super_S6_s0_v2/report.json) and
[flight audit](../results/super_S6_s0_v2/evidence-audit.json) retain all failures.
The audit verified 373,343 physical steps, 37,367 replay frames and 3,729 solved
decisions. Replay positions, timestamps and sensor hit counts exactly match the
recorded flight; normalized quaternions agree to 1e-12. Successful endpoints are
within the unchanged 0.5 m goal radius with positive clearance.

Native solve time median/P95 was 19.24/33.51 ms; total service time was
55.96/113.92 ms. These are measured service times for this concurrent workload.
The worker had no workspace mounts and used image
`sha256:3d90fc7997431649e40cd5a028f11846ccf9d502281110cdfa39dd7d9d734d78`.
Its SUPER source and configuration match the current files. Its service still
rejected valid empty clouds; the later service change accepts them and passed
the separate clean-image integration test. This full flight was not rerun with
that newer service. Exact hashes are in the audit.

```bash
docker run -d --name dp-super-s6 -p 127.0.0.1:50067:50051 \
  drone-playground-ros-planner-worker:noetic
# Wait for "Native planner listening" in docker logs dp-super-s6.
pixi run --frozen drone-playground \
  mode=benchmark experiment=navigation_lidar method=super \
  simulation.num_envs=1 simulation.method_hz=50 \
  method.address=127.0.0.1:50067 method.timeout=30 method.replan_hz=5 \
  'benchmark.scenes=[S01,S02,S03,D01,D02,D03]' benchmark.episodes=4 \
  output=results/super_S6_reproduce_s0 benchmark.record_replay=true
```

This command uses the current clean image. The earlier interrupted run remains
in `results/super_S6_s0`; its partial episodes are not included in the 24-episode
acceptance denominator.
