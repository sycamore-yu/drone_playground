# Native algorithm SDK

The simulator owns state, time and physics. A native service receives an atomic decision request and returns one physical output: a full `Trajectory`, ordered `Waypoint`, or a named `MotionCommand`. Python talks directly to gRPC. C++ algorithms implement `drone_native::Algorithm`; no ROS, MuJoCo or JAX dependency is required by the SDK.

The pattern is informed by [MuJoCo MPC's Python agent](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/python/mujoco_mpc/agent.py) and [service schema](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/grpc/agent.proto). This schema belongs to Drone Playground: `Step` atomically carries state/sensor/goal instead of mutating simulator state through several RPCs.

From the repository root:

```bash
pixi install --locked --manifest-path native/pixi.toml
pixi run --manifest-path native/pixi.toml cmake -S native -B tmp/native-sdk -G Ninja -DDRONE_NATIVE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
pixi run --manifest-path native/pixi.toml cmake --build tmp/native-sdk -j4
JAX_PLATFORMS=cpu pixi run test tests/test_native_contracts.py tests/test_native_grpc.py
```

The build uses its own locked C++ environment. `interop_server` is an analytic protocol test fixture, not a flight algorithm or benchmark baseline. To regenerate the checked-in Python protobuf module after changing the schema:

```bash
pixi run --manifest-path native/pixi.toml protoc -I src/drone_playground/native/proto -I native/.pixi/envs/default/include --python_out=src/drone_playground/native/proto src/drone_playground/native/proto/algorithm.proto
```

Implement `Initialize`, `Reset`, `Step`, and optional `Close` in a C++ subclass, link `drone_native`, construct `Service`, and call `StartServer`. See `tests/interop_server.cc` for a compilable example. Place algorithm parameters in `InitializeRequest.parameters`; keep algorithm source/build dependencies outside the simulator environment. Record the algorithm's source revision, parameters, executable hash and deployment identity with the run. The fixture does not implement MIGHTY, SANDO or any other named research method.

```python
from drone_playground.native.client import NativeClient
from drone_playground.native.contracts import Waypoint

with NativeClient("my_algorithm", command=["/path/to/server", "{address}"]) as client:
    client.reset(goal=Waypoint([[10, 0, 2]], tolerance=0.5))
    result = client.step(time=0.0, state={
        "position": [0, 0, 2], "velocity": [0, 0, 0],
        "quaternion": [0, 0, 0, 1], "angular_velocity": [0, 0, 0],
    })
```

An owned local process defaults to a Unix socket in a private temporary directory. `address=` connects to an existing service without owning that process. An explicit `command` plus `address` supports an owned process in an isolated container. These are local research transports; do not expose their unauthenticated endpoints to an untrusted network. The ROS adapter uses the container's private bridge without published host ports.

Each response must echo the exact protocol/session/episode/sequence/time header. Simulation time is nonnegative and monotone within an episode. Reset uses a fresh episode identity and clears algorithm/map state. RPC timeout faults the Python session; close and create a new client before further decisions. An algorithm must observe its solve budget; the RPC deadline bounds how long the simulator waits. It cannot forcibly interrupt arbitrary C++ solver code. `NO_PLAN`, `INFEASIBLE` and `BUDGET_EXHAUSTED` contain no executable output. An opaque native service advertises `derivatives=none`.

Trajectory coefficients are world xyz/yaw, ascending powers of local segment seconds, SI units. Each segment has its own duration. Queries outside the finite interval fail. `yaw_defined=false` delegates heading to an explicit downstream policy. EGO's positional B-spline is converted analytically; its original causal yaw remains in the optional timestamped execution sample. SUPER's position/yaw polynomials are split at the union of their knots without resampling. A current execution sample is never a replacement for an MPC horizon.

To run another trajectory-producing service through public navigation evaluation, use `method=native`, set `method.algorithm`, and set either `method.deployment.command` (argv list) or `method.deployment.address`. The service must accept the declared sensor: `method.input_sensor=point_cloud` with the default lidar environment, or `depth` with compatible depth sensor and observation overrides. Algorithms requiring other inputs need an explicit input adapter; the simulator does not invent missing measurements. The low-level SDK supports all three physical outputs; this first public external-method evaluation entry executes full trajectories.

A planner's downstream controller is configured independently:

```bash
JAX_PLATFORMS=cuda,cpu pixi run eval method=paper/super env=navigation/static controller@env.execution.tracker=sampling_mpc evaluation.split=dev evaluation.episodes=1
```

Choose `attitude_mpc` for acados (see `docs/runbook.md`). The default `trajectory_tracking` consumes the original ROS execution sample for reproduction. MPC variants consume the full curve and are component-comparison results. They count missing/insufficient horizons and use the declared hold fallback; such fallback is not planner success. ROS services snapshot the entire adapter/protocol bundle per run and close their private planner and master; an idle owner lease bounds orphan lifetime.
