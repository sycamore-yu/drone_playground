# C++ algorithm SDK

This directory contains the ROS-independent SDK. It does not contain a second EGO-Planner or SUPER implementation.

The shared schema is `src/drone_playground/rpc/proto/algorithm.proto`. The current service is `drone.native.v2.Algorithm`, with protocol version 2. Initialize, reset, step and close preserve one session's identity and simulation clock.

`include/drone_native/algorithm.h` defines the native interface. `src/service.cc` validates and transports its outputs. `tests/interop_server.cc` is an interoperability fixture, not a planning baseline.

Physical outputs are Trajectory, Waypoint, StateSetpoint, AttitudeSetpoint, RateSetpoint, ForceTorque and MotorRPM. Each type has explicit fields and SI units. Python definitions are in `references.py` and `control/setpoints.py`. Equal vector widths do not imply compatible inputs.

SafeFlightCorridor and TrajectoryPreview are optional inspection outputs. Each has its own generation time and expiry. They do not certify safety and do not replace the executable output.

Build the SDK and fixture with the installed SDK environment:

```bash
prefix="$PWD/ros_integrations/sdk/.pixi/envs/default"
export PATH="$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S ros_integrations/sdk -B tmp/native-sdk -G Ninja \
  -DCMAKE_PREFIX_PATH="$prefix" -DDRONE_NATIVE_BUILD_TESTS=ON
cmake --build tmp/native-sdk -j 2
```

Rebuild a v1 client or service against the v2 schema before using it with current code. Historical frozen run bundles stay unchanged. ROS planner deployment is described in `../ros1/README.md`.
