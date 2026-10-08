# Algorithm transport

`proto/algorithm.proto` is the single cross-language contract. The service remains
`drone.native.v2.Algorithm`; Python package relocation does not change wire field
numbers, service names, physical meanings or protocol version.

The client owns local process lifecycle or connects to a configured address. It
checks sessions, episode resets, clocks, capabilities, values and deadlines. The
wire decoder returns `runtime.Decision` without another adapter dictionary.

The sibling `ros1` package adapts actual ROS planners. The C++ fixture in
`tests/native/interop_server.cc` directly implements the generated gRPC interface.
