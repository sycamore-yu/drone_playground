# Algorithm RPC

This package is the transport layer for external algorithms: protocol messages,
serialization, lifecycle, deadlines, client/server behavior and process cleanup.
It does not implement a planner or controller.

The matching ROS-independent C++ SDK is in `native/sdk/`. Python deployment
adapters are in `drone_playground.integrations`; ROS1-specific deployment files
are in `native/ros1/`.

Executable physical values are defined by `references.py` and
`control/setpoints.py`. The Protobuf schema mirrors those explicit types and
the optional planning-inspection objects from `planning/corridors.py`.
