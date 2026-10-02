# Algorithm RPC

This package implements gRPC communication with external Python or C++ algorithms: lifecycle, messages, serialization, deadlines, and process cleanup. The matching C++ SDK is at `ros_integrations/sdk/`.

Physical commands are defined in `actions/commands.py`; they do not belong to the transport. The Protobuf wire schema stays under `proto/`. The migration changes Python imports but preserves the on-wire package, fields, command units, and timestamps.
