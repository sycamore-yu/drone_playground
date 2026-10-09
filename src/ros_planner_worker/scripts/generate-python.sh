#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m grpc_tools.protoc -I src/ros_planner_worker/proto --python_out=src --grpc_python_out=src \
  src/ros_planner_worker/proto/drone_playground/simulation/ros_planner/planner.proto
# Generated descriptors are intentionally not reformatted by Ruff.
python - <<'PYGEN'
from pathlib import Path
for path in Path("src/drone_playground/simulation/ros_planner").glob("planner_pb2*.py"):
    path.write_text("# ruff: noqa\n" + path.read_text())
PYGEN
