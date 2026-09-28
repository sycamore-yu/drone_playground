#!/usr/bin/env bash
# Compatibility entry point; the native ROS runtime is owned by native_planners/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/native_planners/setup.sh" "$@"
