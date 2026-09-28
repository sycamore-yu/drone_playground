#!/usr/bin/env bash
# Project-managed native ROS runtime setup.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/native_planners/setup.sh" "$@"
